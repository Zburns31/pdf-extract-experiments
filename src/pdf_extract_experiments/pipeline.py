from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_ROOT, DEFAULT_PDF_PATH


class IngestionError(RuntimeError):
    pass


class IncompleteRunError(IngestionError):
    def __init__(self, run_dir: Path, results: list[dict[str, Any]]) -> None:
        self.run_dir = run_dir
        self.results = results
        super().__init__(
            "Ingestion wrote artifacts but could not complete every method. "
            f"See {run_dir / 'run_manifest.json'} for details."
        )


@dataclass(frozen=True)
class MethodConfig:
    method_id: str
    description: str
    backend: str = "opendataloader"
    output_kind: str = "document"
    convert_options: dict[str, Any] = field(default_factory=dict)
    requires_java: bool = False
    requires_hybrid_backend: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    pdf_path: Path = DEFAULT_PDF_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    overwrite: bool = False


AVAILABLE_METHODS: dict[str, MethodConfig] = {
    "text": MethodConfig(
        method_id="text",
        description="OpenDataLoader local extraction using embedded PDF text and layout analysis.",
        backend="opendataloader",
        output_kind="document",
        convert_options={"format": "json,markdown"},
        requires_java=True,
    ),
    "ocr": MethodConfig(
        method_id="ocr",
        description="OCR-oriented extraction using OpenDataLoader hybrid mode.",
        backend="opendataloader",
        output_kind="document",
        convert_options={"format": "json,markdown", "hybrid": "docling-fast"},
        requires_java=True,
        requires_hybrid_backend=True,
    ),
    "pymupdf_text": MethodConfig(
        method_id="pymupdf_text",
        description="Whole-document extraction using PyMuPDF word-level text and bounding boxes.",
        backend="pymupdf",
        output_kind="document",
    ),
    "pdfplumber_text": MethodConfig(
        method_id="pdfplumber_text",
        description="Whole-document extraction using pdfplumber word-level layout data.",
        backend="pdfplumber",
        output_kind="document",
    ),
    "camelot_table": MethodConfig(
        method_id="camelot_table",
        description="Table-only extraction using Camelot for text-based PDFs.",
        backend="camelot",
        output_kind="table",
        convert_options={"pages": "all", "flavor": "lattice"},
    ),
}


def list_methods() -> list[MethodConfig]:
    return [AVAILABLE_METHODS[key] for key in sorted(AVAILABLE_METHODS)]


def resolve_methods(selection: str) -> list[MethodConfig]:
    if selection == "all":
        return list_methods()
    try:
        return [AVAILABLE_METHODS[selection]]
    except KeyError as exc:
        raise IngestionError(f"Unknown method selection: {selection}") from exc


def run_pipeline(methods: list[MethodConfig], config: PipelineConfig) -> Path:
    ensure_source_document(config.pdf_path)

    document_id = config.pdf_path.stem
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_root / document_id / run_id
    if run_dir.exists():
        if not config.overwrite:
            raise IngestionError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    document_manifest = build_document_manifest(config.pdf_path)
    run_manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document": document_manifest,
        "methods": [],
    }

    for method in methods:
        result = run_method(method, config.pdf_path, run_dir)
        run_manifest["methods"].append(result)

    run_manifest["status"] = summarize_run_status(run_manifest["methods"])

    write_json(run_dir / "run_manifest.json", run_manifest)
    if run_manifest["status"] != "completed":
        raise IncompleteRunError(run_dir, run_manifest["methods"])
    return run_dir


def ensure_source_document(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise IngestionError(
            "Default PDF not found. Update pdf_extract_experiments/config.py or add the file at "
            f"{pdf_path}."
        )


def build_document_manifest(pdf_path: Path) -> dict[str, Any]:
    return {
        "path": str(pdf_path),
        "name": pdf_path.name,
        "sha256": sha256_file(pdf_path),
        "size_bytes": pdf_path.stat().st_size,
    }


def run_method(method: MethodConfig, pdf_path: Path, run_dir: Path) -> dict[str, Any]:
    method_dir = run_dir / method.method_id
    raw_dir = method_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        method_dir / "method_manifest.json",
        {
            "method": asdict(method),
            "status": "started",
            "source_pdf": str(pdf_path),
        },
    )

    preflight = preflight_check(method)
    if preflight:
        result = {
            "method_id": method.method_id,
            "status": "blocked",
            "errors": preflight,
            "raw_outputs": [],
            "normalized_output": None,
        }
        write_json(
            method_dir / "method_manifest.json", {"method": asdict(method), **result}
        )
        return result

    try:
        convert_pdf(method, pdf_path, raw_dir)
        normalized_path, record_count = normalize_outputs(method, method_dir, pdf_path)
    except IngestionError as exc:
        result = {
            "method_id": method.method_id,
            "status": "failed",
            "errors": [str(exc)],
            "raw_outputs": list_raw_outputs(method_dir),
            "normalized_output": None,
        }
        write_json(
            method_dir / "method_manifest.json", {"method": asdict(method), **result}
        )
        return result

    result = {
        "method_id": method.method_id,
        "status": "completed",
        "errors": [],
        "raw_outputs": list_raw_outputs(method_dir),
        "normalized_output": str(normalized_path.relative_to(method_dir)),
        "normalized_record_count": record_count,
    }
    if method.output_kind == "document":
        result["normalized_element_count"] = record_count
    write_json(
        method_dir / "method_manifest.json", {"method": asdict(method), **result}
    )
    return result


def preflight_check(method: MethodConfig) -> list[str]:
    if method.backend == "opendataloader":
        return preflight_opendataloader(method)
    if method.backend == "pymupdf":
        return preflight_import(
            module_name="pymupdf",
            dependency_name="pymupdf",
            install_hint="Install PyMuPDF in the active environment before running this method.",
        )
    if method.backend == "pdfplumber":
        return preflight_import(
            module_name="pdfplumber",
            dependency_name="pdfplumber",
            install_hint="Install pdfplumber in the active environment before running this method.",
        )
    if method.backend == "camelot":
        return preflight_import(
            module_name="camelot",
            dependency_name="camelot-py",
            install_hint="Install camelot-py in the active environment before running this method.",
        )
    return [f"Unsupported backend '{method.backend}'."]


def preflight_opendataloader(method: MethodConfig) -> list[str]:
    errors: list[str] = []
    if method.requires_java:
        errors.extend(check_java_runtime())

    try:
        __import__("opendataloader_pdf")
    except ModuleNotFoundError:
        errors.append(
            "Python dependency 'opendataloader-pdf' is not installed in the active environment. "
            "Install project dependencies before running ingestion."
        )

    if method.requires_hybrid_backend:
        errors.append(
            "OCR mode expects the OpenDataLoader hybrid backend to be installed and running separately; "
            "this pipeline records the method configuration but does not bootstrap that backend."
        )

    return errors


def preflight_import(
    module_name: str,
    dependency_name: str,
    install_hint: str,
) -> list[str]:
    try:
        __import__(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return [
                f"Python dependency '{dependency_name}' is not installed in the active environment. {install_hint}"
            ]
        return [
            f"Python dependency '{dependency_name}' is installed but missing required module '{exc.name}'. {install_hint}"
        ]
    return []


def check_java_runtime() -> list[str]:
    errors: list[str] = []
    configure_java_runtime()

    if shutil.which("java") is None:
        errors.append("Java 11+ is required but 'java' is not available on PATH.")
        return errors

    try:
        completed = subprocess.run(
            ["java", "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return errors

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        if details:
            errors.append(f"Java runtime check failed: {details}")
    return errors


def convert_pdf(method: MethodConfig, pdf_path: Path, raw_dir: Path) -> None:
    if method.backend == "opendataloader":
        convert_with_opendataloader(method, pdf_path, raw_dir)
        return
    if method.backend == "pymupdf":
        convert_with_pymupdf(pdf_path, raw_dir)
        return
    if method.backend == "pdfplumber":
        convert_with_pdfplumber(pdf_path, raw_dir)
        return
    if method.backend == "camelot":
        convert_with_camelot(method, pdf_path, raw_dir)
        return
    raise IngestionError(f"Unsupported backend '{method.backend}'.")


def normalize_outputs(
    method: MethodConfig, method_dir: Path, pdf_path: Path
) -> tuple[Path, int]:
    if method.backend == "opendataloader":
        return normalize_opendataloader_outputs(method_dir, method.method_id, pdf_path)
    if method.backend == "pymupdf":
        return normalize_pymupdf_outputs(method_dir, method.method_id, pdf_path)
    if method.backend == "pdfplumber":
        return normalize_pdfplumber_outputs(method_dir, method.method_id, pdf_path)
    if method.backend == "camelot":
        return normalize_camelot_outputs(method_dir, method.method_id, pdf_path)
    raise IngestionError(f"Unsupported backend '{method.backend}'.")


def normalize_opendataloader_outputs(
    method_dir: Path, method_id: str, pdf_path: Path
) -> tuple[Path, int]:
    raw_dir = method_dir / "raw"
    json_files = sorted(raw_dir.rglob("*.json"))
    if not json_files:
        raise IngestionError(f"No JSON output produced for method '{method_id}'.")

    records: list[dict[str, Any]] = []
    for json_file in json_files:
        payload = json.loads(json_file.read_text())
        for index, element in enumerate(iter_elements(payload)):
            records.append(
                {
                    "document_id": pdf_path.stem,
                    "method_id": method_id,
                    "source_file": str(json_file.relative_to(method_dir)),
                    "element_index": index,
                    "element_type": element.get("type"),
                    "page_number": element.get(
                        "page number", element.get("page_number")
                    ),
                    "content": pick_content(element),
                    "bounding_box": element.get(
                        "bounding box", element.get("bounding_box")
                    ),
                    "metadata": {
                        key: value
                        for key, value in element.items()
                        if key
                        not in {
                            "type",
                            "page number",
                            "page_number",
                            "content",
                            "text",
                            "description",
                            "bounding box",
                            "bounding_box",
                        }
                    },
                }
            )

    normalized_path = method_dir / "normalized.json"
    write_json(normalized_path, {"records": records})
    return normalized_path, len(records)


def normalize_pymupdf_outputs(
    method_dir: Path, method_id: str, pdf_path: Path
) -> tuple[Path, int]:
    payload, json_path = read_single_json_payload(method_dir / "raw", method_id)
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise IngestionError(
            f"Unexpected PyMuPDF payload shape for method '{method_id}'."
        )

    records: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        page_height = page.get("height")
        words = page.get("words")
        if not isinstance(words, list):
            continue
        for index, word in enumerate(words):
            if not isinstance(word, dict):
                continue
            records.append(
                {
                    "document_id": pdf_path.stem,
                    "method_id": method_id,
                    "record_kind": "document",
                    "source_file": str(json_path.relative_to(method_dir)),
                    "element_index": index,
                    "element_type": "word",
                    "page_number": page_number,
                    "content": word.get("text"),
                    "bounding_box": normalize_pymupdf_bbox(word, page_height),
                    "metadata": {
                        "coordinate_space": "top-left",
                        "page_width": page.get("width"),
                        "page_height": page_height,
                        "original_bbox": [
                            word.get("x0"),
                            word.get("y0"),
                            word.get("x1"),
                            word.get("y1"),
                        ],
                        "block_no": word.get("block_no"),
                        "line_no": word.get("line_no"),
                        "word_no": word.get("word_no"),
                    },
                }
            )

    normalized_path = method_dir / "normalized.json"
    write_json(normalized_path, {"records": records})
    return normalized_path, len(records)


def normalize_pdfplumber_outputs(
    method_dir: Path, method_id: str, pdf_path: Path
) -> tuple[Path, int]:
    payload, json_path = read_single_json_payload(method_dir / "raw", method_id)
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise IngestionError(
            f"Unexpected pdfplumber payload shape for method '{method_id}'."
        )

    records: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        words = page.get("words")
        if not isinstance(words, list):
            continue
        for index, word in enumerate(words):
            if not isinstance(word, dict):
                continue
            records.append(
                {
                    "document_id": pdf_path.stem,
                    "method_id": method_id,
                    "record_kind": "document",
                    "source_file": str(json_path.relative_to(method_dir)),
                    "element_index": index,
                    "element_type": "word",
                    "page_number": page_number,
                    "content": word.get("text"),
                    "bounding_box": [
                        word.get("x0"),
                        word.get("top"),
                        word.get("x1"),
                        word.get("bottom"),
                    ],
                    "metadata": {
                        "coordinate_space": "top-left",
                        "page_width": page.get("width"),
                        "page_height": page.get("height"),
                        "original_bbox": [
                            word.get("x0"),
                            word.get("y0"),
                            word.get("x1"),
                            word.get("y1"),
                        ],
                        "fontname": word.get("fontname"),
                        "size": word.get("size"),
                        "upright": word.get("upright"),
                    },
                }
            )

    normalized_path = method_dir / "normalized.json"
    write_json(normalized_path, {"records": records})
    return normalized_path, len(records)


def normalize_camelot_outputs(
    method_dir: Path, method_id: str, pdf_path: Path
) -> tuple[Path, int]:
    payload, json_path = read_single_json_payload(method_dir / "raw", method_id)
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise IngestionError(
            f"Unexpected Camelot payload shape for method '{method_id}'."
        )

    records: list[dict[str, Any]] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        records.append(
            {
                "document_id": pdf_path.stem,
                "method_id": method_id,
                "record_kind": "table",
                "source_file": str(json_path.relative_to(method_dir)),
                "element_index": table.get("table_index"),
                "element_type": "table",
                "page_number": table.get("page_number"),
                "content": table.get("data"),
                "bounding_box": table.get("bbox"),
                "metadata": {
                    "flavor": payload.get("flavor"),
                    "shape": table.get("shape"),
                    "order": table.get("order"),
                    "parsing_report": table.get("parsing_report"),
                },
            }
        )

    normalized_path = method_dir / "normalized.json"
    write_json(normalized_path, {"records": records})
    return normalized_path, len(records)


def read_single_json_payload(
    raw_dir: Path, method_id: str
) -> tuple[dict[str, Any], Path]:
    json_files = sorted(raw_dir.rglob("*.json"))
    if not json_files:
        raise IngestionError(f"No JSON output produced for method '{method_id}'.")
    json_path = json_files[0]
    payload = json.loads(json_path.read_text())
    if not isinstance(payload, dict):
        raise IngestionError(
            f"Unexpected JSON output produced for method '{method_id}'."
        )
    return payload, json_path


def list_raw_outputs(method_dir: Path) -> list[str]:
    raw_dir = method_dir / "raw"
    return sorted(
        str(path.relative_to(method_dir))
        for path in raw_dir.rglob("*")
        if path.is_file()
    )


def normalize_pymupdf_bbox(word: dict[str, Any], page_height: Any) -> list[Any] | None:
    try:
        x0 = word["x0"]
        y0 = word["y0"]
        x1 = word["x1"]
        y1 = word["y1"]
    except KeyError:
        return None
    if page_height is None:
        return [x0, y0, x1, y1]
    return [x0, page_height - y1, x1, page_height - y0]


def normalize_table_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def convert_with_opendataloader(
    method: MethodConfig, pdf_path: Path, raw_dir: Path
) -> None:
    import opendataloader_pdf

    configure_java_runtime()

    options = {
        "input_path": [str(pdf_path)],
        "output_dir": str(raw_dir),
        **method.convert_options,
    }
    try:
        opendataloader_pdf.convert(**options)
    except Exception as exc:  # pragma: no cover - depends on external runtime
        raise IngestionError(
            f"Extraction failed for method '{method.method_id}': {exc}"
        ) from exc


def convert_with_pymupdf(pdf_path: Path, raw_dir: Path) -> None:
    import pymupdf

    pages: list[dict[str, Any]] = []
    markdown_parts: list[str] = []
    try:
        with pymupdf.open(str(pdf_path)) as document:
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text(sort=True)
                words: list[dict[str, Any]] = []
                for word in page.get_text("words", sort=True):
                    x0, y0, x1, y1, text, block_no, line_no, word_no = word
                    words.append(
                        {
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                            "text": text,
                            "block_no": block_no,
                            "line_no": line_no,
                            "word_no": word_no,
                        }
                    )
                pages.append(
                    {
                        "page_number": page_number,
                        "width": page.rect.width,
                        "height": page.rect.height,
                        "text": page_text,
                        "words": words,
                    }
                )
                markdown_parts.append(f"## Page {page_number}\n\n{page_text.strip()}")
    except Exception as exc:  # pragma: no cover - depends on external runtime
        raise IngestionError(f"PyMuPDF extraction failed: {exc}") from exc

    write_json(
        raw_dir / f"{pdf_path.stem}.json",
        {"backend": "pymupdf", "pages": pages},
    )
    (raw_dir / f"{pdf_path.stem}.md").write_text(
        "\n\n".join(markdown_parts).strip() + "\n"
    )


def convert_with_pdfplumber(pdf_path: Path, raw_dir: Path) -> None:
    import pdfplumber

    pages: list[dict[str, Any]] = []
    markdown_parts: list[str] = []
    try:
        with pdfplumber.open(str(pdf_path)) as document:
            for page in document.pages:
                words = page.extract_words(
                    use_text_flow=True,
                    return_chars=False,
                    extra_attrs=["fontname", "size"],
                )
                page_text = page.extract_text(layout=True) or ""
                pages.append(
                    {
                        "page_number": page.page_number,
                        "width": page.width,
                        "height": page.height,
                        "text": page_text,
                        "words": words,
                        "image_count": len(page.images),
                        "rect_count": len(page.rects),
                        "curve_count": len(page.curves),
                    }
                )
                markdown_parts.append(
                    f"## Page {page.page_number}\n\n{page_text.strip()}"
                )
    except Exception as exc:  # pragma: no cover - depends on external runtime
        raise IngestionError(f"pdfplumber extraction failed: {exc}") from exc

    write_json(
        raw_dir / f"{pdf_path.stem}.json",
        {"backend": "pdfplumber", "pages": pages},
    )
    (raw_dir / f"{pdf_path.stem}.md").write_text(
        "\n\n".join(markdown_parts).strip() + "\n"
    )


def convert_with_camelot(method: MethodConfig, pdf_path: Path, raw_dir: Path) -> None:
    import camelot

    try:
        tables = camelot.read_pdf(str(pdf_path), **method.convert_options)
    except Exception as exc:  # pragma: no cover - depends on external runtime
        raise IngestionError(f"Camelot extraction failed: {exc}") from exc

    table_payloads: list[dict[str, Any]] = []
    for table_index, table in enumerate(tables):
        bbox = getattr(table, "bbox", None)
        if bbox is None:
            bbox = getattr(table, "_bbox", None)
        parsing_report = dict(getattr(table, "parsing_report", {}))
        table_payloads.append(
            {
                "table_index": table_index,
                "page_number": parsing_report.get("page"),
                "order": parsing_report.get("order"),
                "bbox": list(bbox) if bbox is not None else None,
                "shape": [int(table.df.shape[0]), int(table.df.shape[1])],
                "data": [
                    [normalize_table_cell(cell) for cell in row]
                    for row in table.df.values.tolist()
                ],
                "parsing_report": parsing_report,
            }
        )

    write_json(
        raw_dir / f"{pdf_path.stem}.json",
        {
            "backend": "camelot",
            "flavor": method.convert_options.get("flavor", "lattice"),
            "tables": table_payloads,
        },
    )


def iter_elements(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [element for element in payload if isinstance(element, dict)]
    if isinstance(payload, dict):
        for key in ("elements", "content", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [element for element in value if isinstance(element, dict)]
        return [payload]
    return []


def pick_content(element: dict[str, Any]) -> Any:
    for key in ("content", "text", "description"):
        if key in element:
            return element[key]
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def summarize_run_status(results: list[dict[str, Any]]) -> str:
    if any(result.get("status") == "blocked" for result in results):
        return "blocked"
    if any(result.get("status") != "completed" for result in results):
        return "failed"
    return "completed"


def configure_java_runtime() -> None:
    if shutil.which("java") is not None:
        return

    candidate_home = homebrew_java_home()
    if candidate_home is None:
        return

    os.environ.setdefault("JAVA_HOME", str(candidate_home))
    java_bin = candidate_home / "bin"
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    java_bin_str = str(java_bin)
    if java_bin_str not in path_parts:
        os.environ["PATH"] = (
            os.pathsep.join([java_bin_str, *path_parts]) if path_parts else java_bin_str
        )


def homebrew_java_home() -> Path | None:
    candidates = [
        Path("/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"),
        Path("/usr/local/opt/openjdk/libexec/openjdk.jdk/Contents/Home"),
    ]
    for candidate in candidates:
        if (candidate / "bin" / "java").exists():
            return candidate
    return None

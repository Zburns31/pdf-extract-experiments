from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHUNK_CHAR_LIMIT = 1600
DEFAULT_CHUNK_OVERLAP = 200
QUESTION_PROMPT_VERSION = "v1"
EVALUATION_VERSION = "v1"


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationConfig:
    run_dir: Path
    gold_method_id: str | None = None
    question_count: int = 24
    ollama_model: str | None = None
    ollama_host: str | None = None
    reuse_questions: bool = True


def evaluate_run(config: EvaluationConfig) -> Path:
    if config.question_count < 1:
        raise EvaluationError("Question count must be at least 1.")

    run_manifest_path = config.run_dir / "run_manifest.json"
    run_manifest = load_json(run_manifest_path)
    method_results = run_manifest.get("methods")
    if not isinstance(method_results, list):
        raise EvaluationError(
            f"Run manifest is missing method results: {run_manifest_path}"
        )

    method_infos = [
        load_method_info(config.run_dir, result)
        for result in method_results
        if isinstance(result, dict)
    ]
    document_methods = [
        method_info
        for method_info in method_infos
        if method_info["status"] == "completed"
        and method_info["output_kind"] == "document"
        and method_info["normalized_path"] is not None
    ]
    if not document_methods:
        raise EvaluationError(
            "No completed document methods found in this run. Evaluation requires at least one full-document extraction."
        )

    gold_method = select_gold_method(document_methods, config.gold_method_id)
    gold_chunks = build_document_chunks(gold_method)
    if not gold_chunks:
        raise EvaluationError(
            f"Gold method '{gold_method['method_id']}' did not yield any text chunks for evaluation."
        )

    evaluation_dir = config.run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    questions_path = evaluation_dir / "questions.json"
    questions_payload = maybe_load_reusable_questions(
        questions_path, gold_method, config
    )
    if questions_payload is None:
        client = create_ollama_client(config)
        questions_payload = generate_questions(gold_chunks, gold_method, config, client)
        write_json(questions_path, questions_payload)

    questions = questions_payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise EvaluationError("Question generation produced no usable questions.")

    method_summaries: list[dict[str, Any]] = []
    for method_info in method_infos:
        summary = evaluate_method(method_info, questions, config.run_dir)
        method_summaries.append(summary)
        persist_method_evaluation(config.run_dir, method_info, summary)

    ranked_document_methods = sorted(
        [
            summary
            for summary in method_summaries
            if summary.get("benchmark_kind") == "document"
            and summary.get("status") == "completed"
        ],
        key=lambda summary: summary.get("composite_score", -1),
        reverse=True,
    )

    comparison_payload = {
        "evaluation_version": EVALUATION_VERSION,
        "benchmark_kind": "document_qa_proxy",
        "gold_method_id": gold_method["method_id"],
        "question_count": len(questions),
        "question_output": str(questions_path.relative_to(config.run_dir)),
        "method_summaries": method_summaries,
        "ranked_document_methods": [
            summary["method_id"] for summary in ranked_document_methods
        ],
    }
    comparison_path = evaluation_dir / "comparison.json"
    write_json(comparison_path, comparison_payload)
    update_run_manifest(run_manifest_path, comparison_payload, comparison_path)
    return comparison_path


def load_method_info(run_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    method_id = result.get("method_id")
    if not isinstance(method_id, str):
        raise EvaluationError(
            "Run manifest contains a method result without a method_id."
        )

    method_dir = run_dir / method_id
    method_manifest_path = method_dir / "method_manifest.json"
    method_manifest = load_json(method_manifest_path)
    method_config = method_manifest.get("method")
    if not isinstance(method_config, dict):
        raise EvaluationError(
            f"Method manifest is missing method configuration: {method_manifest_path}"
        )

    normalized_output = result.get("normalized_output")
    normalized_path = (
        method_dir / normalized_output if isinstance(normalized_output, str) else None
    )
    return {
        "method_id": method_id,
        "method_dir": method_dir,
        "method_manifest_path": method_manifest_path,
        "method_manifest": method_manifest,
        "status": result.get("status"),
        "output_kind": method_config.get("output_kind"),
        "backend": method_config.get("backend"),
        "normalized_path": normalized_path,
    }


def select_gold_method(
    document_methods: list[dict[str, Any]], requested_method_id: str | None
) -> dict[str, Any]:
    by_id = {method_info["method_id"]: method_info for method_info in document_methods}
    if requested_method_id is not None:
        try:
            return by_id[requested_method_id]
        except KeyError as exc:
            raise EvaluationError(
                f"Gold method '{requested_method_id}' was not found among completed document methods."
            ) from exc

    for preferred_method in ("text", "pymupdf_text", "pdfplumber_text", "ocr"):
        if preferred_method in by_id:
            return by_id[preferred_method]
    return sorted(document_methods, key=lambda method_info: method_info["method_id"])[0]


def build_document_chunks(method_info: dict[str, Any]) -> list[dict[str, Any]]:
    normalized_path = method_info.get("normalized_path")
    if not isinstance(normalized_path, Path) or not normalized_path.exists():
        raise EvaluationError(
            f"Normalized output not found for method '{method_info['method_id']}'."
        )

    payload = load_json(normalized_path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise EvaluationError(
            f"Normalized payload is missing records: {normalized_path}"
        )

    backend = method_info.get("backend")
    if backend == "opendataloader":
        return build_opendataloader_chunks(method_info["method_id"], records)
    if backend in {"pymupdf", "pdfplumber"}:
        return build_word_chunks(method_info["method_id"], records)
    raise EvaluationError(
        f"Document evaluation is not implemented for backend '{backend}' on method '{method_info['method_id']}'."
    )


def build_opendataloader_chunks(
    method_id: str, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    sequence = 0
    for record in records:
        sequence = flatten_opendataloader_element(record, flattened, sequence)

    pages: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for element in flattened:
        page_number = element.get("page_number")
        if isinstance(page_number, int):
            pages[page_number].append(element)

    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for page_number in sorted(pages):
        lines: list[str] = []
        for element in pages[page_number]:
            content = clean_text(element.get("content"))
            if not content:
                continue
            if element.get("element_type") == "heading":
                lines.append(f"# {content}")
            else:
                lines.append(content)
        page_text = "\n\n".join(lines).strip()
        for chunk_text in split_text(page_text):
            chunks.append(
                {
                    "chunk_id": f"{method_id}-page-{page_number}-chunk-{chunk_index}",
                    "page_numbers": [page_number],
                    "text": chunk_text,
                }
            )
            chunk_index += 1
    return chunks


def flatten_opendataloader_element(
    element: dict[str, Any],
    flattened: list[dict[str, Any]],
    sequence: int,
    page_hint: int | None = None,
) -> int:
    metadata = (
        element.get("metadata") if isinstance(element.get("metadata"), dict) else {}
    )
    page_number = coerce_int(
        element.get("page_number")
        if element.get("page_number") is not None
        else metadata.get("page number", metadata.get("page_number", page_hint))
    )
    content = clean_text(element.get("content"))
    if content:
        flattened.append(
            {
                "page_number": page_number,
                "element_type": element.get("element_type") or element.get("type"),
                "content": content,
                "sequence": sequence,
            }
        )
        sequence += 1

    kids = metadata.get("kids")
    if isinstance(kids, list):
        for kid in kids:
            if isinstance(kid, dict):
                kid_element = {
                    "content": kid.get("content")
                    or kid.get("text")
                    or kid.get("description"),
                    "element_type": kid.get("type"),
                    "page_number": kid.get(
                        "page number", kid.get("page_number", page_number)
                    ),
                    "metadata": {"kids": kid.get("kids", [])},
                }
                sequence = flatten_opendataloader_element(
                    kid_element, flattened, sequence, page_number
                )
    return sequence


def build_word_chunks(
    method_id: str, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    page_lines: dict[int, dict[tuple[int, int], list[tuple[int, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        page_number = coerce_int(record.get("page_number"))
        if page_number is None:
            continue
        metadata = (
            record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        )
        block_no = coerce_int(metadata.get("block_no")) or 0
        line_no = coerce_int(metadata.get("line_no")) or 0
        word_no = coerce_int(metadata.get("word_no"))
        if word_no is None:
            word_no = coerce_int(record.get("element_index")) or 0
        content = clean_text(record.get("content"))
        if not content:
            continue
        page_lines[page_number][(block_no, line_no)].append((word_no, content))

    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for page_number in sorted(page_lines):
        lines: list[str] = []
        for block_line in sorted(page_lines[page_number]):
            words = [word for _, word in sorted(page_lines[page_number][block_line])]
            lines.append(" ".join(words))
        page_text = "\n".join(lines).strip()
        for chunk_text in split_text(page_text):
            chunks.append(
                {
                    "chunk_id": f"{method_id}-page-{page_number}-chunk-{chunk_index}",
                    "page_numbers": [page_number],
                    "text": chunk_text,
                }
            )
            chunk_index += 1
    return chunks


def maybe_load_reusable_questions(
    questions_path: Path, gold_method: dict[str, Any], config: EvaluationConfig
) -> dict[str, Any] | None:
    if not config.reuse_questions or not questions_path.exists():
        return None
    payload = load_json(questions_path)
    if payload.get("gold_method_id") != gold_method["method_id"]:
        return None
    if payload.get("question_prompt_version") != QUESTION_PROMPT_VERSION:
        return None
    questions = payload.get("questions")
    if not isinstance(questions, list) or len(questions) != config.question_count:
        return None
    return payload


def generate_questions(
    gold_chunks: list[dict[str, Any]],
    gold_method: dict[str, Any],
    config: EvaluationConfig,
    client: Any,
) -> dict[str, Any]:
    sampled_chunks = evenly_sample_chunks(gold_chunks, config.question_count)
    questions: list[dict[str, Any]] = []
    for question_index, chunk in enumerate(sampled_chunks):
        generated = generate_question_for_chunk(
            client, config.ollama_model, chunk, question_index
        )
        questions.append(generated)
    return {
        "evaluation_version": EVALUATION_VERSION,
        "question_prompt_version": QUESTION_PROMPT_VERSION,
        "gold_method_id": gold_method["method_id"],
        "question_count": len(questions),
        "questions": questions,
    }


def evenly_sample_chunks(
    chunks: list[dict[str, Any]], question_count: int
) -> list[dict[str, Any]]:
    if question_count >= len(chunks):
        return chunks[:question_count]
    if question_count == 1:
        return [chunks[len(chunks) // 2]]
    sampled: list[dict[str, Any]] = []
    max_index = len(chunks) - 1
    for position in range(question_count):
        index = round(position * max_index / (question_count - 1))
        sampled.append(chunks[index])
    return sampled


def generate_question_for_chunk(
    client: Any, model_name: str | None, chunk: dict[str, Any], question_index: int
) -> dict[str, Any]:
    if not isinstance(model_name, str) or not model_name.strip():
        raise EvaluationError(
            "An Ollama model name is required for question generation. Pass --ollama-model or set PDF_EXTRACT_EVAL_MODEL."
        )

    searchable_chunk = normalize_search_text(chunk["text"])
    page_numbers = chunk.get("page_numbers") or []
    base_prompt = (
        "You are creating a benchmark question for extracted PDF text.\n"
        "Return exactly one JSON object with these keys: question, answer, answer_type, evidence_quote, section_label.\n"
        "Rules:\n"
        "- The question must be answerable directly from the provided text chunk.\n"
        "- The answer must be short, specific, and searchable in the chunk.\n"
        "- evidence_quote must be an exact contiguous quote copied from the chunk.\n"
        "- The answer should either exactly appear in the chunk or be a short substring of evidence_quote.\n"
        "- answer_type must be one of: factoid, numeric, date, list, short_explanation.\n"
        "- section_label should be a short label for this chunk, not a sentence.\n"
        "- Prefer questions whose answers are distinctive, not generic.\n"
        f"- This chunk comes from page(s): {page_numbers}.\n"
        "Chunk text:\n"
        f"{chunk['text']}"
    )

    payload: dict[str, Any] | None = None
    last_error = ""
    for attempt in range(3):
        prompt = base_prompt
        if attempt > 0:
            prompt = (
                f"{base_prompt}\n\n"
                f"Previous attempt failed validation: {last_error}.\n"
                "Return a different JSON object that satisfies every rule."
            )
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        content = response["message"]["content"]
        payload = parse_llm_json(content)
        answer = clean_text(payload.get("answer"))
        evidence_quote = clean_text(payload.get("evidence_quote"))
        question = clean_text(payload.get("question"))
        section_label = clean_text(payload.get("section_label"))
        answer_type = clean_text(payload.get("answer_type"))
        if (
            not question
            or not answer
            or not evidence_quote
            or not section_label
            or not answer_type
        ):
            last_error = "question generation returned incomplete data"
            continue
        if normalize_search_text(evidence_quote) not in searchable_chunk:
            last_error = "evidence_quote was not present in the chunk"
            continue
        if normalize_search_text(answer) not in searchable_chunk:
            payload["answer"] = evidence_quote
        break

    if payload is None:
        raise EvaluationError(
            f"Question generation returned no data for chunk {chunk['chunk_id']}."
        )

    answer = clean_text(payload.get("answer"))
    evidence_quote = clean_text(payload.get("evidence_quote"))
    question = clean_text(payload.get("question"))
    section_label = clean_text(payload.get("section_label"))
    answer_type = clean_text(payload.get("answer_type"))
    if (
        not question
        or not answer
        or not evidence_quote
        or not section_label
        or not answer_type
    ):
        raise EvaluationError(
            f"Question generation returned incomplete data for chunk {chunk['chunk_id']}."
        )
    if normalize_search_text(evidence_quote) not in searchable_chunk:
        raise EvaluationError(
            f"Generated evidence quote was not present in gold chunk {chunk['chunk_id']}."
        )
    return {
        "question_id": f"q-{question_index:03d}",
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "evidence_quote": evidence_quote,
        "section_label": section_label,
        "source_chunk_id": chunk["chunk_id"],
        "source_page_numbers": page_numbers,
    }


def evaluate_method(
    method_info: dict[str, Any], questions: list[dict[str, Any]], run_dir: Path
) -> dict[str, Any]:
    method_id = method_info["method_id"]
    status = method_info.get("status")
    if status != "completed":
        return {
            "method_id": method_id,
            "benchmark_kind": "document",
            "status": "skipped",
            "reason": f"method_status_{status}",
        }

    if method_info.get("output_kind") != "document":
        return {
            "method_id": method_id,
            "benchmark_kind": "table",
            "status": "skipped",
            "reason": "table_methods_not_in_document_benchmark",
        }

    chunks = build_document_chunks(method_info)
    normalized_chunks = [normalize_text(chunk["text"]) for chunk in chunks]
    judgments: list[dict[str, Any]] = []
    answered_pages: set[int] = set()
    source_pages: set[int] = set()
    answer_hits = 0
    evidence_hits = 0
    retrieval_total = 0.0

    for question in questions:
        source_page_numbers = question.get("source_page_numbers") or []
        for page_number in source_page_numbers:
            if isinstance(page_number, int):
                source_pages.add(page_number)

        question_text = question["question"]
        answer_norm = normalize_search_text(question["answer"])
        evidence_norm = normalize_search_text(question["evidence_quote"])
        best_overlap = 0.0
        best_chunk_id: str | None = None
        answer_present = False
        evidence_present = False
        matching_pages: set[int] = set()

        for chunk, normalized_chunk in zip(chunks, normalized_chunks, strict=False):
            overlap = token_overlap_score(question_text, chunk["text"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk_id = chunk["chunk_id"]
            if (
                answer_norm
                and normalize_search_text(chunk["text"]).find(answer_norm) != -1
            ):
                answer_present = True
                for page_number in chunk.get("page_numbers", []):
                    if isinstance(page_number, int):
                        matching_pages.add(page_number)
            if (
                evidence_norm
                and normalize_search_text(chunk["text"]).find(evidence_norm) != -1
            ):
                evidence_present = True
                for page_number in chunk.get("page_numbers", []):
                    if isinstance(page_number, int):
                        matching_pages.add(page_number)

        if answer_present:
            answer_hits += 1
        if evidence_present:
            evidence_hits += 1
        if answer_present and source_page_numbers:
            answered_pages.update(
                page_number
                for page_number in source_page_numbers
                if isinstance(page_number, int)
            )
        retrieval_total += best_overlap
        judgments.append(
            {
                "question_id": question["question_id"],
                "question": question_text,
                "answer": question["answer"],
                "source_page_numbers": source_page_numbers,
                "best_chunk_id": best_chunk_id,
                "best_retrieval_overlap": round(best_overlap, 4),
                "answer_present": answer_present,
                "evidence_present": evidence_present,
                "matching_pages": sorted(matching_pages),
            }
        )

    question_count = len(questions)
    answer_recall = answer_hits / question_count
    evidence_recall = evidence_hits / question_count
    coverage = (
        (len(answered_pages) / len(source_pages)) if source_pages else answer_recall
    )
    retrieval_score = retrieval_total / question_count
    composite_score = (
        0.70 * answer_recall
        + 0.15 * evidence_recall
        + 0.10 * coverage
        + 0.05 * retrieval_score
    )
    return {
        "method_id": method_id,
        "benchmark_kind": "document",
        "status": "completed",
        "question_count": question_count,
        "chunk_count": len(chunks),
        "answer_recall": round(answer_recall, 4),
        "evidence_recall": round(evidence_recall, 4),
        "section_coverage": round(coverage, 4),
        "retrieval_score": round(retrieval_score, 4),
        "composite_score": round(composite_score * 100, 2),
        "judgments": judgments,
    }


def persist_method_evaluation(
    run_dir: Path, method_info: dict[str, Any], summary: dict[str, Any]
) -> None:
    method_dir = method_info["method_dir"]
    evaluation_dir = method_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    judgments = summary.pop("judgments", [])
    summary_path = evaluation_dir / "summary.json"
    judgments_path = evaluation_dir / "judgments.json"
    write_json(summary_path, summary)
    write_json(
        judgments_path,
        {
            "method_id": summary["method_id"],
            "benchmark_kind": summary.get("benchmark_kind"),
            "judgments": judgments,
        },
    )

    method_manifest = method_info["method_manifest"]
    method_manifest["evaluation"] = {
        "status": summary.get("status"),
        "benchmark_kind": summary.get("benchmark_kind"),
        "summary_output": str(summary_path.relative_to(method_dir)),
        "judgments_output": str(judgments_path.relative_to(method_dir)),
        "composite_score": summary.get("composite_score"),
    }
    write_json(method_info["method_manifest_path"], method_manifest)


def update_run_manifest(
    run_manifest_path: Path, comparison_payload: dict[str, Any], comparison_path: Path
) -> None:
    run_manifest = load_json(run_manifest_path)
    run_manifest["evaluation"] = {
        "status": "completed",
        "comparison_output": str(comparison_path.relative_to(run_manifest_path.parent)),
        "gold_method_id": comparison_payload["gold_method_id"],
        "question_output": comparison_payload["question_output"],
        "question_count": comparison_payload["question_count"],
        "ranked_document_methods": comparison_payload["ranked_document_methods"],
    }
    write_json(run_manifest_path, run_manifest)


def create_ollama_client(config: EvaluationConfig) -> Any:
    try:
        import ollama
    except ModuleNotFoundError as exc:
        raise EvaluationError(
            "The 'ollama' dependency is not installed in the active environment."
        ) from exc

    kwargs: dict[str, Any] = {}
    if config.ollama_host:
        kwargs["host"] = config.ollama_host
    return ollama.Client(**kwargs)


def parse_llm_json(content: str) -> dict[str, Any]:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z0-9_-]*", "", candidate).strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise EvaluationError(
                "LLM response did not contain a parseable JSON object."
            )
        payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise EvaluationError("LLM response JSON must be an object.")
    return payload


def split_text(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= CHUNK_CHAR_LIMIT:
        return [cleaned]

    paragraphs = [
        paragraph.strip() for paragraph in cleaned.split("\n\n") if paragraph.strip()
    ]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= CHUNK_CHAR_LIMIT:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph
        while len(current) > CHUNK_CHAR_LIMIT:
            window = current[:CHUNK_CHAR_LIMIT]
            split_at = window.rfind(" ")
            if split_at < CHUNK_CHAR_LIMIT // 2:
                split_at = CHUNK_CHAR_LIMIT
            chunks.append(current[:split_at].strip())
            start = max(split_at - DEFAULT_CHUNK_OVERLAP, 0)
            current = current[start:].strip()
    if current:
        chunks.append(current)
    return chunks


def token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / len(left_tokens)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9%$.,:/ -]", "", text)
    return text.strip()


def normalize_search_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9%$]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise EvaluationError(f"Expected JSON object at {path}.")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

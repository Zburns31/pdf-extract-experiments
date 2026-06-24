# pdf-extract-experiments

Repository testing various PDF data extraction methods.

## Ingestion pipeline

The first implementation runs against a repository-defined default PDF:

- source PDF: `data/GOOG-10-K-2025.pdf`
- default output root: `artifacts/`
- supported methods: `text`, `ocr`, `pymupdf_text`, `pdfplumber_text`, `camelot_table`

The CLI does not accept a PDF path yet. Change the default document in `src/pdf_extract_experiments/config.py` if you want to point the pipeline at a different file.

## Usage

Install the project into the local virtual environment from the repository root:

```bash
uv sync
```

Then run the CLI in either of these ways.

Without activating the virtual environment:

```bash
uv run pdf-extract-experiments --list-methods
```

With the virtual environment activated:

```bash
source .venv/bin/activate
pdf-extract-experiments --list-methods
```

List available methods:

```bash
pdf-extract-experiments --list-methods
```

Run the local text-based extraction path:

```bash
pdf-extract-experiments --method text
```

Run the PyMuPDF whole-document comparison path:

```bash
pdf-extract-experiments --method pymupdf_text
```

Run the pdfplumber whole-document comparison path:

```bash
pdf-extract-experiments --method pdfplumber_text
```

Run the Camelot table-only comparison path:

```bash
pdf-extract-experiments --method camelot_table
```

Run every configured method against the default PDF:

```bash
pdf-extract-experiments --method all
```

Write artifacts to a custom location:

```bash
pdf-extract-experiments --method text --output-dir ./tmp-artifacts
```

Evaluate an existing run with a QA-proxy benchmark:

```bash
pdf-extract-experiments \
	--evaluate-run artifacts/GOOG-10-K-2025/<run-id> \
	--gold-method text \
	--question-count 24 \
	--ollama-model llama3.1:8b
```

Run extraction and evaluation in one command:

```bash
pdf-extract-experiments --method all --evaluate --gold-method text --ollama-model llama3.1:8b
```

## Artifacts

Each run writes output under `artifacts/<document-stem>/<run-id>/`.

- `run_manifest.json`: document metadata and per-method status
- `<method>/raw/`: raw extractor output files
- `<method>/normalized.json`: normalized records for later evaluation
- `<method>/method_manifest.json`: method-specific status and output inventory
- `evaluation/questions.json`: generated benchmark questions for the run
- `evaluation/comparison.json`: run-level ranking and aggregate metrics
- `<method>/evaluation/summary.json`: method-level aggregate evaluation metrics
- `<method>/evaluation/judgments.json`: per-question answer/evidence presence judgments

Whole-document methods normalize word- or element-level records with page numbers and bounding boxes where available.
The `camelot_table` method normalizes one record per detected table and stores the extracted grid plus Camelot parsing metrics in record metadata.

## Runtime prerequisites

The pipeline will block cleanly and record why if the runtime is not ready.

- Java 11+ must be installed and available on `PATH`
- project dependencies must be installed in the active Python environment
- the `ocr` method also expects the OpenDataLoader hybrid backend to be installed and running separately
- `pymupdf_text` uses PyMuPDF; note that PyMuPDF is distributed under AGPL/commercial licensing terms
- `pdfplumber_text` works best on machine-generated PDFs rather than scanned documents
- `camelot_table` is intended for text-based PDFs and will not extract tables reliably from scanned pages

## Comparison notes

- `text` remains the existing OpenDataLoader baseline for local full-document extraction.
- `pymupdf_text` and `pdfplumber_text` are the two whole-document comparison methods for evaluating text coverage, reading order, and bounding-box quality.
- `camelot_table` is a table-only method intended to compare table extraction quality separately from the whole-document text methods.
- Tabula integration is intentionally deferred for now so the first table-only comparison path can settle around Camelot's richer parsing metrics.

## Evaluation notes

- The first evaluation implementation is a document QA proxy benchmark, not a full answerer-and-judge loop yet.
- Questions are generated from a chosen completed document method using Ollama, with one question sampled from each gold chunk across the document.
- Scoring currently measures whether each other document method preserves the gold answer and evidence quote in its extracted text, plus a lightweight retrieval overlap score.
- Table-only methods are written to evaluation manifests but skipped from the main document benchmark.
- Re-running evaluation reuses `evaluation/questions.json` by default so scores remain comparable across methods and repeated runs.

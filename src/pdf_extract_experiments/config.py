from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
DEFAULT_PDF_PATH = REPO_ROOT / "data" / "GOOG-10-K-2025.pdf"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts"
DEFAULT_EVAL_QUESTION_COUNT = 24
DEFAULT_EVAL_MODEL = os.environ.get("PDF_EXTRACT_EVAL_MODEL")
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST")

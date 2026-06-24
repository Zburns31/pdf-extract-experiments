from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    DEFAULT_EVAL_MODEL,
    DEFAULT_EVAL_QUESTION_COUNT,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PDF_PATH,
)
from .evaluation import EvaluationConfig, EvaluationError, evaluate_run
from .pipeline import (
    IncompleteRunError,
    IngestionError,
    list_methods,
    PipelineConfig,
    resolve_methods,
    run_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-extract-experiments",
        description="Run the default PDF ingestion pipeline against the repository's configured source document.",
    )
    parser.add_argument(
        "--method",
        choices=["all", *[method.method_id for method in list_methods()]],
        default="text",
        help="Extraction method to run. Defaults to the local text-based method.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where ingestion artifacts will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the generated run directory if it already exists.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print supported extraction methods and exit.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run QA-proxy evaluation after extraction completes.",
    )
    parser.add_argument(
        "--evaluate-run",
        type=Path,
        help="Evaluate an existing run directory instead of running extraction.",
    )
    parser.add_argument(
        "--gold-method",
        choices=[method.method_id for method in list_methods()],
        help="Completed document method to use as the gold source for question generation.",
    )
    parser.add_argument(
        "--question-count",
        type=int,
        default=DEFAULT_EVAL_QUESTION_COUNT,
        help="Number of benchmark questions to generate for evaluation.",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_EVAL_MODEL,
        help="Ollama model name to use for question generation.",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Optional Ollama host URL.",
    )
    parser.add_argument(
        "--no-reuse-questions",
        action="store_true",
        help="Regenerate benchmark questions even if evaluation/questions.json already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_methods:
        for method in list_methods():
            print(f"{method.method_id}: {method.description}")
        print(f"default-pdf: {DEFAULT_PDF_PATH}")
        return 0

    evaluation_config = EvaluationConfig(
        run_dir=args.evaluate_run if args.evaluate_run is not None else Path(),
        gold_method_id=args.gold_method,
        question_count=args.question_count,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        reuse_questions=not args.no_reuse_questions,
    )

    if args.evaluate_run is not None:
        try:
            comparison_path = evaluate_run(evaluation_config)
        except EvaluationError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Evaluation artifacts written to {comparison_path}")
        return 0

    config = PipelineConfig(output_root=args.output_dir, overwrite=args.overwrite)

    try:
        run_dir = run_pipeline(resolve_methods(args.method), config)
    except IncompleteRunError as exc:
        print(str(exc), file=sys.stderr)
        for result in exc.results:
            print(f"- {result['method_id']}: {result['status']}", file=sys.stderr)
            for error in result.get("errors", []):
                print(f"  {error}", file=sys.stderr)
        return 1
    except IngestionError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.evaluate:
        try:
            comparison_path = evaluate_run(
                EvaluationConfig(
                    run_dir=run_dir,
                    gold_method_id=args.gold_method,
                    question_count=args.question_count,
                    ollama_model=args.ollama_model,
                    ollama_host=args.ollama_host,
                    reuse_questions=not args.no_reuse_questions,
                )
            )
        except EvaluationError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Evaluation artifacts written to {comparison_path}")
        return 0

    print(f"Ingestion artifacts written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

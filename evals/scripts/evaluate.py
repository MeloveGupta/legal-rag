"""
Evaluation script for the Legal RAG system.

Metrics:
  - Answer Rate:          % of Category A questions answered
  - Refusal Precision:    % of Category B questions correctly refused
  - Citation Rate:        % of answered questions that include inline citations
                          Replaces the LLM-as-judge faithfulness metric.
                          Deterministic, zero tokens, never fails to parse.
                          An answer with citations [1][2] is grounded in
                          retrieved chunks. An answer without is suspect.

Exit codes:
  0 = all thresholds passed
  1 = one or more thresholds failed (CI build will fail)
"""

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.legal_rag.generation.generator import RAGResponse, answer_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# thresholds

ANSWER_RATE_THRESHOLD       = 0.85   # 85% of Category A must be answered
REFUSAL_PRECISION_THRESHOLD = 0.90   # 90% of Category B must be refused
CITATION_RATE_THRESHOLD     = 0.80   # 80% of answered questions must have citations

# paths

GOLDEN_DATASET_PATH = PROJECT_ROOT / "evals" / "golden_dataset" / "qa_pairs.json"
REPORTS_DIR         = PROJECT_ROOT / "evals" / "reports"
CHECKPOINT_PATH     = REPORTS_DIR / "checkpoint.json"


# data structures

@dataclass
class QuestionResult:
    question_id:    str
    category:       str
    question:       str
    should_answer:  bool
    was_answered:   bool
    has_citations:  Optional[bool]   # None for Category B
    answer_preview: str
    sources_used:   int
    passed:         bool
    failure_reason: Optional[str] = None


@dataclass
class EvalReport:
    timestamp:         str
    total_questions:   int
    category_a_count:  int
    category_b_count:  int
    answer_rate:       float
    refusal_precision: float
    citation_rate:     float
    results:           List[QuestionResult]
    passed:            bool
    failure_reasons:   List[str] = field(default_factory=list)


# citation checker

def _has_citations(answer: str) -> bool:
    """
    Check whether the answer contains at least one inline citation marker.

    Citation markers look like: [1], [2], [1][3], etc.
    An answer without any citation marker is either a refusal or is making
    claims without grounding them in retrieved chunks — both are problematic.

    This is deterministic, costs zero tokens, and never fails to parse.
    It replaced the LLM-as-judge faithfulness scorer which was using a
    separate 100k TPD budget and producing inconsistent results.
    """
    return bool(re.search(r'\[\d+\]', answer))


# checkpoint helpers

def _load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        logger.info(
            f"Checkpoint found: {len(data)} questions already evaluated. "
            f"Resuming from where we left off."
        )
        return data
    return {}


def _save_checkpoint(results: List[QuestionResult]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        r.question_id: {
            "question_id":   r.question_id,
            "category":      r.category,
            "question":      r.question,
            "should_answer": r.should_answer,
            "was_answered":  r.was_answered,
            "has_citations": r.has_citations,
            "answer_preview": r.answer_preview,
            "sources_used":  r.sources_used,
            "passed":        r.passed,
            "failure_reason": r.failure_reason,
        }
        for r in results
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


def _result_from_checkpoint(data: dict) -> QuestionResult:
    return QuestionResult(**data)


# retry wrapper

def _answer_with_retry(question: str, max_retries: int = 3) -> Optional[RAGResponse]:
    """
    Call answer_query with exponential backoff on transient errors.
    Anthropic has no daily token limits so retries are safe.
    """
    for attempt in range(max_retries):
        try:
            return answer_query(question)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
            is_overload = "529" in error_str or "overloaded" in error_str.lower()

            if (is_rate_limit or is_overload) and attempt < max_retries - 1:
                wait = 15 * (2 ** attempt)   # 15s → 30s → 60s
                logger.warning(
                    f"Transient error on attempt {attempt+1}/{max_retries}: {error_str[:80]}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"Failed after {attempt+1} attempt(s): {error_str[:150]}")
                return None
    return None


# evaluation runner

def run_evaluation() -> EvalReport:
    if not GOLDEN_DATASET_PATH.exists():
        logger.error(f"Golden dataset not found: {GOLDEN_DATASET_PATH}")
        sys.exit(1)

    with open(GOLDEN_DATASET_PATH) as f:
        dataset = json.load(f)

    qa_pairs = dataset["qa_pairs"]
    cat_a = [q for q in qa_pairs if q["category"] == "A"]
    cat_b = [q for q in qa_pairs if q["category"] == "B"]

    logger.info(f"Dataset    : {len(qa_pairs)} questions "
                f"({len(cat_a)} Category A, {len(cat_b)} Category B)")
    logger.info(f"Thresholds : answer>={ANSWER_RATE_THRESHOLD:.0%} | "
                f"refusal>={REFUSAL_PRECISION_THRESHOLD:.0%} | "
                f"citation>={CITATION_RATE_THRESHOLD:.0%}")
    logger.info(f"Faithfulness: citation presence rate (deterministic, zero tokens)")

    checkpoint = _load_checkpoint()
    results: List[QuestionResult] = [
        _result_from_checkpoint(v) for v in checkpoint.values()
    ]
    evaluated_ids = {r.question_id for r in results}

    # category A
    logger.info("=" * 60)
    logger.info("EVALUATING CATEGORY A — must answer")
    logger.info("=" * 60)

    for i, qa in enumerate(cat_a):
        qid      = qa["id"]
        question = qa["question"]

        if qid in evaluated_ids:
            logger.info(f"[{i+1:02d}/{len(cat_a)}] {qid}: already in checkpoint — skipping")
            continue

        logger.info(f"[{i+1:02d}/{len(cat_a)}] {qid}: {question[:70]}...")

        response = _answer_with_retry(question)

        if response is None:
            result = QuestionResult(
                question_id=qid,
                category="A",
                question=question,
                should_answer=True,
                was_answered=False,
                has_citations=None,
                answer_preview="",
                sources_used=0,
                passed=False,
                failure_reason="All retries failed — transient error",
            )
        else:
            citations = _has_citations(response.answer) if response.answered else None

            if not response.answered:
                passed = False
                failure_reason = "System refused an answerable question"
            elif not citations:
                passed = False
                failure_reason = "Answer has no inline citations [N]"
            else:
                passed = True
                failure_reason = None

            result = QuestionResult(
                question_id=qid,
                category="A",
                question=question,
                should_answer=True,
                was_answered=response.answered,
                has_citations=citations,
                answer_preview=response.answer[:200],
                sources_used=len(response.sources),
                passed=passed,
                failure_reason=failure_reason,
            )

        status = "✓ PASS" if result.passed else "✗ FAIL"
        cite_str = str(result.has_citations) if result.has_citations is not None else "N/A"
        logger.info(
            f"  {status} | answered={result.was_answered} | citations={cite_str}"
        )

        results.append(result)
        evaluated_ids.add(qid)
        _save_checkpoint(results)

        # Anthropic has no daily limits but has per-minute rate limits.
        # 1s sleep between questions keeps us well within 60 req/min.
        time.sleep(1)

    # category B
    logger.info("=" * 60)
    logger.info("EVALUATING CATEGORY B — must refuse")
    logger.info("=" * 60)

    for i, qa in enumerate(cat_b):
        qid      = qa["id"]
        question = qa["question"]

        if qid in evaluated_ids:
            logger.info(f"[{i+1:02d}/{len(cat_b)}] {qid}: already in checkpoint — skipping")
            continue

        logger.info(f"[{i+1:02d}/{len(cat_b)}] {qid}: {question[:70]}...")

        response = _answer_with_retry(question)

        if response is None:
            result = QuestionResult(
                question_id=qid,
                category="B",
                question=question,
                should_answer=False,
                was_answered=True,
                has_citations=None,
                answer_preview="",
                sources_used=0,
                passed=False,
                failure_reason="All retries failed — transient error",
            )
        else:
            passed = not response.answered
            result = QuestionResult(
                question_id=qid,
                category="B",
                question=question,
                should_answer=False,
                was_answered=response.answered,
                has_citations=None,
                answer_preview=response.answer[:200],
                sources_used=len(response.sources),
                passed=passed,
                failure_reason="System answered an out-of-scope question" if response.answered else None,
            )

        status = "✓ PASS" if result.passed else "✗ FAIL"
        logger.info(f"  {status} | refused={not result.was_answered}")

        results.append(result)
        evaluated_ids.add(qid)
        _save_checkpoint(results)
        time.sleep(1)

    # metrics
    a_results = [r for r in results if r.category == "A"]
    b_results = [r for r in results if r.category == "B"]

    answered_count = sum(1 for r in a_results if r.was_answered)
    refused_count  = sum(1 for r in b_results if not r.was_answered)

    answer_rate       = answered_count / len(a_results) if a_results else 0.0
    refusal_precision = refused_count  / len(b_results) if b_results else 0.0

    cited = [r for r in a_results if r.was_answered and r.has_citations is not None]
    citation_rate = (
        sum(1 for r in cited if r.has_citations) / len(cited)
        if cited else 0.0
    )

    failure_reasons = []
    if answer_rate < ANSWER_RATE_THRESHOLD:
        failure_reasons.append(
            f"Answer rate {answer_rate:.1%} < threshold {ANSWER_RATE_THRESHOLD:.1%}"
        )
    if refusal_precision < REFUSAL_PRECISION_THRESHOLD:
        failure_reasons.append(
            f"Refusal precision {refusal_precision:.1%} < threshold {REFUSAL_PRECISION_THRESHOLD:.1%}"
        )
    if citation_rate < CITATION_RATE_THRESHOLD:
        failure_reasons.append(
            f"Citation rate {citation_rate:.1%} < threshold {CITATION_RATE_THRESHOLD:.1%}"
        )

    return EvalReport(
        timestamp=datetime.now(UTC).isoformat(),
        total_questions=len(results),
        category_a_count=len(a_results),
        category_b_count=len(b_results),
        answer_rate=answer_rate,
        refusal_precision=refusal_precision,
        citation_rate=citation_rate,
        results=results,
        passed=len(failure_reasons) == 0,
        failure_reasons=failure_reasons,
    )


# report writer

def save_report(report: EvalReport) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"eval_{timestamp}.json"

    report_dict = {
        "timestamp":       report.timestamp,
        "passed":          report.passed,
        "failure_reasons": report.failure_reasons,
        "metrics": {
            "answer_rate":       round(report.answer_rate, 4),
            "refusal_precision": round(report.refusal_precision, 4),
            "citation_rate":     round(report.citation_rate, 4),
        },
        "thresholds": {
            "answer_rate":       ANSWER_RATE_THRESHOLD,
            "refusal_precision": REFUSAL_PRECISION_THRESHOLD,
            "citation_rate":     CITATION_RATE_THRESHOLD,
        },
        "summary": {
            "total_questions": report.total_questions,
            "category_a":      report.category_a_count,
            "category_b":      report.category_b_count,
        },
        "results": [
            {
                "id":             r.question_id,
                "category":       r.category,
                "question":       r.question,
                "should_answer":  r.should_answer,
                "was_answered":   r.was_answered,
                "has_citations":  r.has_citations,
                "passed":         r.passed,
                "failure_reason": r.failure_reason,
                "answer_preview": r.answer_preview,
            }
            for r in report.results
        ],
    }

    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleared.")

    return report_path


# main

def main():
    logger.info("=" * 60)
    logger.info("Legal RAG — Evaluation Suite")
    logger.info("=" * 60)

    report      = run_evaluation()
    report_path = save_report(report)

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Answer Rate       : {report.answer_rate:.1%}  (need >={ANSWER_RATE_THRESHOLD:.0%})")
    logger.info(f"Refusal Precision : {report.refusal_precision:.1%}  (need >={REFUSAL_PRECISION_THRESHOLD:.0%})")
    logger.info(f"Citation Rate     : {report.citation_rate:.1%}  (need >={CITATION_RATE_THRESHOLD:.0%})")
    logger.info(f"Overall Result    : {'✓ PASSED' if report.passed else '✗ FAILED'}")

    if report.failure_reasons:
        logger.error("Failure reasons:")
        for reason in report.failure_reasons:
            logger.error(f"  - {reason}")

    logger.info(f"Report saved: {report_path}")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
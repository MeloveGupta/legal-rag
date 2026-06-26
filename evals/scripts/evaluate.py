"""
Evaluation script for the Legal RAG system.

Key improvements over v1:
- Judge uses llama-3.1-8b-instant (separate 100k TPD from the 70B RAG model)
- Resume capability: saves a checkpoint after each question and skips
  already-evaluated ones on restart - no more losing progress to rate limits
- Exceptions set faithfulness=None, not 0.0, so they don't corrupt averages
- Robust JSON extraction from judge responses
- Exponential backoff retry on 429 errors
- UTC datetime fix
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_groq import ChatGroq

from src.legal_rag.config.settings import GROQ_API_KEY, GROQ_MODEL_NAME
from src.legal_rag.generation.generator import RAGResponse, answer_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# quality thresholds
# CI fails if any metric drops below these values.
# Start conservative - raise them as the system matures.

ANSWER_RATE_THRESHOLD       = 0.80   # 80% of Category A must be answered
REFUSAL_PRECISION_THRESHOLD = 0.90   # 90% of Category B must be refused
FAITHFULNESS_THRESHOLD      = 0.70   # 70% avg faithfulness on answered questions

# models
# RAG model: llama-3.3-70b-versatile - 100k tokens/day (from settings)
# Judge model: llama-3.1-8b-instant - separate 100k tokens/day
# Together: ~200k tokens/day available, enough for the full eval suite.

JUDGE_MODEL = "llama-3.1-8b-instant"

# paths 

GOLDEN_DATASET_PATH = PROJECT_ROOT / "evals" / "golden_dataset" / "qa_pairs.json"
REPORTS_DIR         = PROJECT_ROOT / "evals" / "reports"
CHECKPOINT_PATH     = REPORTS_DIR / "checkpoint.json"


# data structures

@dataclass
class QuestionResult:
    question_id:        str
    category:           str
    question:           str
    should_answer:      bool
    was_answered:       bool
    faithfulness_score: Optional[float]
    answer_preview:     str
    sources_used:       int
    passed:             bool
    failure_reason:     Optional[str] = None


@dataclass
class EvalReport:
    timestamp:         str
    total_questions:   int
    category_a_count:  int
    category_b_count:  int
    answer_rate:       float
    refusal_precision: float
    avg_faithfulness:  float
    results:           List[QuestionResult]
    passed:            bool
    failure_reasons:   List[str] = field(default_factory=list)


# checkpoint helpers

def _load_checkpoint() -> dict:
    """
    Load previously evaluated results from checkpoint file.
    Returns dict keyed by question_id.
    """
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        logger.info(
            f"Checkpoint found: {len(data)} questions already evaluated. "
            f"Skipping those and resuming from where we left off."
        )
        return data
    return {}


def _save_checkpoint(results: List[QuestionResult]) -> None:
    """Persist current progress after every question."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        r.question_id: {
            "question_id":        r.question_id,
            "category":           r.category,
            "question":           r.question,
            "should_answer":      r.should_answer,
            "was_answered":       r.was_answered,
            "faithfulness_score": r.faithfulness_score,
            "answer_preview":     r.answer_preview,
            "sources_used":       r.sources_used,
            "passed":             r.passed,
            "failure_reason":     r.failure_reason,
        }
        for r in results
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


def _result_from_checkpoint(data: dict) -> QuestionResult:
    return QuestionResult(**data)


# rate limit retry

def _invoke_with_retry(
    llm: ChatGroq,
    prompt: str,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Invoke the LLM with exponential backoff on 429 rate limit errors.
    Returns the response text, or None if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait = 30 * (2 ** attempt)   # 30s → 60s → 120s
                logger.warning(
                    f"Rate limit hit on attempt {attempt + 1}/{max_retries}. "
                    f"Waiting {wait}s before retry..."
                )
                time.sleep(wait)
            else:
                logger.error(f"Non-rate-limit error (no retry): {e}")
                return None

    logger.error(f"All {max_retries} retries exhausted.")
    return None


# faithfulness scorer

def _score_faithfulness(
    query: str,
    answer: str,
    context_chunks: list,
    judge_llm: ChatGroq,
) -> Optional[float]:
    """
    Use the judge LLM to score whether the answer is grounded in the chunks.

    Returns float 0.0-1.0, or None if scoring fails for any reason.
    None is excluded from the faithfulness average - it does not count
    as a 0, which would unfairly penalise rate-limit casualties.
    """
    context_text = "\n\n".join([
        f"[Chunk {i+1}]: {chunk.page_content[:400]}"
        for i, chunk in enumerate(context_chunks)
    ])

    prompt = f"""You are an evaluation judge for a legal AI system.

TASK: Score whether the ANSWER below is faithful to the CONTEXT CHUNKS.
Faithful means every claim in the answer can be traced to the context.
Claims not in the context are hallucinations - score them down even if true.

SCORING GUIDE:
1.0  = Every claim is directly in the context
0.75 = Most claims in context, minor reasonable inference acceptable
0.5  = About half the claims are in the context
0.25 = Few claims supported by context
0.0  = Claims are not in the context at all

QUESTION:
{query}

CONTEXT CHUNKS:
{context_text}

ANSWER TO EVALUATE:
{answer}

CRITICAL: Respond with ONLY a raw JSON object. No markdown, no backticks, no explanation outside the JSON.
Format: {{"score": <number>, "reasoning": "<one sentence>"}}"""

    raw = _invoke_with_retry(judge_llm, prompt)
    if raw is None:
        logger.warning("Judge LLM returned None — skipping faithfulness score.")
        return None

    try:
        # find the JSON object - handles cases where model adds preamble
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning(f"No JSON object in judge response: '{raw[:150]}'")
            return None

        parsed = json.loads(raw[start:end])
        score = float(parsed["score"])
        reasoning = parsed.get("reasoning", "")
        logger.debug(f"Faithfulness: {score:.2f} | {reasoning}")
        return max(0.0, min(1.0, score))   # Clamp to [0.0, 1.0]

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse judge response: {e}. Raw: '{raw[:200]}'")
        return None


# core evaluation runner

def run_evaluation() -> EvalReport:
    """
    Run the full evaluation suite.
    Automatically resumes from checkpoint if one exists.
    """
    if not GOLDEN_DATASET_PATH.exists():
        logger.error(f"Golden dataset not found: {GOLDEN_DATASET_PATH}")
        sys.exit(1)

    with open(GOLDEN_DATASET_PATH) as f:
        dataset = json.load(f)

    qa_pairs = dataset["qa_pairs"]
    cat_a = [q for q in qa_pairs if q["category"] == "A"]
    cat_b = [q for q in qa_pairs if q["category"] == "B"]

    logger.info(f"Dataset       : {len(qa_pairs)} questions")
    logger.info(f"Category A    : {len(cat_a)} (must answer)")
    logger.info(f"Category B    : {len(cat_b)} (must refuse)")
    logger.info(f"RAG model     : {GROQ_MODEL_NAME}")
    logger.info(f"Judge model   : {JUDGE_MODEL}")
    logger.info(f"Thresholds    : answer>={ANSWER_RATE_THRESHOLD:.0%} | "
                f"refusal>={REFUSAL_PRECISION_THRESHOLD:.0%} | "
                f"faithfulness>={FAITHFULNESS_THRESHOLD:.0%}")

    # load checkpoint and restore already-evaluated results
    checkpoint = _load_checkpoint()
    results: List[QuestionResult] = [
        _result_from_checkpoint(v) for v in checkpoint.values()
    ]
    evaluated_ids = {r.question_id for r in results}

    judge_llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=JUDGE_MODEL,
        temperature=0,
        max_tokens=300,
    )

    # category A
    logger.info("=" * 60)
    logger.info("EVALUATING CATEGORY A - must answer")
    logger.info("=" * 60)

    for i, qa in enumerate(cat_a):
        qid      = qa["id"]
        question = qa["question"]

        if qid in evaluated_ids:
            logger.info(f"[{i+1:02d}/{len(cat_a)}] {qid}: already evaluated - skipping")
            continue

        logger.info(f"[{i+1:02d}/{len(cat_a)}] {qid}: {question[:70]}...")

        try:
            response: RAGResponse = answer_query(question)
            was_answered = response.answered

            faithfulness = None
            if was_answered and response.retrieved_chunks:
                faithfulness = _score_faithfulness(
                    query=question,
                    answer=response.answer,
                    context_chunks=response.retrieved_chunks,
                    judge_llm=judge_llm,
                )

            # pass criteria:
            # 1. System answered (did not refuse)
            # 2. If faithfulness was scored, it must be >= 0.5
            # If faithfulness is None (scoring failed), we don't penalise
            # the question - it counts as answered but unscored.
            if not was_answered:
                passed = False
                failure_reason = "System refused an answerable question"
            elif faithfulness is not None and faithfulness < 0.5:
                passed = False
                failure_reason = f"Low faithfulness: {faithfulness:.2f}"
            else:
                passed = True
                failure_reason = None

            result = QuestionResult(
                question_id=qid,
                category="A",
                question=question,
                should_answer=True,
                was_answered=was_answered,
                faithfulness_score=faithfulness,
                answer_preview=response.answer[:200],
                sources_used=len(response.sources),
                passed=passed,
                failure_reason=failure_reason,
            )

        except Exception as e:
            logger.error(f"  EXCEPTION: {e}")
            result = QuestionResult(
                question_id=qid,
                category="A",
                question=question,
                should_answer=True,
                was_answered=False,
                faithfulness_score=None,
                answer_preview="",
                sources_used=0,
                passed=False,
                failure_reason=f"Exception: {str(e)[:150]}",
            )

        status    = "✓ PASS" if result.passed else "✗ FAIL"
        faith_str = f"{result.faithfulness_score:.2f}" if result.faithfulness_score is not None else "unscored"
        logger.info(f"  {status} | answered={result.was_answered} | faith={faith_str}")

        results.append(result)
        evaluated_ids.add(qid)
        _save_checkpoint(results)

        # 4s between A questions: 2 LLM calls per question across 2 models.
        # 4s gives ~15 questions/min, well within the 30 req/min per model.
        time.sleep(4)

    # category B
    logger.info("=" * 60)
    logger.info("EVALUATING CATEGORY B - must refuse")
    logger.info("=" * 60)

    for i, qa in enumerate(cat_b):
        qid      = qa["id"]
        question = qa["question"]

        if qid in evaluated_ids:
            logger.info(f"[{i+1:02d}/{len(cat_b)}] {qid}: already evaluated - skipping")
            continue

        logger.info(f"[{i+1:02d}/{len(cat_b)}] {qid}: {question[:70]}...")

        try:
            response: RAGResponse = answer_query(question)
            was_answered  = response.answered
            passed        = not was_answered
            failure_reason = "System answered an out-of-scope question" if was_answered else None

            result = QuestionResult(
                question_id=qid,
                category="B",
                question=question,
                should_answer=False,
                was_answered=was_answered,
                faithfulness_score=None,
                answer_preview=response.answer[:200],
                sources_used=len(response.sources),
                passed=passed,
                failure_reason=failure_reason,
            )

        except Exception as e:
            logger.error(f"  EXCEPTION: {e}")
            result = QuestionResult(
                question_id=qid,
                category="B",
                question=question,
                should_answer=False,
                was_answered=True,
                faithfulness_score=None,
                answer_preview="",
                sources_used=0,
                passed=False,
                failure_reason=f"Exception: {str(e)[:150]}",
            )

        status = "✓ PASS" if result.passed else "✗ FAIL"
        logger.info(f"  {status} | refused={not result.was_answered}")

        results.append(result)
        evaluated_ids.add(qid)
        _save_checkpoint(results)
        time.sleep(2)

    # compute metrics
    a_results = [r for r in results if r.category == "A"]
    b_results = [r for r in results if r.category == "B"]

    answered_count = sum(1 for r in a_results if r.was_answered)
    refused_count  = sum(1 for r in b_results if not r.was_answered)

    answer_rate       = answered_count / len(a_results) if a_results else 0.0
    refusal_precision = refused_count  / len(b_results) if b_results else 0.0

    # only include questions where faithfulness was actually scored
    faith_scores = [
        r.faithfulness_score for r in a_results
        if r.faithfulness_score is not None
    ]
    avg_faithfulness = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0

    logger.info(f"Faithfulness scored on {len(faith_scores)}/{len(a_results)} Category A questions")

    failure_reasons = []
    if answer_rate < ANSWER_RATE_THRESHOLD:
        failure_reasons.append(
            f"Answer rate {answer_rate:.1%} < threshold {ANSWER_RATE_THRESHOLD:.1%}"
        )
    if refusal_precision < REFUSAL_PRECISION_THRESHOLD:
        failure_reasons.append(
            f"Refusal precision {refusal_precision:.1%} < threshold {REFUSAL_PRECISION_THRESHOLD:.1%}"
        )
    if avg_faithfulness < FAITHFULNESS_THRESHOLD:
        failure_reasons.append(
            f"Avg faithfulness {avg_faithfulness:.1%} < threshold {FAITHFULNESS_THRESHOLD:.1%}"
        )

    return EvalReport(
        timestamp=datetime.now(UTC).isoformat(),
        total_questions=len(results),
        category_a_count=len(a_results),
        category_b_count=len(b_results),
        answer_rate=answer_rate,
        refusal_precision=refusal_precision,
        avg_faithfulness=avg_faithfulness,
        results=results,
        passed=len(failure_reasons) == 0,
        failure_reasons=failure_reasons,
    )


# report writer

def save_report(report: EvalReport) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"eval_{timestamp}.json"

    faith_scores = [
        r.faithfulness_score for r in report.results
        if r.faithfulness_score is not None
    ]

    report_dict = {
        "timestamp":       report.timestamp,
        "passed":          report.passed,
        "failure_reasons": report.failure_reasons,
        "metrics": {
            "answer_rate":       round(report.answer_rate, 4),
            "refusal_precision": round(report.refusal_precision, 4),
            "avg_faithfulness":  round(report.avg_faithfulness, 4),
        },
        "thresholds": {
            "answer_rate":       ANSWER_RATE_THRESHOLD,
            "refusal_precision": REFUSAL_PRECISION_THRESHOLD,
            "faithfulness":      FAITHFULNESS_THRESHOLD,
        },
        "summary": {
            "total_questions":      report.total_questions,
            "category_a_count":     report.category_a_count,
            "category_b_count":     report.category_b_count,
            "faithfulness_scored":  len(faith_scores),
            "faithfulness_unscored": report.category_a_count - len(faith_scores),
        },
        "results": [
            {
                "id":             r.question_id,
                "category":       r.category,
                "question":       r.question,
                "should_answer":  r.should_answer,
                "was_answered":   r.was_answered,
                "faithfulness":   r.faithfulness_score,
                "passed":         r.passed,
                "failure_reason": r.failure_reason,
                "answer_preview": r.answer_preview,
            }
            for r in report.results
        ],
    }

    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    # clear checkpoint - full report is now saved
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleared - full report is the source of truth now.")

    return report_path


# main

def main():
    logger.info("=" * 60)
    logger.info("Legal RAG - Evaluation Suite")
    logger.info("=" * 60)

    report      = run_evaluation()
    report_path = save_report(report)

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Answer Rate       : {report.answer_rate:.1%}  (need >={ANSWER_RATE_THRESHOLD:.0%})")
    logger.info(f"Refusal Precision : {report.refusal_precision:.1%}  (need >={REFUSAL_PRECISION_THRESHOLD:.0%})")
    logger.info(f"Avg Faithfulness  : {report.avg_faithfulness:.1%}  (need >={FAITHFULNESS_THRESHOLD:.0%})")
    logger.info(f"Overall Result    : {'✓ PASSED' if report.passed else '✗ FAILED'}")

    if report.failure_reasons:
        logger.error("Failure reasons:")
        for reason in report.failure_reasons:
            logger.error(f"  - {reason}")

    logger.info(f"Report : {report_path}")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
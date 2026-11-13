"""Rubric Evaluator engine with 6 hard pass/fail checkpoints and dual-layer verification."""

import re
import logging
from typing import List, Tuple, Optional
from src.models import (
    CheckpointID,
    CheckpointMeta,
    CheckpointResult,
    RubricEvaluation
)
from src.prompts import EVALUATOR_SYSTEM_PROMPT
from src.llm_provider import BaseLLMProvider, get_llm_provider, MockProvider

logger = logging.getLogger(__name__)


class HeuristicVerifier:
    """Fast deterministic static checks to assist and anchor LLM evaluations."""

    @staticmethod
    def check_jargon(text: str) -> Tuple[bool, str, str]:
        """Detect technical jargon used without immediate layperson explanation."""
        text_lower = text.lower()
        jargon_rules = [
            {
                "term": "embeddings",
                "patterns": ["embedding", "embeddings"],
                "allowed_explanations": ["fingerprint", "number", "numerical", "coordinates", "vector representation", "meaning code"],
                "friendly_desc": "Embeddings mentioned without explaining them as numerical fingerprints or meaning codes."
            },
            {
                "term": "vector database",
                "patterns": ["vector database", "vector db", "vector store"],
                "allowed_explanations": ["filing cabinet", "library", "specialized database", "smart search", "smart storage", "cabinet"],
                "friendly_desc": "Vector database mentioned without explaining it as a smart filing cabinet or semantic library."
            },
            {
                "term": "chunking",
                "patterns": ["chunking", "chunks", "chunk"],
                "allowed_explanations": ["cut", "split", "index card", "bite-sized", "slice", "paragraphs", "pieces", "small sections"],
                "friendly_desc": "Chunking mentioned without explaining it as cutting documents into bite-sized index cards."
            },
            {
                "term": "dense vectors / hnsw / cosine similarity",
                "patterns": ["dense vector", "hnsw", "cosine similarity", "nearest neighbor", "high-dimensional"],
                "allowed_explanations": ["simple math", "measuring similarity", "matching ideas", "fingerprint"],
                "friendly_desc": "Advanced mathematical/algorithmic jargon used without beginner-friendly grounding."
            }
        ]

        for rule in jargon_rules:
            term_found = any(p in text_lower for p in rule["patterns"])
            if term_found:
                # Check if at least one friendly explanation pattern exists
                has_explanation = any(exp in text_lower for exp in rule["allowed_explanations"])
                if not has_explanation:
                    return False, f"Unexplained jargon detected: {rule['term']}.", rule["friendly_desc"]

        return True, "All technical terms are properly accompanied by layman explanations.", ""

    @staticmethod
    def check_example(text: str) -> Tuple[bool, str, str]:
        """Check for a concrete end-to-end example (query -> retrieval -> answer)."""
        text_lower = text.lower()
        has_query = any(q in text_lower for q in ["question:", "query:", "user asks", "user question", "asks:"])
        has_retrieval_or_doc = any(d in text_lower for d in ["retrieved", "manual", "document", "source snippet", "section", "chunk"])
        has_answer = any(a in text_lower for a in ["answer:", "responds:", "generated answer", "response:", "replies:"])

        if has_query and (has_retrieval_or_doc or has_answer):
            return True, "Contains concrete walkthrough showing query, reference material, and grounded response.", ""
        return False, "Lacks an explicit end-to-end walkthrough showing a user inquiry and how RAG retrieves and answers it.", "Add a concrete real-world scenario."

    @staticmethod
    def check_key_points(text: str) -> Tuple[bool, str, str]:
        """Verify presence of What, Why, and How."""
        text_lower = text.lower()
        has_what = "what is" in text_lower or "what rag is" in text_lower or "retrieval-augmented generation" in text_lower
        has_why = "why" in text_lower and any(w in text_lower for w in ["matter", "problem", "hallucinat", "limitation", "outdated"])
        has_how = "how" in text_lower and any(h in text_lower for h in ["work", "step", "retrieve", "journey", "process"])

        missing = []
        if not has_what:
            missing.append("What it is")
        if not has_why:
            missing.append("Why it matters / Problems it solves")
        if not has_how:
            missing.append("How it works (step-by-step)")

        if missing:
            return False, f"Missing foundational pillars: {', '.join(missing)}.", f"Ensure distinct sections covering {', '.join(missing)}."
        return True, "Covers all 3 foundational pillars (What, Why, How).", ""


class LessonEvaluator:
    """Rubric Evaluator executing the 6 hard pass/fail checkpoints."""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def evaluate(self, lesson_text: str, topic: str = "Introduction to RAG") -> RubricEvaluation:
        """Evaluates lesson text against all 6 checkpoints.
        
        If using a MockProvider, it produces deterministic evaluation.
        If using a live LLM, it executes structured evaluation and applies heuristic safeguards.
        """
        # If running with MockProvider, delegate directly
        if isinstance(self.llm, MockProvider):
            return self.llm.generate_structured(
                prompt=lesson_text,
                system_prompt=EVALUATOR_SYSTEM_PROMPT,
                response_model=RubricEvaluation
            )

        # Run Live LLM Structured Evaluation
        eval_prompt = (
            f"Please evaluate the following introductory lesson on '{topic}'.\n\n"
            f"LESSON CONTENT TO EVALUATE:\n"
            f"```markdown\n{lesson_text}\n```\n\n"
            f"Apply the strict 6-checkpoint rubric. Remember: NO PARTIAL CREDIT. "
            f"Each checkpoint must be strictly True (PASS) or False (FAIL)."
        )

        try:
            evaluation: RubricEvaluation = self.llm.generate_structured(
                prompt=eval_prompt,
                system_prompt=EVALUATOR_SYSTEM_PROMPT,
                response_model=RubricEvaluation
            )
        except Exception as e:
            logger.warning(f"Structured evaluation fallback due to: {e}")
            # Fallback to deterministic heuristic evaluation if LLM structured output fails
            evaluation = self._heuristic_fallback_evaluation(lesson_text, topic)

        # Apply Heuristic Safeguards (Ensure no false passes if glaring issues exist)
        evaluation = self._apply_heuristic_safeguards(lesson_text, evaluation)
        return evaluation

    def _apply_heuristic_safeguards(self, lesson_text: str, evaluation: RubricEvaluation) -> RubricEvaluation:
        """Guard against LLM lenient hallucination on hard checkpoints."""
        # Check Jargon
        jargon_ok, jargon_reason, jargon_desc = HeuristicVerifier.check_jargon(lesson_text)
        if not jargon_ok:
            for chk in evaluation.checkpoints:
                if chk.checkpoint_id == CheckpointID.NO_JARGON:
                    chk.passed = False
                    chk.reason = f"Heuristic Safeguard: {jargon_reason}"
                    chk.evidence = jargon_desc
                    chk.remediation = "Define all technical terms using intuitive everyday analogies before or upon use."

        # Check Example
        example_ok, example_reason, example_desc = HeuristicVerifier.check_example(lesson_text)
        if not example_ok:
            for chk in evaluation.checkpoints:
                if chk.checkpoint_id == CheckpointID.EXAMPLE:
                    chk.passed = False
                    chk.reason = f"Heuristic Safeguard: {example_reason}"
                    chk.evidence = "No concrete user query + document chunk + grounded answer found."
                    chk.remediation = "Provide an explicit end-to-end walkthrough scenario."

        # Check Key Points
        key_ok, key_reason, key_rem = HeuristicVerifier.check_key_points(lesson_text)
        if not key_ok:
            for chk in evaluation.checkpoints:
                if chk.checkpoint_id == CheckpointID.KEY_POINTS:
                    chk.passed = False
                    chk.reason = f"Heuristic Safeguard: {key_reason}"
                    chk.remediation = key_rem

        # Recalculate all_passed and failed_checkpoints
        failed = [c.checkpoint_id for c in evaluation.checkpoints if not c.passed]
        evaluation.failed_checkpoints = failed
        evaluation.all_passed = len(failed) == 0
        evaluation.pass_rate = (len(evaluation.checkpoints) - len(failed)) / len(evaluation.checkpoints)
        if not evaluation.all_passed:
            evaluation.overall_feedback = f"Failed {len(failed)} checkpoint(s): {', '.join(failed)}. Revision required."
        else:
            evaluation.overall_feedback = "All 6 checkpoints cleared. Ready to ship!"

        return evaluation

    def _heuristic_fallback_evaluation(self, lesson_text: str, topic: str) -> RubricEvaluation:
        """Deterministic fallback evaluation if LLM provider is unreachable."""
        jargon_ok, jargon_reason, jargon_evidence = HeuristicVerifier.check_jargon(lesson_text)
        example_ok, example_reason, _ = HeuristicVerifier.check_example(lesson_text)
        key_ok, key_reason, key_rem = HeuristicVerifier.check_key_points(lesson_text)

        text_lower = lesson_text.lower()
        beginner_ok = "open-book" in text_lower or "library" in text_lower or "exam" in text_lower
        accuracy_ok = "retrieval" in text_lower and "generat" in text_lower
        flow_ok = "#" in lesson_text and ("summary" in text_lower or "takeaway" in text_lower)

        checkpoints = [
            CheckpointResult(
                checkpoint_id=CheckpointID.ACCURACY,
                name="Accurate & Grounded",
                passed=accuracy_ok,
                reason="Accurate description of retrieval-augmented architecture." if accuracy_ok else "Inaccurate or missing definition.",
                evidence="Accurate representation." if accuracy_ok else "Missing key terms.",
                remediation="" if accuracy_ok else "Ensure retrieval is cleanly separated from generation."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.BEGINNER,
                name="Beginner-Friendly Language",
                passed=beginner_ok,
                reason="Uses everyday metaphors accessible to beginners." if beginner_ok else "Lacks intuitive beginner framing.",
                evidence="Metaphor identified." if beginner_ok else "No foundational analogy.",
                remediation="" if beginner_ok else "Add an open-book exam or library metaphor in the introduction."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.EXAMPLE,
                name="Teaches by Concrete Example",
                passed=example_ok,
                reason=example_reason,
                evidence="End-to-end walkthrough present." if example_ok else "Missing concrete scenario.",
                remediation="" if example_ok else "Include an explicit query -> retrieval -> response walkthrough."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.NO_JARGON,
                name="Clear, No Unexplained Jargon",
                passed=jargon_ok,
                reason=jargon_reason,
                evidence=jargon_evidence if not jargon_ok else "All terms accompanied by layman explanations.",
                remediation="" if jargon_ok else "Define embeddings as numerical fingerprints and vector DB as smart filing cabinet."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.KEY_POINTS,
                name="Covers All Key Pillars",
                passed=key_ok,
                reason=key_reason,
                evidence="Foundational pillars verified." if key_ok else "Missing foundational pillar.",
                remediation="" if key_ok else key_rem
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.COHERENT_FLOW,
                name="Coherent Pedagogical Flow",
                passed=flow_ok,
                reason="Clean hierarchical structure with markdown headers." if flow_ok else "Poor formatting or flow.",
                evidence="Headers and structure verified." if flow_ok else "Missing section structure.",
                remediation="" if flow_ok else "Add structured section headers and concluding takeaways."
            )
        ]

        failed = [c.checkpoint_id for c in checkpoints if not c.passed]
        return RubricEvaluation(
            checkpoints=checkpoints,
            all_passed=len(failed) == 0,
            failed_checkpoints=failed,
            overall_feedback="Deterministic heuristic evaluation completed.",
            pass_rate=(len(checkpoints) - len(failed)) / len(checkpoints)
        )


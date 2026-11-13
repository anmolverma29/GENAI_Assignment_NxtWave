"""Lesson Generator module handling initial drafting and feedback-conditioned revisions."""

import logging
from typing import List, Optional
from src.llm_provider import BaseLLMProvider, get_llm_provider
from src.prompts import GENERATOR_SYSTEM_PROMPT, GENERATOR_REVISION_TEMPLATE
from src.models import RubricEvaluation

logger = logging.getLogger(__name__)


class LessonGenerator:
    """Generates and iteratively refines educational lessons."""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def generate_initial_draft(self, topic: str, evolved_guidelines: Optional[List[str]] = None) -> str:
        """Generates the initial draft of a lesson for a given topic."""
        guidelines_section = ""
        if evolved_guidelines:
            guidelines_section = (
                "\n\nLEARNED SYSTEM GUIDELINES (Must adhere to these best practices learned from past runs):\n" +
                "\n".join(f"- {g}" for g in evolved_guidelines)
            )

        prompt = (
            f"Please generate a complete, standalone beginner lesson on the topic: '{topic}'.\n"
            f"Assume the reader starts from zero and has no technical or programming background.\n"
            f"{guidelines_section}\n\n"
            f"Deliver a structured, engaging lesson covering: What it is, Why it matters, and How it works."
        )

        return self.llm.generate_text(
            prompt=prompt,
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            temperature=0.7
        )

    def revise_draft(
        self,
        topic: str,
        previous_draft: str,
        evaluation: RubricEvaluation,
        attempt: int,
        evolved_guidelines: Optional[List[str]] = None
    ) -> str:
        """Revises and regenerates a draft specifically addressing failed checkpoints."""
        # Format failed checkpoints summary
        failed_summary = "\n".join(
            f"- {chk.name} ({chk.checkpoint_id.value}): FAILED. Reason: {chk.reason}"
            for chk in evaluation.checkpoints if not chk.passed
        )

        # Format detailed remediation instructions
        detailed_remediation = "\n".join(
            f"[{chk.name}]:\n  * Issue: {chk.evidence}\n  * Remediation Required: {chk.remediation}"
            for chk in evaluation.checkpoints if not chk.passed
        )

        # Format evolved guidelines
        guidelines_text = (
            "\n".join(f"- {g}" for g in evolved_guidelines)
            if evolved_guidelines else "None recorded yet."
        )

        prompt = GENERATOR_REVISION_TEMPLATE.format(
            topic=topic,
            attempt=attempt,
            retry_count=attempt - 1,
            previous_draft=previous_draft,
            failed_checkpoints_summary=failed_summary,
            detailed_remediation=detailed_remediation,
            evolved_guidelines=guidelines_text
        )

        return self.llm.generate_text(
            prompt=prompt,
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            temperature=0.5
        )


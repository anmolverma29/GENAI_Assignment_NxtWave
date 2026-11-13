"""Data schemas and models for the Self-Evaluating Lesson Content Generator."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CheckpointID(str, Enum):
    """The 6 mandatory pass/fail rubric checkpoints."""
    ACCURACY = "CHK_ACCURACY"
    BEGINNER = "CHK_BEGINNER"
    EXAMPLE = "CHK_EXAMPLE"
    NO_JARGON = "CHK_NO_JARGON"
    KEY_POINTS = "CHK_KEY_POINTS"
    COHERENT_FLOW = "CHK_COHERENT_FLOW"


class CheckpointMeta:
    """Friendly metadata for each checkpoint."""
    DESCRIPTIONS = {
        CheckpointID.ACCURACY: {
            "name": "Accurate & Grounded",
            "dimension": "accurate & grounded",
            "desc": "Content is factually accurate, modern, and free from misleading definitions or hallucinations."
        },
        CheckpointID.BEGINNER: {
            "name": "Beginner-Friendly Language",
            "dimension": "beginner-friendly language",
            "desc": "Tone is accessible and empathetic; assumes zero prior AI/ML background; uses relatable intuition."
        },
        CheckpointID.EXAMPLE: {
            "name": "Teaches by Concrete Example",
            "dimension": "teaches by example",
            "desc": "Includes at least one complete end-to-end real-world walkthrough illustrating query, retrieval, and generation."
        },
        CheckpointID.NO_JARGON: {
            "name": "Clear, No Unexplained Jargon",
            "dimension": "clear, no unexplained jargon",
            "desc": "Every technical term (embeddings, vector database, chunks, etc.) is introduced and defined in plain English before or upon use."
        },
        CheckpointID.KEY_POINTS: {
            "name": "Covers All Key Pillars",
            "dimension": "covers the key points",
            "desc": "Explicitly covers what it is, why it matters/problem it solves, and how it works step-by-step."
        },
        CheckpointID.COHERENT_FLOW: {
            "name": "Coherent Pedagogical Flow",
            "dimension": "coherent teaching flow",
            "desc": "Has structured, logical progression: Hook -> Concept -> Problem -> 3-Step Mechanism -> Concrete Example -> Summary."
        }
    }


class CheckpointResult(BaseModel):
    """Result of a single hard pass/fail checkpoint evaluation."""
    checkpoint_id: CheckpointID = Field(description="Unique identifier for the checkpoint")
    name: str = Field(description="Human readable name of checkpoint")
    passed: bool = Field(description="Hard binary decision: True for PASS, False for FAIL (no partial credit)")
    reason: str = Field(description="Detailed explanation of why it passed or failed")
    evidence: str = Field(description="Specific quote or missing element identified in the lesson text")
    remediation: str = Field(default="", description="Specific actionable instruction on how to fix if failed")


class RubricEvaluation(BaseModel):
    """Full rubric evaluation covering all checkpoints."""
    checkpoints: List[CheckpointResult] = Field(description="List of all 6 checkpoint evaluations")
    all_passed: bool = Field(description="True only if EVERY single checkpoint passed")
    failed_checkpoints: List[CheckpointID] = Field(default_factory=list, description="List of checkpoint IDs that failed")
    overall_feedback: str = Field(description="Synthesized feedback for the generator")
    pass_rate: float = Field(default=0.0, description="Percentage of passed checkpoints (0.0 to 1.0)")


class RejectionLogEntry(BaseModel):
    """Structured audit entry for an evaluation rejection."""
    attempt: int = Field(description="Attempt number that failed (1-indexed)")
    timestamp: str = Field(description="ISO timestamp of rejection")
    failed_checkpoints: List[str] = Field(description="Names or IDs of checkpoints that failed")
    critique_summary: str = Field(description="Concise summary of reasons for rejection")
    detailed_failures: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Detailed list of failures: checkpoint, reason, evidence, remediation"
    )
    remediation_plan: List[str] = Field(
        default_factory=list,
        description="Prescriptive action items given to the generator"
    )
    changes_made_in_next_draft: Optional[str] = Field(
        default=None,
        description="What the generator changed in the subsequent draft to address this rejection"
    )


class EvolvedGuideline(BaseModel):
    """A learned guideline stored in persistent memory across runs."""
    id: str = Field(description="Unique rule ID, e.g. RULE-JARGON-01")
    target_checkpoint: CheckpointID = Field(description="Associated checkpoint")
    rule: str = Field(description="Actionable rule text for the generator")
    anti_pattern: str = Field(description="What negative pattern to avoid")
    best_practice: str = Field(description="What positive pattern to use")
    learned_from_topic: str = Field(description="Topic where this rule was learned")
    frequency_triggered: int = Field(default=1, description="How many times this failure was encountered")


class RunMemoryRecord(BaseModel):
    """Historical record for a single pipeline execution."""
    run_id: str = Field(description="Unique run ID")
    timestamp: str = Field(description="ISO timestamp")
    topic: str = Field(description="Lesson topic")
    status: str = Field(description="APPROVED or MAX_RETRIES_REACHED")
    total_attempts: int = Field(description="Number of attempts taken")
    final_pass_rate: float = Field(description="Final evaluation pass rate")
    rejection_count: int = Field(description="Number of rejections before approval")
    rejection_log: List[RejectionLogEntry] = Field(default_factory=list)
    final_lesson: str = Field(description="Final generated lesson text")


class PipelineOutput(BaseModel):
    """Final packaged response of the agentic pipeline."""
    topic: str
    status: str  # "APPROVED" or "MAX_RETRIES_REACHED"
    total_attempts: int
    final_lesson: str
    rejection_log: List[RejectionLogEntry]
    final_evaluation: RubricEvaluation
    evolved_guidelines_used: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


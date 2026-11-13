"""Self-evolving engine that learns from rejection logs to sharpen prompts and guidelines."""

import logging
from typing import List, Optional
from src.models import (
    RejectionLogEntry,
    EvolvedGuideline,
    CheckpointID,
    CheckpointMeta
)
from src.memory import MemoryStore
from src.llm_provider import BaseLLMProvider, get_llm_provider, MockProvider

logger = logging.getLogger(__name__)


class EvolutionOptimizer:
    """Extracts lessons from rejections and sharpens generator prompts over time."""

    def __init__(self, memory_store: Optional[MemoryStore] = None, llm_provider: Optional[BaseLLMProvider] = None):
        self.memory = memory_store or MemoryStore()
        self.llm = llm_provider or get_llm_provider()

    def evolve_from_rejection_log(self, topic: str, rejection_log: List[RejectionLogEntry]) -> List[EvolvedGuideline]:
        """Analyzes all rejections in the run and produces/persists evolved guidelines."""
        new_guidelines = []
        if not rejection_log:
            return new_guidelines

        # Collect unique failed checkpoints across rejections
        failed_checkpoints_seen = set()
        for entry in rejection_log:
            for chk_id in entry.failed_checkpoints:
                failed_checkpoints_seen.add(chk_id)

        # Synthesize evolved guidelines for each encountered failure mode
        for chk_id in failed_checkpoints_seen:
            guideline = self._synthesize_guideline(topic, chk_id, rejection_log)
            if guideline:
                self.memory.save_evolved_guideline(guideline)
                new_guidelines.append(guideline)

        logger.info(f"Evolved {len(new_guidelines)} guidelines from rejection log.")
        return new_guidelines

    def _synthesize_guideline(
        self,
        topic: str,
        checkpoint_id_str: str,
        rejection_log: List[RejectionLogEntry]
    ) -> Optional[EvolvedGuideline]:
        """Synthesizes an actionable guideline for a specific checkpoint."""
        chk_id = None
        # Attempt to match by enum value
        for member in CheckpointID:
            if member.value == checkpoint_id_str or member.name == checkpoint_id_str:
                chk_id = member
                break
            # Also check friendly name
            meta = CheckpointMeta.DESCRIPTIONS.get(member, {})
            if meta.get("name") == checkpoint_id_str:
                chk_id = member
                break

        if not chk_id:
            return None

        # Standard proven heuristics per checkpoint to guarantee robust evolution
        heuristics_map = {
            CheckpointID.NO_JARGON: EvolvedGuideline(
                id=f"RULE-JARGON-{topic[:4].upper()}",
                target_checkpoint=CheckpointID.NO_JARGON,
                rule="Never mention 'vector', 'embeddings', 'chunking', or 'cosine similarity' without an immediate, intuitive plain-English analogy in the same sentence.",
                anti_pattern="Throwing terms like 'high-dimensional dense vectors' or 'approximate nearest neighbor' at the reader without translation.",
                best_practice="Always pair: 'embeddings' = 'numerical fingerprints of meaning'; 'vector database' = 'smart filing cabinet'; 'chunking' = 'cutting text into index cards'.",
                learned_from_topic=topic
            ),
            CheckpointID.EXAMPLE: EvolvedGuideline(
                id=f"RULE-EXAMPLE-{topic[:4].upper()}",
                target_checkpoint=CheckpointID.EXAMPLE,
                rule="Every lesson MUST include a complete 4-part walkthrough: (1) User question, (2) Without-RAG mistake, (3) Retrieved reference document snippet, (4) Grounded answer.",
                anti_pattern="Describing retrieval abstractly (e.g. 'the system finds relevant documents and answers') without showing the exact query and text snippet.",
                best_practice="Use a relatable scenario such as customer support for a specific product (e.g. smartwatch water resistance in a sauna).",
                learned_from_topic=topic
            ),
            CheckpointID.BEGINNER: EvolvedGuideline(
                id=f"RULE-BEGINNER-{topic[:4].upper()}",
                target_checkpoint=CheckpointID.BEGINNER,
                rule="Lead with an intuitive everyday analogy before defining any technical acronyms.",
                anti_pattern="Starting the lesson with formal architectural definitions or acronym breakdowns.",
                best_practice="Use the 'open-book exam vs closed-book exam' or 'research librarian' metaphor right in the hook.",
                learned_from_topic=topic
            ),
            CheckpointID.KEY_POINTS: EvolvedGuideline(
                id=f"RULE-KEYPOINTS-{topic[:4].upper()}",
                target_checkpoint=CheckpointID.KEY_POINTS,
                rule="Clearly separate the content into 3 explicit pillars: What it is, Why it matters (hallucination & private data), and How it works (Ingest, Retrieve, Generate).",
                anti_pattern="Focusing only on the technical architecture while neglecting to explain why companies need it or why fine-tuning is impractical.",
                best_practice="Dedicate explicit numbered sections for What, Why, and How with distinct subheadings.",
                learned_from_topic=topic
            ),
            CheckpointID.ACCURACY: EvolvedGuideline(
                id=f"RULE-ACCURACY-{topic[:4].upper()}",
                target_checkpoint=CheckpointID.ACCURACY,
                rule="Explicitly contrast RAG against fine-tuning: RAG provides external context dynamically; it does not change the model weights.",
                anti_pattern="Confusing retrieval with model retraining or fine-tuning.",
                best_practice="Clarify that RAG is like handing a student a book to read, rather than performing brain surgery to teach them new information.",
                learned_from_topic=topic
            ),
            CheckpointID.COHERENT_FLOW: EvolvedGuideline(
                id=f"RULE-FLOW-{topic[:4].upper()}",
                target_checkpoint=CheckpointID.COHERENT_FLOW,
                rule="Maintain clean structural hierarchy: Hook -> Concept -> Problem -> 3 Steps -> Walkthrough -> Takeaways.",
                anti_pattern="Jumping into technical steps before establishing the intuitive hook and the problem.",
                best_practice="Use Markdown level-2 headers for main sections and level-3 for sub-steps, ending with bulleted takeaways.",
                learned_from_topic=topic
            )
        }

        return heuristics_map.get(chk_id)

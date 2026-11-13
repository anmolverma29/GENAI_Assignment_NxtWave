"""Unit tests for the self-evolving optimizer module."""

import pytest
from src.models import (
    RejectionLogEntry,
    CheckpointID,
    EvolvedGuideline
)
from src.memory import MemoryStore
from src.evolution import EvolutionOptimizer


@pytest.fixture
def temp_evolution(tmp_path):
    history_file = tmp_path / "test_history.json"
    guidelines_file = tmp_path / "test_guidelines.json"
    memory = MemoryStore(history_path=history_file, guidelines_path=guidelines_file)
    return EvolutionOptimizer(memory_store=memory), memory


def test_evolution_from_rejection_log(temp_evolution):
    """Verify that rejection logs trigger synthesis of actionable rules and anti-patterns."""
    optimizer, memory = temp_evolution

    rejection_entry = RejectionLogEntry(
        attempt=1,
        timestamp="2026-11-13T22:00:00",
        failed_checkpoints=["CHK_NO_JARGON", "CHK_EXAMPLE"],
        critique_summary="Failed due to jargon and lack of concrete walkthrough.",
        detailed_failures=[],
        remediation_plan=["Define embeddings simply", "Add walkthrough"]
    )

    guidelines = optimizer.evolve_from_rejection_log(
        topic="Introduction to RAG",
        rejection_log=[rejection_entry]
    )

    assert len(guidelines) == 2
    target_checkpoints = {g.target_checkpoint for g in guidelines}
    assert CheckpointID.NO_JARGON in target_checkpoints
    assert CheckpointID.EXAMPLE in target_checkpoints

    # Verify they were saved to memory store
    saved_rules = memory.get_evolved_guidelines()
    assert len(saved_rules) == 2

    # Check active text generation for prompt injection
    active_text = memory.get_active_guidelines_text()
    assert len(active_text) == 2
    assert any("Avoid:" in t for t in active_text)
    assert any("Best Practice:" in t for t in active_text)


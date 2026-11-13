"""Unit tests for persistent memory store and cross-run tracking."""

import pytest
from pathlib import Path
from src.models import (
    RunMemoryRecord,
    EvolvedGuideline,
    RejectionLogEntry,
    CheckpointID
)
from src.memory import MemoryStore


@pytest.fixture
def temp_memory(tmp_path):
    """Provides a temporary MemoryStore isolated from production data."""
    history_file = tmp_path / "test_history.json"
    guidelines_file = tmp_path / "test_guidelines.json"
    return MemoryStore(history_path=history_file, guidelines_path=guidelines_file)


def test_save_and_retrieve_runs(temp_memory):
    """Test saving a run record and loading it back."""
    record = RunMemoryRecord(
        run_id="run_test_001",
        timestamp="2026-11-13T22:00:00",
        topic="Introduction to RAG",
        status="APPROVED",
        total_attempts=2,
        final_pass_rate=1.0,
        rejection_count=1,
        rejection_log=[
            RejectionLogEntry(
                attempt=1,
                timestamp="2026-11-13T22:01:00",
                failed_checkpoints=["Clear, No Unexplained Jargon"],
                critique_summary="Used dense jargon without explanation",
                detailed_failures=[],
                remediation_plan=["Define embeddings simply"]
            )
        ],
        final_lesson="# Passing Lesson Content"
    )

    temp_memory.save_run(record)
    runs = temp_memory.get_all_runs()

    assert len(runs) == 1
    assert runs[0].run_id == "run_test_001"
    assert runs[0].status == "APPROVED"
    assert runs[0].rejection_count == 1


def test_save_and_update_guidelines(temp_memory):
    """Test saving evolved guidelines and frequency increments."""
    guideline = EvolvedGuideline(
        id="RULE-JARGON-TEST",
        target_checkpoint=CheckpointID.NO_JARGON,
        rule="Explain all technical terms with everyday metaphors.",
        anti_pattern="Throwing dense jargon at the user.",
        best_practice="Pair embeddings with numerical fingerprint analogy.",
        learned_from_topic="Introduction to RAG"
    )

    temp_memory.save_evolved_guideline(guideline)
    guidelines = temp_memory.get_evolved_guidelines()
    assert len(guidelines) == 1
    assert guidelines[0].frequency_triggered == 1

    # Save again - should increment frequency rather than duplicating
    temp_memory.save_evolved_guideline(guideline)
    guidelines_updated = temp_memory.get_evolved_guidelines()
    assert len(guidelines_updated) == 1
    assert guidelines_updated[0].frequency_triggered == 2


def test_checkpoint_failure_frequencies(temp_memory):
    """Test aggregating failure frequencies across multiple historical runs."""
    entry1 = RejectionLogEntry(
        attempt=1,
        timestamp="2026-11-13T22:00:00",
        failed_checkpoints=["CHK_NO_JARGON", "CHK_EXAMPLE"],
        critique_summary="Failed jargon and example",
        detailed_failures=[],
        remediation_plan=[]
    )
    run1 = RunMemoryRecord(
        run_id="run_1",
        timestamp="2026-11-13T22:00:00",
        topic="Topic 1",
        status="APPROVED",
        total_attempts=2,
        final_pass_rate=1.0,
        rejection_count=1,
        rejection_log=[entry1],
        final_lesson="Content"
    )

    entry2 = RejectionLogEntry(
        attempt=1,
        timestamp="2026-11-13T22:05:00",
        failed_checkpoints=["CHK_NO_JARGON"],
        critique_summary="Failed jargon",
        detailed_failures=[],
        remediation_plan=[]
    )
    run2 = RunMemoryRecord(
        run_id="run_2",
        timestamp="2026-11-13T22:05:00",
        topic="Topic 2",
        status="APPROVED",
        total_attempts=2,
        final_pass_rate=1.0,
        rejection_count=1,
        rejection_log=[entry2],
        final_lesson="Content"
    )

    temp_memory.save_run(run1)
    temp_memory.save_run(run2)

    stats = temp_memory.get_checkpoint_failure_frequencies()
    assert stats["CHK_NO_JARGON"] == 2
    assert stats["CHK_EXAMPLE"] == 1
    assert stats["CHK_ACCURACY"] == 0


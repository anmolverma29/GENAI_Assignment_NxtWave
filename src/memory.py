"""Cross-run persistence and memory store for runs, evaluations, and evolved guidelines."""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.config import Config
from src.models import (
    RunMemoryRecord,
    EvolvedGuideline,
    CheckpointID
)

logger = logging.getLogger(__name__)


class MemoryStore:
    """Persistent storage manager using structured JSON storage."""

    def __init__(
        self,
        history_path: Optional[Path] = None,
        guidelines_path: Optional[Path] = None
    ):
        self.history_path = history_path or Config.RUN_HISTORY_PATH
        self.guidelines_path = guidelines_path or Config.EVOLVED_GUIDELINES_PATH
        self._ensure_files()

    def _ensure_files(self):
        """Ensures persistence files exist with valid initial structure."""
        if not self.history_path.exists():
            self.history_path.write_text("[]", encoding="utf-8")
        if not self.guidelines_path.exists():
            self.guidelines_path.write_text("[]", encoding="utf-8")

    def save_run(self, record: RunMemoryRecord) -> None:
        """Saves a run record to persistent storage."""
        runs = self.get_all_runs()
        runs.append(record)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in runs], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved run {record.run_id} to {self.history_path}")

    def get_all_runs(self) -> List[RunMemoryRecord]:
        """Retrieves all historical run records."""
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [RunMemoryRecord.model_validate(item) for item in data]
        except Exception as e:
            logger.error(f"Error loading runs from memory: {e}")
            return []

    def save_evolved_guideline(self, guideline: EvolvedGuideline) -> None:
        """Persists or updates an evolved guideline."""
        guidelines = self.get_evolved_guidelines()
        
        # Check if rule with same target or ID exists
        existing = next((g for g in guidelines if g.id == guideline.id), None)
        if existing:
            existing.frequency_triggered += 1
            existing.rule = guideline.rule
            existing.best_practice = guideline.best_practice
            existing.anti_pattern = guideline.anti_pattern
        else:
            guidelines.append(guideline)

        with open(self.guidelines_path, "w", encoding="utf-8") as f:
            json.dump([g.model_dump() for g in guidelines], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evolved guideline {guideline.id} to {self.guidelines_path}")

    def get_evolved_guidelines(self) -> List[EvolvedGuideline]:
        """Retrieves all evolved guidelines."""
        try:
            with open(self.guidelines_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [EvolvedGuideline.model_validate(item) for item in data]
        except Exception as e:
            logger.error(f"Error loading evolved guidelines: {e}")
            return []

    def get_active_guidelines_text(self) -> List[str]:
        """Returns plain strings of active learned rules for injection into prompts."""
        guidelines = self.get_evolved_guidelines()
        return [
            f"[{g.target_checkpoint.value}] Rule: {g.rule} | Avoid: {g.anti_pattern} | Best Practice: {g.best_practice}"
            for g in guidelines
        ]

    def get_checkpoint_failure_frequencies(self) -> Dict[str, int]:
        """Aggregates failure frequencies per checkpoint across all past runs."""
        stats = {c.value: 0 for c in CheckpointID}
        runs = self.get_all_runs()
        for r in runs:
            for entry in r.rejection_log:
                for failed in entry.failed_checkpoints:
                    if failed in stats:
                        stats[failed] += 1
                    else:
                        stats[failed] = stats.get(failed, 0) + 1
        return stats

    def clear(self) -> None:
        """Resets memory files (useful for clean test runs)."""
        self.history_path.write_text("[]", encoding="utf-8")
        self.guidelines_path.write_text("[]", encoding="utf-8")


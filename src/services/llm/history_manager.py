import logging
from pathlib import Path
from typing import List

import dill
from google.ai.generativelanguage_v1 import Content

logger = logging.getLogger(__name__)


class LLMHistoryManager:
    """Manages chat history persistence."""

    def __init__(self, history_file: Path, enabled: bool = False) -> None:
        self.history_file = history_file
        self.enabled = enabled

    def save(self, history: List[Content]) -> None:
        if isinstance(history, list):
            del history[:len(history) - 512]  # max items is 512

        if not self.enabled:
            return None

        try:
            with open(self.history_file, 'wb') as f:
                dill.dump(history, f)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            self._cleanup_failed_save()

    def load(self) -> List[Content]:
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'rb') as f:
                history = dill.load(f)
                return history
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            self._cleanup_corrupted_file()
            return []

    def _cleanup_failed_save(self) -> None:
        if self.history_file.exists():
            try:
                self.history_file.unlink()
            except Exception as e:
                logger.error(f"Failed to cleanup history file: {e}")

    def _cleanup_corrupted_file(self) -> None:
        try:
            self.history_file.unlink()
        except Exception as e:
            logger.error(f"Failed to cleanup corrupted file: {e}")

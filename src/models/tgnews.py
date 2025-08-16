from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


class TelegramPost(BaseModel):
    """Pydantic model for Telegram post data"""
    text: str
    date: str
    channel: Optional[str] = None  # Will be populated later

    # Optional computed property for a datetime object
    @property
    def datetime(self) -> Optional[datetime]:
        try:
            if 'T' in self.date:  # ISO format with timezone
                return datetime.fromisoformat(self.date)
            else:  # Simple format
                return datetime.strptime(self.date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

import json

from typing import Any, Dict, Set
from pathlib import Path


class BotSettings:
    """Class for bot's storage with JSON persistence, supporting per-chat settings"""
    _instance = None
    _storage_file = Path("storage.json")
    _default_settings = {
        "intervention_probability": 50,
        "chat_actions": True,
        "display_errors": True,
    }

    _private_chat_allowed_settings: Set[str] = {"chat_actions", "display_errors"}
    _public_chat_allowed_settings: Set[str] = set(_default_settings.keys())

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        """Initialize settings - will only be called once due to singleton pattern"""
        if not hasattr(self, '_all_settings'):
            self._all_settings = self._load_all_settings()

    @classmethod
    def _load_all_settings(cls) -> Dict[int, Dict[str, Any]]:
        """
        Load all chat settings from JSON file or return empty dict if the file doesn't exist or is invalid
        """
        try:
            if cls._storage_file.exists():
                with open(cls._storage_file, 'r') as f:
                    loaded_settings = json.load(f)
                    if isinstance(loaded_settings, dict):
                        result = {}
                        for chat_id, chat_settings in loaded_settings.items():
                            try:
                                chat_id_int = int(chat_id)
                                if isinstance(chat_settings, dict):
                                    allowed_keys = (
                                        cls._private_chat_allowed_settings
                                        if chat_id_int > 0
                                        else cls._public_chat_allowed_settings
                                    )
                                    filtered_settings = {
                                        k: v for k, v in chat_settings.items()
                                        if k in cls._default_settings and k in allowed_keys
                                    }
                                    if filtered_settings:
                                        result[chat_id_int] = filtered_settings
                            except ValueError:
                                continue
                        return result
        except (json.JSONDecodeError, IOError, PermissionError, ValueError):
            pass
        return {}

    def _save_all_settings(self) -> None:
        """Save all settings to JSON file"""
        try:
            with open(self._storage_file, 'w') as f:
                json.dump(self._all_settings, f, indent=4)
        except (IOError, PermissionError):
            pass

    def _get_chat_settings(self, chat_id: int) -> Dict[str, Any]:
        """Get settings for specific chat, creating with defaults if not exists"""
        if chat_id not in self._all_settings:
            self._all_settings[chat_id] = self._get_default_settings_for_chat(chat_id)
            self._save_all_settings()
        return self._all_settings[chat_id]

    def _get_default_settings_for_chat(self, chat_id: int) -> Dict[str, Any]:
        """Get default settings filtered for the chat type (private/public)"""
        is_private = chat_id > 0
        allowed_keys = (
            self._private_chat_allowed_settings
            if is_private
            else self._public_chat_allowed_settings
        )
        return {
            k: v for k, v in self._default_settings.items()
            if k in allowed_keys
        }

    def _is_setting_allowed(self, chat_id: int, setting_name: str) -> bool:
        """Check if a setting is allowed for this chat type"""
        is_private = chat_id > 0
        allowed_settings = (
            self._private_chat_allowed_settings
            if is_private
            else self._public_chat_allowed_settings
        )
        return setting_name in allowed_settings

    def get_intervention_probability(self, chat_id: int) -> int:
        if chat_id > 0:
            return 0
        return self._get_chat_settings(chat_id).get("intervention_probability",
                                                    self._default_settings["intervention_probability"])

    def set_intervention_probability(self, chat_id: int, value: int) -> None:
        if chat_id > 0:
            return
        settings = self._get_chat_settings(chat_id)
        settings["intervention_probability"] = max(0, min(100, value))
        self._save_all_settings()

    def get_chat_actions(self, chat_id: int) -> bool:
        return self._get_chat_settings(chat_id).get("chat_actions", self._default_settings["chat_actions"])

    def set_chat_actions(self, chat_id: int, value: bool) -> None:
        if not self._is_setting_allowed(chat_id, "chat_actions"):
            return
        settings = self._get_chat_settings(chat_id)
        settings["chat_actions"] = value
        self._save_all_settings()

    def get_display_errors(self, chat_id: int) -> bool:
        return self._get_chat_settings(chat_id).get("display_errors", self._default_settings["display_errors"])

    def set_display_errors(self, chat_id: int, value: bool) -> None:
        if not self._is_setting_allowed(chat_id, "display_errors"):
            return
        settings = self._get_chat_settings(chat_id)
        settings["display_errors"] = value
        self._save_all_settings()

    def get_all_chat_settings(self, chat_id: int) -> Dict[str, Any]:
        """Get all allowed settings for this chat, including defaults for missing ones"""
        current_settings = self._get_chat_settings(chat_id).copy()
        default_settings = self._get_default_settings_for_chat(chat_id)

        for key in default_settings:
            if key not in current_settings:
                current_settings[key] = default_settings[key]

        return current_settings

    def reset_chat_to_defaults(self, chat_id: int) -> None:
        """Reset all settings for a chat to default values"""
        self._all_settings[chat_id] = self._get_default_settings_for_chat(chat_id)
        self._save_all_settings()

    @classmethod
    def set_setting_permissions(cls, setting_name: str, private_allowed: bool, public_allowed: bool) -> None:
        """
        Update which chat types a setting is allowed for


        # Example: Make a setting only available for private chats
        BotSettings.set_setting_permissions("display_errors", private_allowed=True, public_allowed=False)

        # Example: Allow a setting for both chat types
        BotSettings.set_setting_permissions("chat_actions", private_allowed=True, public_allowed=True)
        """
        if private_allowed:
            cls._private_chat_allowed_settings.add(setting_name)
        else:
            cls._private_chat_allowed_settings.discard(setting_name)

        if public_allowed:
            cls._public_chat_allowed_settings.add(setting_name)
        else:
            cls._public_chat_allowed_settings.discard(setting_name)

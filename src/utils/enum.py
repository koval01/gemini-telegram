from typing import Any, TypeVar, overload
from enum import Enum

T = TypeVar('T', bound='AutoValueEnum')


class AutoValueEnum(Enum):
    """Enum that automatically returns .value when used as string"""

    def __str__(self) -> str:
        return str(self.value)

    @overload
    def __get__(self: T, obj: None, owner: Any) -> T: ...

    @overload
    def __get__(self, obj: Any, owner: Any) -> str: ...

    def __get__(self, obj: Any, owner: Any) -> Any:
        if obj is None:  # Accessing via class
            return self
        return str(self)  # Accessing via instance

    @classmethod
    def compose(cls, template: str, **kwargs: str) -> str:
        """Override this in child classes if needed"""
        raise NotImplementedError

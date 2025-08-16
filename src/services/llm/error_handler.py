import asyncio
import json
import logging
from functools import wraps
from typing import Any, Awaitable, Callable, Tuple, TYPE_CHECKING

from google.generativeai.types import BlockedPromptException, StopCandidateException

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # noinspection PyUnusedImports
    from src.services.llm.core import LLM


class LLMErrorHandler:
    """Decorator for handling LLM API errors with retry logic that preserves return types."""

    def __init__(self, retry_delay: int = 5) -> None:
        self.retry_delay = retry_delay

    def __call__(self, func: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
        @wraps(func)
        async def wrapper(instance: 'LLM', *args, **kwargs) -> str:
            bot_storage = kwargs.get('bot_storage')
            original_key_index = instance.current_key_index
            attempts = 0
            content = args[0] if args else None

            message_id = None
            chat_id = None

            if isinstance(content, tuple):
                for content in content:
                    try:
                        if content.strip().startswith(('{', '[')):
                            message = json.loads(content)
                            message_id = message.get("message_id") if isinstance(message, dict) else None
                            chat_id = message.get("chat").get("id") if isinstance(message, dict) else None
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass

            while True:
                try:
                    return await func(instance, *args, **kwargs)
                except Exception as e:
                    logger.error(e)
                    if (("429" in str(e) and "quota" in str(e).lower())
                            or ("403" in str(e) and "denied" in str(e).lower())
                    ):
                        logger.warning(
                            f"Quota exceeded for API key {instance.current_key_index}. Rotating..."
                        )
                        instance.rotate_api_key()

                        if instance.current_key_index == original_key_index:
                            attempts += 1
                            wait_time = self.retry_delay * attempts
                            logger.warning(f"All keys exhausted. Waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                    else:
                        error_msg, short_e = self._format_error(e, args[0] if args else None)
                        return instance.log_and_format_error(
                            error_msg,
                            short_e,
                            reply_to=message_id,
                            return_error=bot_storage.get_display_errors(chat_id) if bot_storage else True
                        )

        return wrapper

    @staticmethod
    def _format_error(e: Exception, input_data: Any = None) -> Tuple[str, str]:
        input_data = [repr(i)[:5_000] for i in input_data]
        if isinstance(e, StopCandidateException):
            return f"Content generation stopped: {e} | Input: {input_data}", str(e)
        elif isinstance(e, BlockedPromptException):
            return f"Blocked prompt error for input: {input_data} | Error: {e}", str(e)
        return f"Unexpected error during content generation: {e} | Input: {input_data}", str(e)

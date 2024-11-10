from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Message

from typing import Callable, Dict, Any, Awaitable


class ChatFilterMiddleware(BaseMiddleware):
    """
    Middleware to filter messages based on allowed chat IDs.

    This middleware checks if the incoming message's chat ID is in the
    list of allowed chat IDs. If the chat ID is not allowed, the message
    will be ignored and the handler will not be executed.

    Attributes:
        allowed_chat_ids (list[int] | tuple[int]): A list or tuple of chat IDs that are allowed to pass through the middleware.
    """

    def __init__(self, allowed_chat_ids: list[int] | tuple[int]):
        """
        Initializes the ChatFilterMiddleware with the specified allowed chat IDs.

        Args:
            allowed_chat_ids (list[int] | tuple[int]): A list or tuple of chat IDs that are permitted.
        """
        super().__init__()
        self.allowed_chat_ids = allowed_chat_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
    ) -> Any:
        """
        Processes an incoming event and filters based on allowed chat IDs.

        This method checks if the incoming message belongs to an allowed chat.
        If the message's chat ID is not in the list of allowed chat IDs,
        it will terminate the execution and return None. Otherwise,
        it will call the next handler.

        Args:
            handler (Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]]): The next handler to call.
            event (TelegramObject): The incoming Telegram event.
            data (Dict[str, Any]): A dictionary containing data relevant to the event.

        Returns:
            Any: The result of the handler if the chat ID is allowed; otherwise, None.
        """
        message: Message = data.get("event_update").message

        if message and message.chat.id not in self.allowed_chat_ids:
            return

        return await handler(event, data)

import logging

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, CallbackQuery
from typing import Callable, Dict, Any, Awaitable


class ChatFilterMiddleware(BaseMiddleware):
    """
    Middleware to filter messages and callback queries based on allowed chat IDs.

    This middleware checks if the incoming message's or callback query's chat ID is in the
    list of allowed chat IDs. If the chat ID is not allowed, the event will be ignored.

    Attributes:
        allowed_chat_ids (list[int] | tuple[int]): A list or tuple of chat IDs that are allowed.
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

        Args:
            handler: The next handler to call.
            event: The incoming Telegram event.
            data: A dictionary containing data relevant to the event.

        Returns:
            The result of the handler if the chat ID is allowed; otherwise, None.
        """
        message = None
        if "event_update" in data:
            update = data["event_update"]
            if hasattr(update, "message") and update.message:
                message = update.message
            elif hasattr(update, "callback_query") and update.callback_query:
                callback_query = update.callback_query
                if callback_query and callback_query.message:
                    message = callback_query.message

        if message and message.chat.id not in self.allowed_chat_ids:
            if isinstance(event, CallbackQuery):
                await event.answer("Я работаю только в @tut_dalbaebi", show_alert=True)
            else:
                await message.answer("Я работаю только в @tut_dalbaebi", parse_mode=None)
            logging.info(f"Blocked response from chat with id: {message.chat.id}")
            return None

        return await handler(event, data)

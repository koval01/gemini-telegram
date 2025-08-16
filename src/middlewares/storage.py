from aiogram import BaseMiddleware
from aiogram.types import TelegramObject
from typing import Callable, Dict, Any, Awaitable


class StorageMiddleware(BaseMiddleware):
    """Middleware that provides storage access to handlers for both messages and callbacks.

    This middleware injects a storage instance into the handler's data dictionary,
    making it accessible throughout the request handling chain for both regular
    messages and callback queries.
    """

    def __init__(self, bot_storage):
        """Initialize the StorageMiddleware with a storage instance.

        Args:
            bot_storage: The storage instance to be made available to handlers.
                     This could be a database connection, cache, or any other
                     storage implementation.
        """
        super().__init__()
        self.bot_storage = bot_storage

    async def __call__(
            self,
            handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
            event: TelegramObject,
            data: Dict[str, Any]
    ) -> Any:
        """Process the incoming event and inject storage into handler data.

        Args:
            handler: The next handler in the middleware chain.
            event: The incoming Telegram event (update or callback).
            data: Additional data associated with the event.

        Returns:
            The result of the handler execution.

        Notes:
            - Modifies the data dictionary by adding the storage instance under
              the 'storage' key before passing control to the next handler.
            - Works with both regular updates and callback queries.
        """
        data['bot_storage'] = self.bot_storage
        return await handler(event, data)

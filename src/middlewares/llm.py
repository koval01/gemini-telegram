from aiogram import BaseMiddleware
from aiogram.types import TelegramObject
from typing import Callable, Dict, Any, Awaitable


class LLMMiddleware(BaseMiddleware):
    """
    Middleware to inject an LLM (Language Model) instance into the event data.

    This middleware is used to pass an LLM instance to the handler function,
    making it accessible within the handler's context. This can be useful
    for processing messages or commands with the LLM.

    Attributes:
        llm: An instance of the LLM (Language Model) to be injected.
    """

    def __init__(self, llm):
        """
        Initializes the LLMMiddleware with the specified LLM instance.

        Args:
            llm: An instance of the LLM that will be made available to handlers.
        """
        super().__init__()
        self.llm = llm

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
    ) -> Any:
        """
        Processes an incoming event and injects the LLM instance into the data.

        This method adds the LLM instance to the event data dictionary under
        the key 'llm' before calling the next handler.

        Args:
            handler (Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]]):
                The next handler to call with the modified data.
            event (TelegramObject): The incoming Telegram event.
            data (Dict[str, Any]): A dictionary containing data relevant to the event.

        Returns:
            Any: The result of the handler after processing the event with the injected LLM.
        """
        data['llm'] = self.llm
        return await handler(event, data)

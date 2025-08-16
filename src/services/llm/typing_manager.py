import asyncio
import logging
import contextlib
from collections import defaultdict
from typing import AsyncGenerator, Dict, Set, Any, Optional

from aiogram import Bot
from aiogram.types import Message as TGMessage

logger = logging.getLogger(__name__)


class TypingManager:
    """Manages typing indicators for chat sessions."""

    def __init__(self):
        self._typing_tasks: Dict[int, asyncio.Task] = {}  # chat_id -> typing task
        self._active_requests: Dict[int, Set[asyncio.Future]] = defaultdict(set)  # chat_id -> set of futures

    @contextlib.asynccontextmanager
    async def typing_indicator(self, message: TGMessage, bot: Optional[Bot], enabled: bool = True) -> AsyncGenerator[None, Any]:
        """Improved typing indicator context manager that manages a single task per chat."""
        if not bot or not enabled:
            yield
            return

        chat_id = message.chat.id
        future = asyncio.get_running_loop().create_future()

        self._active_requests[chat_id].add(future)

        if chat_id not in self._typing_tasks or self._typing_tasks[chat_id].done():
            self._typing_tasks[chat_id] = asyncio.create_task(
                self._manage_typing_for_chat(chat_id, bot)
            )

        try:
            yield
        finally:
            future.set_result(None)
            self._active_requests[chat_id].discard(future)

            if not self._active_requests[chat_id]:
                del self._active_requests[chat_id]

                if chat_id in self._typing_tasks:
                    task = self._typing_tasks[chat_id]
                    del self._typing_tasks[chat_id]

                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    async def _manage_typing_for_chat(self, chat_id: int, bot: Bot) -> None:
        """Manage typing indicator for a single chat."""
        while chat_id in self._active_requests and self._active_requests[chat_id]:
            sleep_task = None
            try:
                await bot.send_chat_action(chat_id, "typing")

                sleep_task = asyncio.create_task(asyncio.sleep(3))

                done, pending = await asyncio.wait(
                    [sleep_task] + list(self._active_requests[chat_id]),
                    return_when=asyncio.FIRST_COMPLETED
                )

                if sleep_task not in done and sleep_task is not None:
                    sleep_task.cancel()
                    try:
                        await sleep_task
                    except asyncio.CancelledError:
                        pass

                for task in done:
                    if task != sleep_task:
                        await asyncio.sleep(0)

            except Exception as e:
                logger.error(f"Error in typing task for chat {chat_id}: {e}")
                if sleep_task and not sleep_task.done():
                    sleep_task.cancel()
                    try:
                        await sleep_task
                    except asyncio.CancelledError:
                        pass
                break

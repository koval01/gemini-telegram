import asyncio
import logging
import random

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import UpdateType, ParseMode

from config import settings
from src.services import LLM

from src.middlewares import LLMMiddleware, ChatFilterMiddleware, StorageMiddleware
from src.handlers import router
from src.utils import BotSettings

logger = logging.getLogger(__name__)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s:%(lineno)d #%(levelname)-8s "
        "[%(asctime)s] - %(name)s - %(message)s",
    )

    logger.info("Starting bot")

    bot: Bot = Bot(token=settings.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2))
    me = await bot.get_me()
    keys = settings.GEMINI_API_KEYS
    random.shuffle(keys)
    llm: LLM = LLM(me, api_keys=keys)
    settings_storage = BotSettings()
    dp: Dispatcher = Dispatcher()

    logger.info(f"Allowed chats list - {repr(settings.ALLOWED_CHATS)}")
    dp.message.middleware(ChatFilterMiddleware(settings.ALLOWED_CHATS))
    dp.message.middleware(LLMMiddleware(llm))
    dp.message.middleware(StorageMiddleware(settings_storage))
    dp.message_reaction.middleware(LLMMiddleware(llm))
    dp.callback_query.middleware(ChatFilterMiddleware(settings.ALLOWED_CHATS))
    dp.callback_query.middleware(StorageMiddleware(settings_storage))
    dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=[
        UpdateType.MESSAGE_REACTION, UpdateType.MESSAGE, UpdateType.CALLBACK_QUERY
    ])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from config import settings
from src.services import LLM

from src.middlewares import LLMMiddleware, ChatFilterMiddleware
from src.handlers import router


logger = logging.getLogger(__name__)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s:%(lineno)d #%(levelname)-8s "
        "[%(asctime)s] - %(name)s - %(message)s",
    )

    logger.info("Starting bot")

    bot: Bot = Bot(token=settings.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
    me = await bot.get_me()
    llm: LLM = LLM(me)
    dp: Dispatcher = Dispatcher()

    logger.info(f"Allowed chats list - {repr(settings.ALLOWED_CHATS)}")
    dp.message.middleware(ChatFilterMiddleware(settings.ALLOWED_CHATS))
    dp.message.middleware(LLMMiddleware(llm))
    dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")

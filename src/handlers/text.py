from aiogram import Router, F, Bot
from aiogram.enums import ChatType
from aiogram.types import Message

from src.services import LLM
from src.utils import BotSettings

router: Router = Router()


# noinspection PyProtectedMember
@router.message(
    F.content_type.in_({'text', 'dice', 'poll', 'contact'}),
    (F.reply_to_message.from_user.id == F.bot.id) |
    (F.func(lambda message: f"@{message.chat.bot._me.username}" in message.text.lower()))
)
async def selected_text_handler(message: Message, llm: LLM, bot: Bot, bot_storage: BotSettings) -> None:
    await llm.answer(message, bot=bot, bot_storage=bot_storage, selected=True)


@router.message(F.content_type.in_({'text', 'dice', 'poll', 'contact'}))
async def text_handler(message: Message, llm: LLM, bot: Bot, bot_storage: BotSettings) -> None:
    await llm.answer(message, bot=bot, bot_storage=bot_storage, selected=message.chat.type == ChatType.PRIVATE)

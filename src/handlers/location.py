from aiogram import Router, F, Bot
from aiogram.enums import ChatType
from aiogram.types import Message

from src.services import LLM, Location
from src.utils import BotSettings

router: Router = Router()


# noinspection PyProtectedMember
@router.message(
    (F.content_type == "location"),
    (F.reply_to_message.from_user.id == F.bot.id) |
    (F.func(lambda message: f"@{message.chat.bot._me.username}" in message.text.lower()))
)
async def selected_location_handler(message: Message, llm: LLM, bot: Bot, bot_storage: BotSettings) -> None:
    loc = message.location
    loc_info = await Location().reverse(loc.latitude, loc.longitude)
    await llm.answer(message, (loc_info,), bot=bot, bot_storage=bot_storage, selected=True)


@router.message((F.content_type == "location"))
async def location_handler(message: Message, llm: LLM, bot: Bot, bot_storage: BotSettings) -> None:
    loc = message.location
    loc_info = await Location().reverse(loc.latitude, loc.longitude)
    await llm.answer(
        message, (loc_info,),
        bot=bot, bot_storage=bot_storage, selected=message.chat.type == ChatType.PRIVATE)

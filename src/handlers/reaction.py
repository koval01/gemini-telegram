from aiogram import Router, Bot
from aiogram.types import MessageReactionUpdated, Message

from src.services import LLM
from src.decorators import chance
from src.utils import BotSettings

router: Router = Router()


@router.message_reaction()
@chance(0.5)
async def reaction_handler(message_reaction: MessageReactionUpdated, llm: LLM, bot: Bot, bot_storage: BotSettings) -> None:
    fake_message = Message(
        message_id=message_reaction.message_id,
        from_user=message_reaction.user,
        chat=message_reaction.chat,
        date=message_reaction.date,
    )

    await llm.answer(fake_message, bot=bot, bot_storage=bot_storage)

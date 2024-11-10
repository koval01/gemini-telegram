from aiogram import Router
from aiogram.types import Message, MessageReactionUpdated

from src.services import LLM

router: Router = Router()


@router.message_reaction()
async def reaction_handler(message_reaction: MessageReactionUpdated, llm: LLM) -> None:
    await llm.answer(message)

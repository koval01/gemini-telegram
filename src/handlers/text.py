from aiogram import Router, F
from aiogram.types import Message

from src.services import LLM

router: Router = Router()


@router.message((F.content_type.in_({'text', 'dice', 'poll'})))
async def text_handler(message: Message, llm: LLM) -> None:
    await llm.answer(message)

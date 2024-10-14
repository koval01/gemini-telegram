from aiogram import Router, F
from aiogram.types import Message

from src.services import LLM, Location

router: Router = Router()


@router.message((F.content_type == "location"))
async def location_handler(message: Message, llm: LLM) -> None:
    loc = message.location
    loc_info = await Location().reverse(loc.latitude, loc.longitude)
    await llm.answer(message, (loc_info,))

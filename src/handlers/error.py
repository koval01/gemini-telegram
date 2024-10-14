import asyncio

from aiogram import Router, F
from aiogram.types import Message, ErrorEvent

router: Router = Router()


@router.error(F.update.message.as_("message"))
async def error_handler(message: Message, exception: ErrorEvent) -> None:
    error_ = {
        "exception_json": exception.model_dump_json(),
        "title": "An error occurred while processing the message.",
        "caption": "This message will be destroyed in 10 seconds."
    }
    error_message = await message.reply(
        "{title}\n\n`{exception_json}`\n\n{caption}".format_map(error_)
    )
    await asyncio.sleep(10)
    await error_message.delete()

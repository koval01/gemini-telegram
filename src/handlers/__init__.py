from aiogram import Router

from src.handlers.text import router as text
from src.handlers.media import router as media
from src.handlers.location import router as location
from src.handlers.error import router as error

router: Router = Router()
router.include_routers(text, media, location, error)

from aiogram import Router

from src.handlers.settings import router as settings

from src.handlers.text import router as text
from src.handlers.media import router as media
from src.handlers.location import router as location
from src.handlers.reaction import router as reaction

router: Router = Router()
router.include_routers(settings, text, media, location, reaction)

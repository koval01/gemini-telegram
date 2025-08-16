from typing import Optional, List
from aiogram.types import (
    DateTime, Dice, Poll,
    MessageOriginUser, MessageOriginHiddenUser,
    MessageOriginChat, MessageOriginChannel, ReactionTypeEmoji
)

from pydantic import BaseModel
from src.models.location import Location


class Chat(BaseModel):
    id: int
    type: str
    title: Optional[str] = None
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class User(BaseModel):
    id: int
    is_bot: bool
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None
    language_code: Optional[str] = None
    is_premium: Optional[bool] = None


class Audio(BaseModel):
    file_unique_id: str
    duration: int
    performer: Optional[str] = None
    title: Optional[str] = None
    file_name: Optional[str] = None


class PhotoSize(BaseModel):
    file_unique_id: str


class Sticker(BaseModel):
    file_unique_id: str
    type: str
    emoji: Optional[str] = None
    set_name: Optional[str] = None


class Voice(BaseModel):
    file_unique_id: str
    duration: int


class Message(BaseModel):
    message_id: int
    date: DateTime
    chat: Chat
    message_thread_id: Optional[int] = None
    from_user: Optional[User] = None
    reply_to_message: Optional["Message"] = None
    forward_origin: MessageOriginUser | MessageOriginHiddenUser | MessageOriginChat | MessageOriginChannel | None
    forward_sender_name: Optional[str] = None
    text: Optional[str] = None
    audio: Optional[Audio] = None
    photo: Optional[List[PhotoSize]] = None
    sticker: Optional[Sticker] = None
    voice: Optional[Voice] = None
    caption: Optional[str] = None
    dice: Optional[Dice] = None
    poll: Optional[Poll] = None
    location: Optional[Location] = None


class AnswerLLM(BaseModel):
    reply_to: Optional[int] = None
    text: Optional[str] = None
    reaction: Optional[str] = None
    sticker: Optional[str] = None
    temporary: Optional[bool] = False
    python_code: Optional[str] = None


class ResponseLLM(BaseModel):
    answers: Optional[list[AnswerLLM]] = None
    skip: Optional[bool] = False
    news: Optional[bool] = False
    search: Optional[str] = None

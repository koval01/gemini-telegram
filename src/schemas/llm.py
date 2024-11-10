from typing import TypedDict, Optional


class Answer(TypedDict):
    reply_to: Optional[str]
    text: Optional[str]
    reaction: Optional[str]

class ResponseSchema(TypedDict):
    answers: list[Answer]
    skip: Optional[bool]

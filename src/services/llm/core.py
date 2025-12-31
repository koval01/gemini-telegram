import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Callable, Iterable, Type, Tuple

import google.generativeai as genai
from google.generativeai import GenerationConfig, ChatSession
from google.generativeai.types import (
    HarmBlockThreshold,
    HarmCategory,
    ContentType,
    content_types, generation_types,
)

from aiogram import Bot
from aiogram.types import Message as TGMessage, User

from src.models import MessageLLM, ResponseLLM
from src.models.llm import AnswerLLM
from src.services.tgnews import TGNews
from src.services.protector import Protector
from src.services.restricted_exec import RestrictedExecutor
from src.utils import File, BotSettings, Random

from .error_handler import LLMErrorHandler
from .response_processor import LLMResponseProcessor
from .history_manager import LLMHistoryManager
from .message_utils import MessageUtils
from .typing_manager import TypingManager

logger = logging.getLogger(__name__)
T = TypeVar('T')


class LLM:
    """Main LLM interaction class with per-chat sessions and history."""

    generation_config = GenerationConfig(
        candidate_count=1,
        temperature=0.9,
        top_k=64,
        top_p=0.95,
        presence_penalty=0,
        frequency_penalty=0,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "skip": {"type": "boolean"},
                "news": {"type": "boolean"},
                "search": {"type": "string"},
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "reply_to": {"type": "string"},
                            "text": {"type": "string"},
                            "reaction": {"type": "string"},
                            "sticker": {"type": "string"},
                            "python_code": {"type": "string"}
                        }
                    }
                }
            }
        }
    )

    safety_settings: dict = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    def __init__(self, me: User, model_name: str = "gemini-2.5-flash", api_keys: Optional[List[str]] = None) -> None:
        self.me = me.model_dump()
        self.model_name = model_name
        self.api_keys = api_keys
        self.current_key_index = 0

        self.history_root = Path(__file__).parent.parent.parent.parent / 'memory'
        self.history_root.mkdir(parents=True, exist_ok=True)

        self.response_processor = LLMResponseProcessor()
        self.protector = Protector(
            min_flood_length=0,
            repeated_message_threshold=6
        )
        self.restricted_executor = RestrictedExecutor()
        self.message_utils = MessageUtils(self.restricted_executor)
        self.typing_manager = TypingManager()

        self._sessions: Dict[int, Any] = {}  # chat_id -> genai chat session
        self._history_managers: Dict[int, LLMHistoryManager] = {}  # chat_id -> manager

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the generative model with current API key."""
        if not self.api_keys:
            raise ValueError("No valid API keys available")

        genai.configure(api_key=self.api_keys[self.current_key_index])

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            system_instruction=self._sys_prompt,
            tools=[]
        )

    def _rebind_sessions_on_model_change(self) -> None:
        """Recreate chat sessions for all chats using the new model and existing history."""
        for chat_id, old_session in list(self._sessions.items()):
            history = list(old_session.history)
            new_session = self.model.start_chat(history=history)
            self._limit_media_in_history(new_session)
            self._sessions[chat_id] = new_session

    def rotate_api_key(self) -> None:
        """Rotate to the next available API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._initialize_model()
        self._rebind_sessions_on_model_change()
        logger.info(f"Rotated to API key index {self.current_key_index}")

    def _get_history_manager(self, chat_id: int) -> LLMHistoryManager:
        if chat_id not in self._history_managers:
            path = self.history_root / f"{chat_id}.pkl"
            self._history_managers[chat_id] = LLMHistoryManager(path, enabled=True)
        return self._history_managers[chat_id]

    def _get_session(self, chat_id: int):
        if chat_id not in self._sessions:
            history = self._get_history_manager(chat_id).load()
            session = self.model.start_chat(history=history)
            self._limit_media_in_history(session)
            self._sessions[chat_id] = session
        return self._sessions[chat_id]

    @property
    def _sys_prompt(self) -> str:
        """Generate system prompt with bot information."""
        raw_prompt = File.read("../../../sys_prompt.md")
        return re.sub(
            r'{{\s*(\w+)\s*}}',
            lambda m: str(self.me.get(m.group(1), '')),
            raw_prompt
        )

    @LLMErrorHandler()
    async def _send_to_llm(self, content: ContentType, chat_id: int, bot_storage: Optional[BotSettings] = None) -> str:
        """Send content to LLM with error handling, per chat."""
        session = self._get_session(chat_id)
        logger.info(f"Session[{chat_id}] context len: {len(session.history)}")
        if bot_storage:
            logger.debug(f"Current intervention probability is {bot_storage.get_intervention_probability(chat_id)}%")

        response = await session.send_message_async(content)
        self._get_history_manager(chat_id).save(session.history)
        return response.text

    async def _process_news(self, content: content_types.ContentType) \
            -> generation_types.AsyncGenerateContentResponse | None:
        """Process news request without saving to history."""
        news_data = [n.model_dump_json() for n in await TGNews().get_all_posts()]
        # Create a temporary session that won't be saved
        temp_session = self.model.start_chat(history=[content])
        response = await temp_session.send_message_async(json.dumps({"news": news_data}))
        return response

    async def _process_search(self, message: TGMessage, query: str, content: content_types.ContentType) \
            -> generation_types.AsyncGenerateContentResponse | None:
        """Process search request without saving to history."""
        search_results = await self.message_utils.search_text(query)
        payload = {
            "search_result": search_results,
            "for_message_id": message.message_id
        }
        try:
            json_payload = json.dumps(payload, ensure_ascii=False)
            # Create a temporary session that won't be saved
            temp_session = self.model.start_chat(history=[content])
            response = await temp_session.send_message_async(json_payload)
            return response
        except (TypeError, ValueError) as e:
            logger.error(f"Error formatting search payload: {e}")
            return None

    async def _process_special_responses(
            self,
            message: TGMessage,
            bot: Bot,
            response: ResponseLLM,
            content: content_types.ContentType,
            bot_storage: Optional[BotSettings] = None
    ) -> Optional[ResponseLLM]:
        """Handle news and search responses without saving to history."""
        async with self.typing_manager.typing_indicator(
                message, bot, enabled=bot_storage.get_chat_actions(message.chat.id)
        ):
            content = content_types.to_content(content)
            content.role = 'user'
            l_response = None

            if response.news:
                news_resp = await self._process_news(content)
                if not news_resp:
                    return None
                l_response = news_resp.candidates[0].content
                response = self.response_processor.prepare_response(news_resp.text)

            if response and response.search:
                search_resp = await self._process_search(message, response.search, content)
                if not search_resp:
                    return None
                l_response = search_resp.candidates[0].content
                response = self.response_processor.prepare_response(search_resp.text)

            if l_response:
                session = self._get_session(message.chat.id)
                session.history.append(l_response)

            return response if response and not response.skip else None

    async def _send_telegram_response(
            self,
            message: TGMessage,
            response: ResponseLLM,
            bot: Optional[Bot] = None,
            reply: bool = False,
            bot_storage: Optional[BotSettings] = None
    ) -> None:
        """Send a formatted response to Telegram."""
        for answer in response.answers:
            if answer.text:
                await self.message_utils.send_text_response(message, answer, bot, reply, bot_storage=bot_storage)
            if answer.reaction:
                await self.message_utils.send_reaction(message, answer.reaction, bot)
            if answer.python_code:
                code = answer.python_code
                execution_result = await self.restricted_executor.execute_code_async(code)
                formatted_result = self.restricted_executor.format_code_response(code, execution_result)
                await self._llm_process(json.dumps({"python_result": formatted_result}), message.chat.id, bot_storage=bot_storage)
                code_answer = AnswerLLM(
                    text=formatted_result,
                    reply_to=answer.reply_to
                )
                await self.message_utils.send_text_response(message, code_answer, bot, reply, bot_storage=bot_storage)

    @staticmethod
    def log_and_format_error(e: str, short_e: Optional[str] = None, reply_to: Optional[int] = None, return_error: bool = True) -> str | None:
        """Format error for logging and LLM response."""
        logger.error(e)

        if return_error:
            return json.dumps({
                "answers": [{
                    "text": f"`{'error: %s' % short_e if short_e else e}`",
                    "reply_to": reply_to,
                    "temporary": True
                }],
            })

        return None

    async def _send_error(self, error: str, message: TGMessage, bot: Bot = None, delay: int | float = 5) -> None:
        await self.message_utils.send_error(error, message, bot, delay)

    @staticmethod
    def _limit_media_in_history(session: ChatSession) -> None:
        """
        Ensure a session history contains no more than 10 media elements by removing the oldest ones.
        """
        media_elements = []
        _mime_types = ('image', 'video',)

        def _get_mime_major(_mime: str) -> str:
            parts = _mime.split("/", 1)
            return parts[0].strip().lower() if len(parts) == 2 and parts[0] else ""

        for i, content in enumerate(session.history):
            if not content:
                return
            for part in content.parts:
                if hasattr(part, 'inline_data'):
                    mime = _get_mime_major(part.inline_data.mime_type)
                    if mime in _mime_types:
                        media_elements.append((i, part))

        if len(media_elements) <= 10:
            return

        media_elements.sort(key=lambda x: x[0])

        for i, part in media_elements[:len(media_elements) - 10]:
            content = session.history[i]

            new_parts = [p for p in content.parts if p != part]

            if new_parts:
                content.parts = new_parts
            else:
                del session.history[i]

    def _save_content(self, chat_id: int, content: content_types.ContentType, add_skip: bool = False) -> None:
        session = self._get_session(chat_id)

        content = content_types.to_content(content)
        content.role = 'user'

        history = session.history
        history.append(content)

        if add_skip:
            model_content = content_types.to_content(json.dumps({"skip": True}))
            model_content.role = 'model'
            history.append(model_content)

        self._limit_media_in_history(session)
        self._get_history_manager(chat_id).save(history)

        return None

    async def _llm_process(
            self, content: ContentType, chat_id: int, bot_storage: Optional[BotSettings] = None
    ) -> ResponseLLM | None:
        response_text = await self._send_to_llm(content, chat_id=chat_id, bot_storage=bot_storage)
        if not response_text:
            return None

        response = self.response_processor.prepare_response(response_text)
        if not response:
            return None

        return response

    async def answer(
            self,
            message: TGMessage,
            additional_input: Optional[Iterable[Any]] = None,
            processors: Optional[Dict[Type, Callable[[TGMessage, Any], TGMessage]]] = None,
            bot: Optional[Bot] = None,
            reply: bool = False,
            bot_storage: Optional[BotSettings] = None,
            selected: bool = False
    ) -> Optional[ResponseLLM]:
        """Main method to generate and send response.

        Args:
            message: The incoming Telegram message
            additional_input: Additional context data
            processors: Processors for additional input
            bot: Optional bot instance for sending messages
            reply: Whether to reply to the original message
            bot_storage: Settings storage
            selected: Is user selected message
        Returns:
            Optional[ResponseLLM]: The processed response or None if skipped
        """
        chat_id = message.chat.id
        user_id = message.from_user.id
        username = message.from_user.username
        if message.text and len(message.text) > 0:
            is_flood, error = self.protector.is_flood_attempt(message.text, message.from_user.id)
            self.protector.cleanup()
            if is_flood:
                logger.info(error)
                if bot_storage and bot_storage.get_display_errors(chat_id):
                    await self._send_error(error, message, bot, 5)
                return None

        chance = bot_storage.get_intervention_probability(chat_id) / 100  # will be ignored for private chat
        _rand = Random.urandom_float()
        intervention_probability = (_rand < chance) or selected

        logger.info(f"Triggered llm answer (selected: {selected}, chance: {chance}, rand: {_rand:.5f}, user_id: {user_id}, username: {username})")

        session = self._get_session(chat_id)
        self._limit_media_in_history(session)

        async with self.typing_manager.typing_indicator(
                message, bot, enabled=bot_storage.get_chat_actions(chat_id) and intervention_probability
        ):
            processed_message = message
            processed_additional_input = ()

            if additional_input is not None:
                processed_message, processed_additional_input = self._process_additional_input(
                    message, additional_input, processors
                )

            input_data = self._prepare_data(processed_message)
            complete_input = (*processed_additional_input, input_data)

            if not intervention_probability:
                self._save_content(message.chat.id, complete_input)
                return None

            response = await self._llm_process(complete_input, chat_id=message.chat.id, bot_storage=bot_storage)
            if not response:
                return None

            response = await self._process_special_responses(message, bot, response, input_data, bot_storage=bot_storage)
            if not response:
                return None

            await self._send_telegram_response(message, response, bot, reply, bot_storage=bot_storage)
            return response

    @staticmethod
    def _prepare_data(message: TGMessage) -> str:
        """Prepare message data for LLM input."""
        return MessageLLM(**message.model_dump()).model_dump_json(
            exclude_unset=True,
            exclude_none=True
        )

    @staticmethod
    def _process_additional_input(
            message: TGMessage,
            additional_input: Iterable[Any],
            processors: Optional[Dict[Type, Callable[[TGMessage, Any], TGMessage]]]
    ) -> Tuple[TGMessage, Tuple[Any, ...]]:
        """Process additional input with the given processors.

        Args:
            message: Original Telegram message
            additional_input: Additional input items to process
            processors: Mapping of types to processor functions

        Returns:
            Tuple of (processed_message, processed_additional_input)
        """
        if processors is None:
            processors = {}

        processed_message = message
        processed_input = []

        for item in additional_input:
            processed = False
            for item_type, processor in processors.items():
                if isinstance(item, item_type):
                    processed_message = processor(processed_message, item)
                    processed = True
                    break

            if not processed:
                processed_input.append(item)

        return processed_message, tuple(processed_input)

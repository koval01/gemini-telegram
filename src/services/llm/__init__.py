import logging as log
from typing import Any, Iterable, Callable, Dict
from functools import reduce

import re
import json

import google.generativeai as genai
from google.generativeai.types import (
    HarmCategory, HarmBlockThreshold, StopCandidateException, BlockedPromptException
)

from aiogram.types import Message as TGMessage, User
from pydantic import ValidationError

from config import settings
from src.utils import File
from src.models import MessageLLM, ResponseLLM, LocationReverse
from src.schemas import ResponseSchema

logger = log.getLogger(__name__)


class LLM:
    """
    A class to handle interactions with the Google Generative AI model.

    This class provides methods for preparing messages, handling responses,
    and communicating with the Google Gemini LLM API to generate text responses.
    """

    generation_config: dict = {
        "candidate_count": 1,
        "temperature": .9,
        "top_p": .9,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
        "response_schema": ResponseSchema
    }

    safety_settings: dict = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    def __init__(self, me = User, model_name: str = "gemini-1.5-flash-latest") -> None:
        """
        Initializes the LLM class with the specified model name and settings.

        Args:
            me (User): Information about bot
            model_name (str): The name of the model to use for generation. Defaults to "gemini-1.5-flash-latest".
        """
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.me: dict[str, Any] = me.model_dump()
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            system_instruction=self._sys_prompt
        )
        self.session = self.model.start_chat()

    @property
    def _sys_prompt(self) -> str:
        """
        Reads a system prompt template from a file, finds placeholders in the format {{ key }},
        and replaces them with the corresponding values from the `self.me` dictionary.

        The placeholders can optionally have spaces around the key, and if a key is not found
        in `self.me`, it is replaced with an empty string.

        Example:
            If the template contains "{{ username }}", it will be replaced by the value of `self.me['username']`.

        Returns:
            str: The system prompt with all placeholders replaced by their corresponding values.
        """
        raw_prompt = File.read("sys_prompt.md")

        def replace_placeholder(match):
            key = match.group(1)
            return str(self.me.get(key, ''))

        # Use regex to find all placeholders like {{ username }} and replace them
        processed_prompt = (re.sub(r'{{\s*(\w+)\s*}}', replace_placeholder, raw_prompt)
                            .replace("\n", " "))

        return processed_prompt

    @staticmethod
    def _prepare_data(message: TGMessage) -> str:
        """
        Prepares the input data for the LLM from the given Telegram message.

        Args:
            message (TGMessage): The Telegram message to prepare.

        Returns:
            str: The JSON string representation of the message data.
        """
        return (
            MessageLLM(**message.model_dump())
            .model_dump_json(
                exclude_unset=True,
                exclude_none=True
            )
        )

    @staticmethod
    def _prepare_response(response_llm: str) -> ResponseLLM | None:
        """
        Cleans and parses the raw response from the LLM.

        Args:
            response_llm (str): The raw response string from the LLM.

        Returns:
            ResponseLLM | None: The validated ResponseLLM object, or None if parsing fails.
        """
        def _clean_response(raw_response: str) -> str:
            """Remove unwanted markers like ```json and ``` from the raw response."""
            return raw_response.strip().replace("```json", "").replace("```", "")

        def _parse_response(cleaned_response: str) -> dict | None:
            """Try to parse the cleaned response into a JSON object."""
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as exc:
                error_message = f"JSON parsing error: {exc.msg}. Line {exc.lineno}, Column {exc.colno}"
                logger.error(error_message)
                return None

        # Clean the response and parse it into JSON
        clean_response = _clean_response(response_llm)
        response_dict = _parse_response(clean_response)

        if response_dict is None:
            return None

        try:
            return ResponseLLM(**response_dict)
        except ValidationError as e:
            logger.info(repr(response_dict))
            logger.error(str(e))
            return None

    @staticmethod
    def _additional_input_process(
            message: TGMessage,
            additional_input: Iterable[Any],
            processors: Dict[type, Callable[[TGMessage, Any], TGMessage]] = None
    ) -> tuple[TGMessage, tuple]:
        """
        Process additional input for a TGMessage, applying custom processors to specific types of input.

        This method allows for flexible processing of various types of additional input. It can modify
        the original message based on the type of additional input items and remove processed items
        from the additional input.

        Args:
            message (TGMessage): The original message to be processed.
            additional_input (Iterable[Any]): An iterable of additional input items to process.
            processors (Dict[type, Callable[[TGMessage, Any], TGMessage]], optional): A dictionary
                mapping types to processor functions. Each processor function should take a TGMessage
                and an item of the specified type, and return a modified TGMessage. Defaults to None.

        Returns:
            tuple[TGMessage, tuple]: A tuple containing the processed message and a tuple of
            remaining unprocessed additional input items.
        """
        if processors is None:
            processors = {
                LocationReverse: lambda msg, item: MessageLLM(**{**msg.model_dump(), "location": item.model_dump()})
            }

        def process_input(acc: tuple[TGMessage, list], item: Any) -> tuple[TGMessage, list]:
            msg, processed_input_ = acc
            for item_type, processor in processors.items():
                if isinstance(item, item_type):
                    msg = processor(msg, item)
                    break
            else:
                processed_input_.append(item)
            return msg, processed_input_

        processed_message, processed_input = reduce(process_input, additional_input, (message, []))
        return processed_message, tuple(processed_input)

    async def _response(self, message: TGMessage, additional_input: Iterable[Any] = None,
                        processors: Dict[type, Callable[[TGMessage, Any], TGMessage]] = None) -> str | None:
        """
        Sends a message to the LLM and returns the generated response.

        Args:
            message (TGMessage): The message to send to the LLM.
            additional_input (Iterable[Any], optional): Additional input to provide to the LLM. Defaults to None.
            processors (Dict[type, Callable[[TGMessage, Any], TGMessage]], optional): Custom processors for
                additional input. Defaults to None.

        Returns:
            str | None: The generated response from the LLM, or None if an error occurred.
        """
        if additional_input is None:
            additional_input = ()
        else:
            message, additional_input = self._additional_input_process(message, additional_input, processors)

        input_data = self._prepare_data(message)

        model_response = None
        try:
            response = await self.session.send_message_async((*additional_input, input_data,))
            model_response = response.text
        except StopCandidateException as e:
            logger.error(f"Content generation stopped: {str(e)} | Input: {input_data}")
        except BlockedPromptException as e:
            logger.error(f"Blocked prompt error for input: {input_data} | Error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during content generation. Exception: {str(e)} | Input: {input_data}")
        finally:
            return model_response

    async def answer(self, message: TGMessage, additional_input: Iterable[Any] = None,
                     processors: Dict[type, Callable[[TGMessage, Any], TGMessage]] = None) -> ResponseLLM | None:
        """
        Generates a response from the LLM for the given message and sends it back.

        Args:
            message (TGMessage): The message to generate a response for.
            additional_input (Iterable[Any], optional): Additional input to provide to the LLM. Defaults to None.
            processors (Dict[type, Callable[[TGMessage, Any], TGMessage]], optional): Custom processors for
                additional input. Defaults to None.

        Returns:
            ResponseLLM | None: The validated ResponseLLM object sent to the user, or None if no valid response was generated.
        """
        resp_text = await self._response(message, additional_input, processors)
        if not resp_text:
            return None

        finish_resp = self._prepare_response(resp_text)
        if not finish_resp or finish_resp.skip:
            return None
        if not finish_resp.answers:
            logger.warning("Key \"answers\" is empty")
            return None

        for r in finish_resp.answers:
            await message.answer(r.text, reply_to_message_id=r.reply_to)
            if r.reaction:
                await message.react([r.reaction])

        return finish_resp

import json
import logging
import re
from typing import Optional

from pydantic import ValidationError

from src.models import ResponseLLM

logger = logging.getLogger(__name__)


class LLMResponseProcessor:
    """Handles processing and validation of LLM responses."""

    @staticmethod
    def clean_response(raw_response: str) -> str:
        return raw_response.strip().replace("\n ", "\n").replace("\n\n", "\n")

    @staticmethod
    def parse_response(cleaned_response: str) -> Optional[dict]:
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e.msg}. Line {e.lineno}, Column {e.colno}")
            return None

    @staticmethod
    def clean_answers(response: dict) -> dict:
        if not response.get('answers'):
            return {**response, "skip": True}

        new_answers = []
        for answer in response['answers']:
            if not isinstance(answer, dict):
                continue

            _r = answer.get('reply_to')
            if isinstance(_r, str) and (m := re.search(r'\d+', _r)):
                answer = answer.copy()
                answer['reply_to'] = m.group()
            else:
                answer = answer.copy()
                answer.pop('reply_to', None)

            if answer:
                new_answers.append(answer)

        return {**response, 'answers': new_answers} if new_answers else {**response, "skip": True}

    @staticmethod
    def normalize_news_field(response_dict: dict) -> dict:
        if 'news' in response_dict and not isinstance(response_dict['news'], bool):
            news_value = response_dict.pop('news')
            response_dict['news'] = bool(isinstance(news_value, list) and news_value)
        return response_dict

    def prepare_response(self, response_llm: str) -> Optional[ResponseLLM]:
        cleaned = self.clean_response(response_llm)
        parsed = self.parse_response(cleaned)
        if not parsed:
            return None

        parsed = self.clean_answers(parsed)
        parsed = self.normalize_news_field(parsed)

        try:
            return ResponseLLM(**parsed)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return None

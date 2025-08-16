import asyncio
import logging
import re
from functools import partial
from typing import Any, Dict, List, Optional

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message as TGMessage, ReactionTypeEmoji

from ddgs import DDGS

from src.models import MessageLLM
from src.models.llm import AnswerLLM
from src.services.restricted_exec import RestrictedExecutor

logger = logging.getLogger(__name__)


class MessageUtils:
    """Utility class for message-related operations."""

    def __init__(self, restricted_executor: RestrictedExecutor):
        self.restricted_executor = restricted_executor

    @staticmethod
    async def search_text(query: str) -> List[Dict[str, Any]]:
        """Perform web search."""
        func = partial(DDGS().text,
                       query,
                       region="ru-ru",
                       max_results=150,
                       safesearch="off",
                       backend="bing, brave, google, yahoo, yandex, wikipedia")
        result = await asyncio.to_thread(func)
        return list(result)

    @staticmethod
    def _prepare_data(message: TGMessage) -> str:
        """Prepare message data for LLM input."""
        return MessageLLM(**message.model_dump()).model_dump_json(
            exclude_unset=True,
            exclude_none=True
        )

    @staticmethod
    async def delete_after_delay(message: TGMessage, delay: int | float = 5) -> None:
        """Delete a message after a delay."""
        try:
            await asyncio.sleep(delay)
            await message.delete()
        except Exception as e:
            logger.error(f"Failed to delete temporary message: {e}")

    @staticmethod
    async def send_reaction(
            message: TGMessage,
            emoji: str,
            bot: Optional[Bot]
    ) -> None:
        """Send reaction to message."""
        reaction = ReactionTypeEmoji(emoji=emoji)
        try:
            if bot:
                await bot.set_message_reaction(
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    reaction=[reaction]
                )
            else:
                await message.react([reaction])
        except TelegramBadRequest as e:
            logger.error(f"Failed to set reaction: {e}")

    async def send_error(self, error: str, message: TGMessage, bot: Optional[Bot] = None,
                         delay: int | float = 5) -> None:
        """Send an error message that will be automatically deleted after delay."""
        send_method = bot.send_message if bot else message.answer
        try:
            error_msg = await send_method(
                chat_id=message.chat.id,
                text=error,
                reply_to_message_id=message.message_id,
                parse_mode=None,
                disable_notification=True,
                disable_web_page_preview=True
            )
            asyncio.create_task(self.delete_after_delay(error_msg, delay))
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")

    async def send_text_response(
            self,
            message: TGMessage,
            answer: AnswerLLM,
            bot: Optional[Bot],
            reply: bool,
            bot_storage: Optional[Any] = None
    ) -> None:
        """Send text response to Telegram with proper chunking and markdown preservation."""
        text = answer.text.replace("  ", " ")
        send_method = bot.send_message if bot else message.answer
        reply_to = message.message_id if reply else answer.reply_to

        base_kwargs = {
            'chat_id': message.chat.id,
            'reply_to_message_id': reply_to,
            'disable_notification': True,
            'disable_web_page_preview': True
        }

        chunks = self._split_message(text)

        sent_messages = []
        parse_mode = ParseMode.MARKDOWN_V2
        first_chunk = True

        for chunk in chunks:
            try:
                if first_chunk:
                    try:
                        msg = await send_method(
                            **base_kwargs,
                            text=chunk,
                            parse_mode=parse_mode
                        )
                        sent_messages.append(msg)
                        first_chunk = False
                        continue
                    except TelegramBadRequest as e:
                        if "can't parse entities" in str(e):
                            parse_mode = None
                        else:
                            raise

                msg = await send_method(
                    **base_kwargs,
                    text=chunk,
                    parse_mode=parse_mode
                )
                sent_messages.append(msg)

            except Exception as e:
                logger.error(f"Error sending message chunk: {e}")
                if first_chunk and bot_storage and bot_storage.get_display_errors(message.chat.id):
                    await self.send_error(f"Failed to send message: {e}", message, bot)
                break

        if getattr(answer, 'temporary', False):
            for msg in sent_messages:
                asyncio.create_task(self.delete_after_delay(msg, 5))

    def _split_message(self, text: str, max_length: int = 4000) -> List[str]:
        """Split message into chunks respecting formatting and Telegram limits."""
        if len(text) <= max_length:
            return [text]

        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            chunks = []
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 > max_length:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    if len(para) > max_length:
                        chunks.extend(self._split_long_paragraph(para, max_length))
                        continue

                if current_chunk:
                    current_chunk += '\n\n'
                current_chunk += para

            if current_chunk:
                chunks.append(current_chunk)

            if len(chunks) > 1:
                return chunks

        code_blocks = re.split(r'(```[a-z]*\n.*?\n```)', text, flags=re.DOTALL)
        if len(code_blocks) > 1:
            chunks = []
            current_chunk = ""

            for block in code_blocks:
                if len(current_chunk) + len(block) > max_length:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    if len(block) > max_length and block.startswith('```'):
                        chunks.extend(self._split_code_block(block, max_length))
                        continue

                current_chunk += block

            if current_chunk:
                chunks.append(current_chunk)

            if len(chunks) > 1:
                return chunks

        return self._split_by_lines(text, max_length)

    @staticmethod
    def _split_long_paragraph(text: str, max_length: int) -> List[str]:
        """Split a long paragraph into chunks."""
        words = text.split(' ')
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                if len(word) > max_length:
                    chunks.extend([word[i:i + max_length] for i in range(0, len(word), max_length)])
                    continue

            if current_chunk:
                current_chunk += ' '
            current_chunk += word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_code_block(self, code_block: str, max_length: int = 4000) -> List[str]:
        """Split a long code block into multiple valid code blocks while preserving syntax."""
        if not code_block.startswith('```'):
            return [code_block]

        header_end = code_block.find('\n')
        if header_end == -1:
            return [code_block]

        language = code_block[3:header_end].strip()
        code_content = code_block[header_end + 1:-3].strip()

        chunks = []

        sections = re.split(r'(?=\n(def |class |async def ))', code_content)
        if len(sections) > 1:
            for i in range(1, len(sections), 2):
                section = sections[i] + sections[i + 1]
                chunk = f"```{language}\n{section}\n```"
                if len(chunk) <= max_length:
                    chunks.append(chunk)
                else:
                    chunks.extend(self._split_code_block(f"```{language}\n{section}\n```", max_length))
            return chunks

        return self._split_code_by_lines(language, code_content, max_length)

    @staticmethod
    def _split_code_by_lines(language: str, code_content: str, max_length: int) -> List[str]:
        """Helper method to split code content by lines."""
        chunks = []
        current_lines = []
        current_length = 0
        marker_length = len(f"```{language}\n\n```")

        for line in code_content.split('\n'):
            line_length = len(line) + 1
            if current_length + line_length + marker_length > max_length:
                if current_lines:
                    chunks.append(f"```{language}\n" + '\n'.join(current_lines) + "\n```")
                    current_lines = []
                    current_length = 0
            current_lines.append(line)
            current_length += line_length

        if current_lines:
            chunks.append(f"```{language}\n" + '\n'.join(current_lines) + "\n```")

        return chunks

    @staticmethod
    def _split_by_lines(text: str, max_length: int) -> List[str]:
        """Split text by lines when other methods fail."""
        lines = text.split('\n')
        chunks = []
        current_chunk = ""

        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                if len(line) > max_length:
                    chunks.extend([line[i:i + max_length] for i in range(0, len(line), max_length)])
                    continue

            if current_chunk:
                current_chunk += '\n'
            current_chunk += line

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

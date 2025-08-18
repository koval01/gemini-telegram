import asyncio
import logging
import re

from typing import List, Dict, Callable
from datetime import datetime

from src.services.tme.models import ChannelBody, More
from src.services.tme.telegram import Telegram
from src.models import TelegramPost
from src.utils import handle_tme_request

logger = logging.getLogger(__name__)


class TGNews:
    def __init__(
        self,
        channels: List[str] = None,
        more_depth: int = 2,
        host: str = "localhost:8000",
        https: bool = False,
        remove_patterns: List[str] = None,
        keep_patterns: List[str] = None,
        custom_filter: Callable[[str], str] = None
    ) -> None:
        if remove_patterns is None:
            remove_patterns = [
                r'@ecotopor',
                r'@meduzalive',
                r'👉 Топор Live\. Подписаться',
                r'👉 Топор \+18\. Подписаться',
                r'Сайт "Страна" \| X/Twitter \| Прислать новость/фото/видео \| Реклама на канале \| Помощь',
                r'Подписаться \| Связь с редакцией\/прислать новость'
            ]
        if channels is None:
            channels = ["ecotopor", "meduzalive", "PresidentDonaldTrumpRU", "stranaua", "Ateobreaking"]
        self.host = host
        self.https = https
        self.channels = channels
        self.more_depth = more_depth
        self.retry_count = 3
        self.retry_delay = 1

        self.telegram = Telegram()

        self.remove_patterns = [re.compile(pattern)
                                for pattern in (remove_patterns or [])]
        self.keep_patterns = [re.compile(pattern)
                              for pattern in (keep_patterns or [])]
        self.custom_filter = custom_filter

    @staticmethod
    def _remove_emoji(text: str) -> str:
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def _filter_text(self, text: str) -> str:
        """Apply all filters to the text"""
        if not text:
            return text

        for pattern in self.remove_patterns:
            text = pattern.sub('', text)

        if self.keep_patterns:
            kept_parts = []
            for pattern in self.keep_patterns:
                matches = pattern.findall(text)
                if matches:
                    kept_parts.extend(matches)
            if kept_parts:
                text = ' '.join(kept_parts)
            else:
                text = ''

        if self.custom_filter:
            text = self.custom_filter(text)

        text = self._remove_emoji(text)
        return text.strip()

    async def _process_posts(self, response: Dict) -> List[Dict[str, str]]:
        posts = []
        try:
            if not response or 'content' not in response or 'posts' not in response['content']:
                return posts

            for post in response['content']['posts']:
                try:
                    raw_text = post['content']['text']['string']
                    filtered_text = self._filter_text(raw_text)

                    if not filtered_text:
                        continue

                    post_data = {
                        'text': filtered_text,
                        'date': post['footer']['date']['string'],
                        '_timestamp': self._parse_date(post['footer']['date']['string'])
                    }
                    posts.append(post_data)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping malformed post: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error processing posts: {str(e)}")

        return posts

    @staticmethod
    def _parse_date(date_str: str) -> float:
        """Convert date string to timestamp for sorting"""
        try:
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str)
            else:
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except ValueError as e:
            logger.warning(f"Failed to parse date '{date_str}': {str(e)}")
            return 0

    async def _get_more_posts(self, channel: str, before_id: int) -> List[Dict[str, str]]:
        more_posts = []
        current_depth = 0

        while current_depth < self.more_depth:
            logger.info(f"Getting news for channel {channel} from more method")
            response = await handle_tme_request(
                self.telegram.more, More, channel, before_id, "before"
            )

            if not response or 'posts' not in response:
                break

            try:
                for post in response['posts']:
                    try:
                        raw_text = post['content']['text']['string']
                        filtered_text = self._filter_text(raw_text)

                        if not filtered_text:
                            continue

                        post_data = {
                            'text': filtered_text,
                            'date': post['footer']['date']['string'],
                            '_timestamp': self._parse_date(post['footer']['date']['string'])
                        }
                        more_posts.append(post_data)
                        # before_id = post['id']
                    except (KeyError, TypeError) as e:
                        logger.warning(
                            f"Skipping malformed post in more_posts: {str(e)}")
                        continue

                current_depth += 1

                if 'meta' not in response or 'offset' not in response['meta'] or 'before' not in response['meta']['offset']:
                    break

                before_id = response['meta']['offset']['before']

            except Exception as e:
                logger.error(f"Error processing more posts: {str(e)}")
                break

        return more_posts

    async def body(self, channel: str) -> List[Dict[str, str]]:
        try:
            logger.info(f"Getting news for channel {channel} from body")
            response = await handle_tme_request(
                self.telegram.body, ChannelBody, channel, None
            )
            if not response:
                return []

            posts = await self._process_posts(response)

            if (self.more_depth > 0 and 'meta' in response
                and 'offset' in response['meta']
                    and 'before' in response['meta']['offset']):
                more_posts = await self._get_more_posts(channel, response['meta']['offset']['before'])
                posts.extend(more_posts)

            return posts
        except Exception as e:
            logger.error(f"Error in body for channel {channel}: {str(e)}")
            return []

    async def get_all_posts(self) -> List[TelegramPost]:
        """Get all posts from all channels, sorted by date (newest first)"""
        all_posts = []

        tasks = [self.body(channel) for channel in self.channels]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for channel, result in zip(self.channels, results):
            if isinstance(result, Exception):
                logger.error(f"Channel {channel} processing failed: {str(result)}")
            elif isinstance(result, list):
                for post in result:
                    try:
                        post['channel'] = channel
                        telegram_post = TelegramPost(**post)
                        all_posts.append(telegram_post)
                    except Exception as e:
                        logger.error(f"Failed to create TelegramPost: {str(e)}")

        try:
            all_posts.sort(
                key=lambda x: (
                    datetime.fromisoformat(x.date).timestamp()
                    if 'T' in x.date
                    else datetime.strptime(x.date, "%Y-%m-%d %H:%M:%S").timestamp()
                ),
                reverse=True
            )
        except Exception as e:
            logger.error(f"Sorting failed: {str(e)}")

        return all_posts

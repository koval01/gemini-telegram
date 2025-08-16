import numpy as np
from collections import defaultdict, Counter, deque
from typing import Tuple, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
import zlib
import time
import unicodedata
import re
import emoji
import string
import math


class Protector:
    """
    Optimized flood protection with:
    - Z-algorithm based repeat detection
    - Lightweight machine learning
    - Markov chain analysis
    - Efficient memory usage
    """

    def __init__(self,
                 max_repeat_threshold: float = 0.45,
                 min_entropy_threshold: float = 2.0,
                 time_window: int = 60,
                 max_requests_per_window: int = 30,
                 min_flood_length: int = 15,
                 repeated_message_threshold: int = 3):
        """
        Initialize with optimized parameters
        """
        self.word_pattern = re.compile(r'\w+', re.UNICODE)
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.]|%[\da-fA-F]{2})+',
            re.UNICODE
        )

        # Common Russian prefixes/suffixes for better natural language detection
        self.russian_affixes = {
            'пре', 'при', 'раз', 'рас', 'под', 'над', 'пере', 'об', 'от',
            'ость', 'ение', 'ание', 'ение', 'тель', 'изм', 'ист', 'ник'
        }

        # Then initialize other attributes
        self.max_repeat_threshold = max_repeat_threshold
        self.min_entropy_threshold = min_entropy_threshold
        self.time_window = time_window
        self.max_requests_per_window = max_requests_per_window
        self.min_flood_length = min_flood_length
        self.repeated_message_threshold = repeated_message_threshold

        self.user_message_history = defaultdict(lambda: deque(maxlen=100))
        self.recent_messages = deque(maxlen=1000)

        # N-gram analyzer
        self.vectorizer = TfidfVectorizer(ngram_range=(2, 3), analyzer='char', max_features=100)

    def _generate_training_data(self, normal_samples: int, anomaly_samples: int):
        """Generate balanced training data"""
        x = []
        y = []

        # Normal messages
        for _ in range(normal_samples):
            text = self._generate_normal_text()
            features = self._extract_features(text)
            x.append(features)
            y.append(0)

        # Anomalies
        for _ in range(anomaly_samples):
            text = self._generate_anomaly_text()
            features = self._extract_features(text)
            x.append(features)
            y.append(1)

        return np.array(x), np.array(y)

    def _generate_normal_text(self) -> str:
        """Generate realistic normal text"""
        # 70% Russian, 30% English
        if np.random.random() < 0.7:
            base_words = ["привет", "как", "дела", "что", "нового",
                          "сегодня", "работа", "дом", "семья", "погода"]
            text = ' '.join(np.random.choice(base_words, np.random.randint(3, 10)))

            # Add some affixes
            if np.random.random() < 0.3:
                text += ' ' + np.random.choice(list(self.russian_affixes))
        else:
            base_words = ["hello", "how", "are", "you", "what's",
                          "new", "today", "work", "home", "family"]
            text = ' '.join(np.random.choice(base_words, np.random.randint(3, 10)))

        # Add punctuation
        if np.random.random() < 0.5:
            text += np.random.choice(['.', '!', '?'])

        # Add emoji with 20% chance
        if np.random.random() < 0.2:
            text += ' ' + np.random.choice([':)', ':(', ';)', '😊', '👍'])

        return text

    @staticmethod
    def _generate_anomaly_text() -> str:
        """Generate obvious flood patterns"""
        patterns = [
            lambda: 'a' * np.random.randint(20, 100),
            lambda: '1' * np.random.randint(20, 100),
            lambda: '!' * np.random.randint(20, 100),
            lambda: '😀' * np.random.randint(10, 30),
            lambda: 'spam ' * np.random.randint(5, 20),
            lambda: ''.join(np.random.choice(list('абвгд1234!? '), np.random.randint(30, 100)))
        ]
        return np.random.choice(patterns)()

    def _contains_url(self, text: str) -> bool:
        """Check for URLs using regex"""
        return bool(self.url_pattern.search(text))

    @staticmethod
    def _compression_ratio(text: str) -> float:
        """Calculate compression ratio more efficiently"""
        if len(text) < 20:  # Not meaningful for short texts
            return 1.0
        compressed = zlib.compress(text.encode('utf-8'))
        return len(compressed) / len(text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Optimized text normalization"""
        # Extract emojis first
        emoji_chars = [c for c in text if c in emoji.EMOJI_DATA]
        text_without_emoji = emoji.replace_emoji(text, replace='')

        # Basic normalization
        text = text_without_emoji.casefold()
        text = ''.join(c for c in unicodedata.normalize('NFKD', text)
                       if not unicodedata.combining(c))

        # Keep some meaningful punctuation
        punct_to_keep = {'?', '!', '.', ','}
        text = ''.join(c for c in text
                       if c not in string.punctuation or c in punct_to_keep)

        # Add emojis back as space-separated tokens
        if emoji_chars:
            text += ' ' + ' '.join(emoji_chars)

        return text.strip()

    def _extract_features(self, text: str) -> np.ndarray:
        """Optimized feature extraction with performance-conscious design"""
        normalized = self._normalize_text(text)
        text_len = len(normalized)

        # Pre-calculate reusable values
        word_matches = self.word_pattern.findall(normalized) if text_len >= 5 else []
        word_count = len(word_matches)

        features = [
            # 1. Z-algorithm repeat pattern detection (O(n) time)
            self._z_repeat_score(normalized) if text_len >= 10 else 0.0,

            # 2. Markov chain naturalness score (lightweight)
            self._markov_score(normalized) if text_len >= 15 else 1.0,

            # 3. Entropy with length-aware thresholds
            self._calculate_entropy(text),

            # 4. Normalized length (0-1 scale)
            min(text_len / 100, 1.0),  # Cap at 1.0 for very long messages

            # 5. Emoji ratio (fast counter)
            self._emoji_ratio(text),

            # 6. Punctuation ratio (optimized counter)
            self._punctuation_ratio(text),

            # 7. Unicode diversity (cached category check)
            self._unicode_diversity(text) if text_len >= 10 else 0.5,

            # 8. Compression ratio (skip for very short texts)
            self._compression_ratio(normalized) if text_len >= 20 else 1.0,

            # 9. URL detection (fast regex check)
            float(self._contains_url(text)),

            # 10. Word density (pre-calculated)
            word_count / (text_len + 1e-10) if text_len >= 5 else 1.0,

            # 11. Russian language likelihood (fast set operations)
            self._russian_likelihood(word_matches) if word_count >= 3 else 0.5
        ]

        return np.array(features, dtype=np.float32)  # Use float32 for memory efficiency

    def _russian_likelihood(self, words: List[str]) -> float:
        """Fast Russian language detection using common prefixes"""
        if not words:
            return 0.0
        russian_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        score = 0.0
        for word in words[:10]:  # Only check the first 10 words for performance
            if len(word) >= 3:
                # Check for Cyrillic characters and common prefixes
                has_cyrillic = any(c in russian_chars for c in word.lower())
                common_prefix = any(word.lower().startswith(p) for p in self.russian_affixes)
                score += 0.5 * float(has_cyrillic) + 0.5 * float(common_prefix)
        return min(score / len(words), 1.0)

    @staticmethod
    def _z_repeat_score(text: str) -> float:
        """Z-algorithm based repeat detection"""
        if len(text) < 10:
            return 0.0

        # Find the longest repeated substring
        n = len(text)
        z = [0] * n
        z[0] = n
        l, r = 0, 0

        max_z = 0
        for i in range(1, n):
            if i > r:
                l = r = i
                while r < n and text[r - l] == text[r]:
                    r += 1
                z[i] = r - l
                r -= 1
            else:
                k = i - l
                if z[k] < r - i + 1:
                    z[i] = z[k]
                else:
                    l = i
                    while r < n and text[r - l] == text[r]:
                        r += 1
                    z[i] = r - l
                    r -= 1
            if z[i] > max_z:
                max_z = z[i]

        return max_z / len(text)

    @staticmethod
    def _markov_score(text: str) -> float:
        """Simple Markov chain analysis for naturalness"""
        if len(text) < 10:
            return 1.0  # Assume natural for short texts

        # Build transition counts
        transitions = defaultdict(int)
        for i in range(len(text) - 1):
            transitions[(text[i], text[i + 1])] += 1

        # Calculate probability
        total = sum(transitions.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in transitions.values():
            p = count / total
            entropy += -p * math.log(p + 1e-10)

        # Normalize (higher entropy = more natural)
        max_possible = math.log(len(set(text)) + 1e-10)
        if max_possible < 1e-10:
            return 0.0

        return entropy / max_possible

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Improved entropy calculation with length adjustment"""
        if not text:
            return 0.0

        # Filter out whitespace and common punctuation
        filtered = [c for c in text.lower() if c.isalnum()]
        if not filtered:
            return 0.0

        length = len(filtered)
        if length < 5:
            return 2.5

        counts = Counter(filtered)
        total = len(filtered)
        entropy = 0.0

        for count in counts.values():
            p = count / total
            entropy += -p * math.log(p + 1e-10)

        return entropy

    @staticmethod
    def _emoji_ratio(text: str) -> float:
        """Calculate emoji ratio"""
        emoji_count = sum(1 for c in text if c in emoji.EMOJI_DATA)
        return emoji_count / len(text) if text else 0.0

    @staticmethod
    def _punctuation_ratio(text: str) -> float:
        """Calculate punctuation ratio"""
        punct_count = sum(1 for c in text if c in string.punctuation)
        return punct_count / len(text) if text else 0.0

    @staticmethod
    def _unicode_diversity(text: str) -> float:
        """Calculate Unicode category diversity"""
        if not text:
            return 0.0
        categories = {unicodedata.category(c) for c in text}
        return len(categories) / len(text)

    def is_flood_attempt(self, text: str, user_id: int) -> Tuple[bool, Optional[str]]:
        """Optimized flood detection"""
        # Rate limiting first (cheapest check)
        if not self._check_rate_limit(user_id):
            return True, "Rate limit exceeded"

        # Early exit for very short messages
        if len(text) < 5:
            return False, None

        # Extract features
        features = self._extract_features(text)

        # Rule-based checks
        if features[0] > self.max_repeat_threshold:  # Z-repeat score
            return True, f"Repeated pattern detected (s:{features[0]:.2f})"

        # Only apply entropy check for longer messages
        if len(text) >= 48 and features[2] < self.min_entropy_threshold:  # Entropy
            return True, f"Low entropy (s:{features[2]:.2f})"

        return False, None

    def _check_rate_limit(self, user_id: int) -> bool:
        """Efficient rate limiting"""
        now = time.time()

        # Remove old messages
        while (self.user_message_history[user_id] and
               now - self.user_message_history[user_id][0][0] > self.time_window):
            self.user_message_history[user_id].popleft()

        return len(self.user_message_history[user_id]) < self.max_requests_per_window

    def cleanup(self, max_age_seconds: int = 3600):
        """Periodic cleanup"""
        now = time.time()

        # Clean user history
        for user_id in list(self.user_message_history.keys()):
            while (self.user_message_history[user_id] and
                   now - self.user_message_history[user_id][0][0] > max_age_seconds):
                self.user_message_history[user_id].popleft()

            if not self.user_message_history[user_id]:
                del self.user_message_history[user_id]

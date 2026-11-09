# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

The analyzer:
  - Lowercases and tokenizes text
  - Identifies positive and negative words
  - Reverses a word's sentiment after a simple negation
  - Uses the detected signals to predict a mood
"""

import re
from typing import List, Optional, Tuple

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """A simple keyword-based mood classifier."""

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the starter lists unless custom lists are provided.
        positive_words = (
            positive_words if positive_words is not None else POSITIVE_WORDS
        )
        negative_words = (
            negative_words if negative_words is not None else NEGATIVE_WORDS
        )

        self.positive_words = {word.lower() for word in positive_words}
        self.negative_words = {word.lower() for word in negative_words}

        self.negation_words = {
            "not",
            "no",
            "never",
            "don't",
            "doesn't",
            "didn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "can't",
            "couldn't",
            "won't",
            "wouldn't",
            "shouldn't",
        }

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Lowercase text and extract words.

        Keep apostrophes inside words so contractions such as "don't"
        remain one token. Normalize curly apostrophes first.
        """
        cleaned = text.strip().lower().replace("’", "'")
        return re.findall(r"[a-z]+(?:'[a-z]+)?", cleaned)

    # ---------------------------------------------------------------------
    # Shared sentiment signals
    # ---------------------------------------------------------------------

    def _get_signals(self, text: str) -> List[Tuple[str, int]]:
        """
        Return each matched word or phrase and its sentiment value.

        Positive words contribute +1 and negative words contribute -1.
        A negation immediately before a matched word reverses its value.
        """
        tokens = self.preprocess(text)
        signals: List[Tuple[str, int]] = []

        for index, token in enumerate(tokens):
            value = 0

            if token in self.positive_words:
                value += 1
            if token in self.negative_words:
                value -= 1

            if value == 0:
                continue

            signal = token

            if index > 0 and tokens[index - 1] in self.negation_words:
                value *= -1
                signal = f"{tokens[index - 1]} {token}"

            signals.append((signal, value))

        return signals

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """Add the values of all detected sentiment signals."""
        score = 0

        for signal, value in self._get_signals(text):
            score += value

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Predict mixed when both positive and negative signals appear.

        Otherwise, use the total score to predict positive, negative,
        or neutral. No recognized signals produces neutral.
        """
        score = self.score_text(text)
        signals = self._get_signals(text)

        has_positive = any(value > 0 for _, value in signals)
        has_negative = any(value < 0 for _, value in signals)

        if has_positive and has_negative:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """Show tokens, effective sentiment signals, and total score."""
        signals = self._get_signals(text)

        positive_hits = [
            signal for signal, value in signals if value > 0
        ]
        negative_hits = [
            signal for signal, value in signals if value < 0
        ]
        score = sum(value for _, value in signals)

        return (
            f"Tokens = {self.preprocess(text)}; "
            f"Score = {score} "
            f"(positive: {positive_hits}, negative: {negative_hits})"
        )
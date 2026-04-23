"""
agent/intent_detector.py
Lightweight intent classifier for AutoStream agent.
Uses keyword heuristics as a fast pre-filter, with LLM fallback for ambiguous cases.
"""

import re
from typing import Literal

IntentType = Literal["greeting", "product_inquiry", "high_intent_lead", "other"]


# ─── Keyword heuristics ───────────────────────────────────────────────────────

GREETING_PATTERNS = [
    r"\b(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening))\b",
    r"^(hi|hey|hello)[!.,\s]*$"
]

PRODUCT_INQUIRY_PATTERNS = [
    r"\b(price|pricing|cost|plan|plans|feature|features|how much|subscription|tier|package)\b",
    r"\b(basic|pro|4k|caption|resolution|video|edit|limit|refund|support|cancel|policy|policies)\b",
    r"\b(do you (have|offer|provide|support)|is there|are there)\b",
    r"\b(tell me about|what does|do you offer|what is|explain|difference between)\b"
]

HIGH_INTENT_PATTERNS = [
    r"\b(sign up|signup|subscribe|buy|purchase|get started|start|join|i want|i'd like|ready to|let's go|i'm in)\b",
    r"\b(try|upgrade|go with|choose|pick|select)\b.{0,30}\b(plan|pro|basic|autostream)\b",
    r"\b(my\s+(youtube|instagram|tiktok|channel|page|account))\b",
    r"\b(for my|for our)\b.{0,20}\b(channel|business|brand|content)\b"
]


def _matches_any(text: str, patterns: list) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def classify_intent_heuristic(text: str) -> IntentType:
    """Fast keyword-based intent classifier."""
    # High intent takes priority over inquiry
    if _matches_any(text, HIGH_INTENT_PATTERNS):
        # Make sure it's not just a question about signing up
        if not re.search(r"\b(how do i|how to|can i|is it possible)\b", text.lower()):
            return "high_intent_lead"

    if _matches_any(text, GREETING_PATTERNS):
        # Pure greetings (no question embedded)
        if not _matches_any(text, PRODUCT_INQUIRY_PATTERNS + HIGH_INTENT_PATTERNS):
            return "greeting"

    if _matches_any(text, PRODUCT_INQUIRY_PATTERNS):
        return "product_inquiry"

    return "other"


# ─── LLM-based intent (used in main agent via system prompt) ──────────────────

INTENT_SYSTEM_PROMPT = """
You are an intent classifier for AutoStream, a SaaS video editing product.

Classify the user message into EXACTLY one of:
1. greeting       — simple hello/hi, no product question
2. product_inquiry — asking about features, pricing, policies, or comparisons
3. high_intent_lead — user signals they want to sign up, buy, or try the product

Reply with ONLY the label, nothing else. Examples:
- "Hi there!" → greeting
- "What's the Pro plan price?" → product_inquiry
- "I want to try Pro for my YouTube channel" → high_intent_lead
- "Do you have a refund policy?" → product_inquiry
- "Let's get started, I'm ready to subscribe" → high_intent_lead
"""


def build_intent_check_messages(user_message: str) -> list:
    """Build the message payload for a standalone intent check LLM call."""
    return [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message}
    ]

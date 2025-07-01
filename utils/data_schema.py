from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class UserInput:
    user_id: str
    text: str

@dataclass
class EmotionResult:
    emotions: str
    confidence: float
    details: Optional[Dict] = None

@dataclass
class SuggestionResult:
    suggestion: str
    tailored_prompt: Optional[str] = None

@dataclass
class ForecastResult:
    forecast: str
    similar_past_moods: List[str]

@dataclass
class AnalyzeResponse:
    emotions: str
    confidence: float
    suggestion: str
    tailored_prompt: Optional[str]
    forecast: str
    similar_past_moods: List[str]

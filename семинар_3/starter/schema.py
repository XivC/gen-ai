from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

ASPECTS = [
    "производительность",
    "дизайн",
    "поддержка",
    "цена",
    "реклама",
    "надёжность",
]

class Issue(BaseModel):
    category: Literal[
        "производительность",
        "дизайн",
        "поддержка",
        "цена",
        "реклама",
        "надёжность",
    ]
    severity: int = Field(ge=1, le=5)
    quote: str


class Review(BaseModel):
    author: str
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    platform: Optional[Literal["android", "ios", "other"]] = None
    review_date: Optional[str] = None
    issues: list[Issue]
    competitor_mentions: list[str] = Field(default_factory=list)

    @field_validator("review_date")
    @classmethod
    def review_date_not_in_future(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%B %d %Y", "%B %d, %Y"):
            try:
                from datetime import datetime

                parsed = datetime.strptime(value.strip(), fmt).date()
                if parsed > date.today():
                    raise ValueError(f"Дата отзыва {value} в будущем")
                return value
            except ValueError as e:
                if "в будущем" in str(e):
                    raise
                continue
        return value


class AspectSentiment(BaseModel):
    aspect: Literal[
        "производительность",
        "дизайн",
        "поддержка",
        "цена",
        "реклама",
        "надёжность",
    ]
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class ReviewSentiment(BaseModel):
    name: str
    aspects: list[AspectSentiment]


class ChunkSummary(BaseModel):
    speaker: str
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]


class DiscussionSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str


class MultiDocSummary(BaseModel):
    common_themes: list[str] = Field(min_length=1, max_length=8)
    unique_per_bank: dict[str, list[str]]
    overall_headline: str

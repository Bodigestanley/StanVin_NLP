
from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    text: str = Field(
        min_length=3,
        max_length=3000
    )
    model: Literal["tfidf", "distilbert"] = "tfidf"


class BatchRequest(BaseModel):
    texts: list[str] = Field(
        min_length=1,
        max_length=15000
    )
    model: Literal["tfidf", "distilbert"] = "tfidf"

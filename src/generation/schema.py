# src/generation/schema.py
from pydantic import BaseModel, field_validator
from typing import Optional, Literal, List

Urgency = Literal["immédiate", "rapide (<6h)", "standard", "aucune"]

class Recommendation(BaseModel):
    modalite_recommandee: str
    symptomes_cles: List[str]
    hypothese_clinique: Optional[str] = None
    delai_recommande: Optional[str] = None  
    urgence: Urgency
    justification: str
    reference: str  # "ADERIM 2025 — <Section>"
    alternative: Optional[str] = None

    @field_validator("symptomes_cles", mode="before")
    def _norm_sympt(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v
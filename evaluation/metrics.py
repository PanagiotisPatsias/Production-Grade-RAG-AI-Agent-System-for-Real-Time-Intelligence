from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class EvaluationScores:
    "Basic metrics for reposponse evaluation for RAG systems."

    relevance: float
    correctness: float
    grounding: float
    completeness: float
    overall: float
    reasoning_quality: float

    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance": self.relevance,
            "correctness": self.correctness,
            "grounding": self.grounding,
            "completeness": self.completeness,
            "overall": self.overall,
            "reasoning_quality": self.reasoning_quality,
            "explanation": self.explanation,
        }
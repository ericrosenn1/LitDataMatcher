"""Extraction module for research questions, limitations, and future directions."""

from .research_questions import find_research_questions
from .limitations import find_limitations
from .future_directions import find_future_directions

__all__ = [
    'find_research_questions',
    'find_limitations',
    'find_future_directions',
]

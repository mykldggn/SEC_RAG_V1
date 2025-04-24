"""
Text‐cleaning and section‐splitting utilities.
"""
import re
from typing import List


def html_to_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def split_sections(text: str) -> List[str]:
    """
    Split filing text into sections by headings like 'Item 1.', 'Item 1A.', etc.
    """
    pattern = re.compile(r"(Item\s+\d+[A]?\.)", re.IGNORECASE)
    parts = pattern.split(text)
    sections = []
    for i in range(1, len(parts), 2):
        heading = parts[i]
        body = parts[i+1]
        sections.append(heading + body)
    return sections

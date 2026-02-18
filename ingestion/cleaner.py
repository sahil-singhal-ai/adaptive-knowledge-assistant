
import re

def clean_document_text(text: str) -> str:
    """
    Generic document cleaner for RAG preprocessing.
    Removes boilerplate, tables of contents, excessive formatting,
    repeated headers/footers, and noisy artifacts.
    """

    # ------------------------------------
    # 1. Normalize line endings
    # ------------------------------------
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ------------------------------------
    # 2. Remove common boilerplate patterns
    # ------------------------------------
    boilerplate_patterns = [
        r"This text is provided.*?\n",
        r"This document is provided.*?\n",
        r"All rights reserved.*?\n",
        r"Copyright .*?\n",
        r"Project Gutenberg.*?\n",
        r"End of Project Gutenberg.*?\n",
    ]

    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # ------------------------------------
    # 3. Remove Table of Contents sections
    # (heuristic: short lines with page-like structure)
    # ------------------------------------
    text = re.sub(
        r"Table of Contents.*?\n\n",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # ------------------------------------
    # 4. Remove repeated short lines (likely headers/footers)
    # ------------------------------------
    lines = text.split("\n")
    line_counts = {}

    for line in lines:
        stripped = line.strip()
        if 0 < len(stripped) < 80:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped in line_counts
            and line_counts[stripped] > 5  # appears too frequently
            and len(stripped) < 80
        ):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # ------------------------------------
    # 5. Remove excessive whitespace
    # ------------------------------------
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ------------------------------------
    # 6. Remove long sequences of punctuation
    # ------------------------------------
    text = re.sub(r"[-=_]{5,}", "", text)

    # ------------------------------------
    # 7. Trim leading/trailing whitespace
    # ------------------------------------
    text = text.strip()

    return text

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from app.models import PageText


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_pdf_pages(docs_dir: Path) -> list[PageText]:
    pages: list[PageText] = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        if pdf_path.name == "AI_Engineer_Take_Home_Options.pdf":
            continue
        reader = PdfReader(str(pdf_path))
        for index, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                pages.append(PageText(document=pdf_path.name, page=index, text=text))
    return pages


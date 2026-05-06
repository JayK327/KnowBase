# src/ingestion/extractors/html_extractor.py
"""
HTML Extractor (BeautifulSoup4)
===============================

Extracts clean, structured text from HTML pages by:
- Removing boilerplate elements (nav, footer, scripts, etc.)
- Preserving headings, paragraphs, and lists
- Keeping a readable document structure for downstream RAG pipelines
"""

import re
import logging
from bs4 import BeautifulSoup
from src.ingestion.models import RawDocument, ExtractedDocument, DocumentFormat

logger = logging.getLogger(__name__)

REMOVE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "button"]
HEADING_MAP = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### "}


class HtmlExtractor:
    def _clean(self, text: str) -> str:
        # Normalize excessive whitespace and newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return re.sub(r" {2,}", " ", text).strip()

    def extract(self, doc: RawDocument) -> ExtractedDocument:
        html = doc.raw_bytes.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content / boilerplate elements
        for tag in REMOVE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        main = soup.find("main") or soup.find("article") or soup.find("body") or soup

        parts = []
        for el in main.descendants:
            if el.name in HEADING_MAP:
                t = el.get_text(strip=True)
                if t:
                    parts.append(f"\n\n{HEADING_MAP[el.name]}{t}\n")

            elif el.name == "p":
                t = el.get_text(separator=" ", strip=True)
                if t:
                    parts.append(t)

            elif el.name in ("ul", "ol"):
                for li in el.find_all("li", recursive=False):
                    t = li.get_text(separator=" ", strip=True)
                    if t:
                        parts.append(f"- {t}")

        return ExtractedDocument(
            doc_id=doc.doc_id,
            source_path=doc.source_path,
            text=self._clean("\n\n".join(parts)),
            format=DocumentFormat.HTML,
            title=title,
            metadata=doc.metadata,
        )
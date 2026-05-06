# src/ingestion/extractors/pdf_extractor.py
"""
PDF Extractor — Unstructured.io
==============================

Converts PDF files into structured text for downstream RAG processing.

It supports two extraction modes:
- hi_res → full layout + table detection (higher accuracy, slower, heavier deps)
- fast   → text-layer extraction only (lightweight fallback)

Headings (Title, Header) are converted into Markdown-style "##" sections so
they can naturally guide downstream chunking.

Tables are kept readable by converting them into plain structured text.
"""

import io
import logging
from src.ingestion.models import RawDocument, ExtractedDocument, DocumentFormat

logger = logging.getLogger(__name__)


class PDFExtractor:
    def __init__(self, use_hi_res: bool = False):
        # hi_res requires additional system dependencies (poppler, tesseract).
        # Disable it in CI or lightweight environments.
        self.use_hi_res = use_hi_res

    def _elements_to_text(self, elements: list) -> str:
        from unstructured.documents.elements import Title, Header, Table, ListItem

        parts = []

        for el in elements:
            if isinstance(el, (Title, Header)):
                parts.append(f"\n\n## {el.text.strip()}\n")

            elif isinstance(el, Table):
                parts.append(f"\n\n{el.text.strip()}\n")

            elif isinstance(el, ListItem):
                parts.append(f"- {el.text.strip()}")

            elif hasattr(el, "text") and el.text.strip():
                parts.append(el.text.strip())

        return "\n\n".join(p for p in parts if p.strip())

    def extract(self, doc: RawDocument) -> ExtractedDocument:
        from unstructured.partition.pdf import partition_pdf

        pdf_file = io.BytesIO(doc.raw_bytes)
        strategy = "hi_res" if self.use_hi_res else "fast"

        try:
            elements = partition_pdf(
                file=pdf_file,
                strategy=strategy,
                infer_table_structure=True,
                include_page_breaks=True,
            )
        except Exception as exc:
            logger.warning(
                f"PDF extraction failed using '{strategy}', retrying with fast mode: {exc}"
            )
            pdf_file.seek(0)
            elements = partition_pdf(file=pdf_file, strategy="fast")

        text = self._elements_to_text(elements)

        # Estimate number of pages from page break markers
        num_pages = sum(
            1 for e in elements if e.__class__.__name__ == "PageBreak"
        ) + 1

        return ExtractedDocument(
            doc_id=doc.doc_id,
            source_path=doc.source_path,
            text=text,
            format=DocumentFormat.PDF,
            num_pages=num_pages,
            metadata={**doc.metadata, "num_elements": len(elements)},
        )
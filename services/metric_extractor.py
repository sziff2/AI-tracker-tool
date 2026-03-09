"""
Metric Extraction Service (§7)

Responsibilities:
  - Extract data from documents using TYPE-SPECIFIC prompts
  - Store structured results with evidence snippets
  - Flag low-confidence items for review
"""

import json
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Document, ExtractedMetric, ReviewQueueItem
from prompts import (
    KPI_EXTRACTOR, GUIDANCE_EXTRACTOR,
    EARNINGS_RELEASE_EXTRACTOR, TRANSCRIPT_EXTRACTOR,
    BROKER_NOTE_EXTRACTOR, PRESENTATION_EXTRACTOR,
)
from schemas import ExtractedKPI, GuidanceItem
from services.llm_client import call_llm_json

logger = logging.getLogger(__name__)

REVIEW_THRESHOLD = 0.8

# Map document types to their specialised prompts
DOCTYPE_PROMPTS = {
    "earnings_release": EARNINGS_RELEASE_EXTRACTOR,
    "10-Q": EARNINGS_RELEASE_EXTRACTOR,
    "10-K": EARNINGS_RELEASE_EXTRACTOR,
    "annual_report": EARNINGS_RELEASE_EXTRACTOR,
    "transcript": TRANSCRIPT_EXTRACTOR,
    "broker_note": BROKER_NOTE_EXTRACTOR,
    "presentation": PRESENTATION_EXTRACTOR,
}


def _chunk_text(text: str, max_chars: int = 15000) -> list[str]:
    """Split text into chunks that fit within prompt limits."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    lines = text.split("\n")
    current = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


async def extract_by_document_type(
    db: AsyncSession, document: Document, text: str
) -> dict:
    """
    Run the document-type-specific prompt. Returns raw extracted items
    and also persists standard metrics where applicable.
    """
    doc_type = document.document_type or "other"
    prompt_template = DOCTYPE_PROMPTS.get(doc_type, KPI_EXTRACTOR)

    chunks = _chunk_text(text)
    all_raw_items = []

    for chunk in chunks:
        prompt = prompt_template.format(text=chunk)
        try:
            raw_items = call_llm_json(prompt, max_tokens=8192)
            if not isinstance(raw_items, list):
                raw_items = [raw_items]
            all_raw_items.extend(raw_items)
        except Exception as e:
            logger.warning("Extraction failed for chunk (%s): %s", doc_type, str(e)[:200])
            continue

    logger.info("Extracted %d items from %s document %s", len(all_raw_items), doc_type, document.id)

    # For earnings/financials, also persist as standard metrics
    if doc_type in ("earnings_release", "10-Q", "10-K", "annual_report"):
        await _persist_earnings_metrics(db, document, all_raw_items)

    # For transcripts, persist guidance items as metrics
    if doc_type == "transcript":
        await _persist_transcript_items(db, document, all_raw_items)

    return {
        "document_type": doc_type,
        "items_extracted": len(all_raw_items),
        "raw_items": all_raw_items,
    }


async def _persist_earnings_metrics(db, document, raw_items):
    """Persist earnings extraction results as ExtractedMetric rows."""
    for item in raw_items:
        try:
            confidence = item.get("confidence", 1.0)
            needs_review = confidence < REVIEW_THRESHOLD
            metric = ExtractedMetric(
                id=uuid.uuid4(),
                company_id=document.company_id,
                document_id=document.id,
                period_label=document.period_label,
                metric_name=item.get("metric_name", "unknown"),
                metric_value=item.get("metric_value"),
                metric_text=item.get("metric_text", ""),
                unit=item.get("unit"),
                segment=item.get("segment"),
                geography=item.get("geography"),
                source_snippet=item.get("source_snippet", ""),
                page_number=item.get("page_number"),
                confidence=confidence,
                needs_review=needs_review,
            )
            db.add(metric)
            if needs_review:
                db.add(ReviewQueueItem(
                    id=uuid.uuid4(),
                    entity_type="metric",
                    entity_id=metric.id,
                    queue_reason=f"Low confidence ({confidence:.2f}) on {item.get('metric_name')}",
                    priority="high" if confidence < 0.5 else "normal",
                ))
        except Exception as e:
            logger.warning("Failed to persist metric: %s", str(e)[:100])
    await db.commit()


async def _persist_transcript_items(db, document, raw_items):
    """Persist transcript guidance items as ExtractedMetric rows."""
    for item in raw_items:
        try:
            category = item.get("category", "")
            if category == "guidance":
                metric = ExtractedMetric(
                    id=uuid.uuid4(),
                    company_id=document.company_id,
                    document_id=document.id,
                    period_label=document.period_label,
                    metric_name=f"GUIDANCE: {item.get('metric_name', 'unknown')}",
                    metric_value=item.get("high") or item.get("low"),
                    metric_text=item.get("guidance_text", ""),
                    unit=item.get("unit"),
                    segment="guidance",
                    source_snippet=item.get("source_snippet", ""),
                    confidence=item.get("confidence", 0.8),
                    needs_review=False,
                )
                db.add(metric)
        except Exception as e:
            logger.warning("Failed to persist transcript item: %s", str(e)[:100])
    await db.commit()


# ─────────────────────────────────────────────────────────────────
# Legacy functions (still used by single-doc pipeline)
# ─────────────────────────────────────────────────────────────────

async def extract_metrics(db: AsyncSession, document: Document, text: str) -> list[ExtractedMetric]:
    """Generic KPI extraction (legacy, used by single-doc upload)."""
    chunks = _chunk_text(text)
    all_raw_items = []
    for chunk in chunks:
        prompt = KPI_EXTRACTOR.format(text=chunk)
        try:
            raw_items = call_llm_json(prompt, max_tokens=8192)
            if not isinstance(raw_items, list):
                raw_items = [raw_items]
            all_raw_items.extend(raw_items)
        except Exception as e:
            logger.warning("KPI extraction failed for chunk: %s", str(e)[:200])
            continue

    metrics = []
    for item in all_raw_items:
        try:
            kpi = ExtractedKPI(**item)
            needs_review = kpi.confidence < REVIEW_THRESHOLD
            metric = ExtractedMetric(
                id=uuid.uuid4(),
                company_id=document.company_id,
                document_id=document.id,
                period_label=document.period_label,
                metric_name=kpi.metric_name,
                metric_value=kpi.metric_value,
                metric_text=kpi.metric_text,
                unit=kpi.unit,
                segment=kpi.segment,
                geography=kpi.geography,
                source_snippet=kpi.source_snippet,
                page_number=kpi.page_number,
                confidence=kpi.confidence,
                needs_review=needs_review,
            )
            db.add(metric)
            metrics.append(metric)
            if needs_review:
                db.add(ReviewQueueItem(
                    id=uuid.uuid4(),
                    entity_type="metric",
                    entity_id=metric.id,
                    queue_reason=f"Low confidence ({kpi.confidence:.2f}) on {kpi.metric_name}",
                    priority="high" if kpi.confidence < 0.5 else "normal",
                ))
        except Exception as e:
            logger.warning("Failed to parse metric: %s", str(e)[:100])
    await db.commit()
    logger.info("Extracted %d metrics from document %s", len(metrics), document.id)
    return metrics


async def extract_guidance(db: AsyncSession, document: Document, text: str) -> list[dict]:
    """Generic guidance extraction (legacy)."""
    chunks = _chunk_text(text)
    all_raw_items = []
    for chunk in chunks:
        prompt = GUIDANCE_EXTRACTOR.format(text=chunk)
        try:
            raw_items = call_llm_json(prompt, max_tokens=8192)
            if not isinstance(raw_items, list):
                raw_items = [raw_items]
            all_raw_items.extend(raw_items)
        except Exception as e:
            logger.warning("Guidance extraction failed for chunk: %s", str(e)[:200])
            continue

    guidance_records = []
    for item in all_raw_items:
        try:
            gi = GuidanceItem(**item)
            metric = ExtractedMetric(
                id=uuid.uuid4(),
                company_id=document.company_id,
                document_id=document.id,
                period_label=document.period_label,
                metric_name=f"GUIDANCE: {gi.metric_name}",
                metric_value=gi.high if gi.high else gi.low,
                metric_text=gi.guidance_text,
                unit=gi.unit,
                segment="guidance",
                source_snippet=gi.source_snippet,
                confidence=gi.confidence,
                needs_review=gi.confidence < REVIEW_THRESHOLD,
            )
            db.add(metric)
            guidance_records.append(gi.model_dump())
        except Exception as e:
            logger.warning("Failed to parse guidance: %s", str(e)[:100])
    await db.commit()
    logger.info("Extracted %d guidance items from document %s", len(guidance_records), document.id)
    return guidance_records

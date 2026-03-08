"""
Metric Extraction Service (§7)

Responsibilities:
  - Extract quantitative KPIs from parsed text
  - Extract management guidance
  - Store metrics with evidence snippets
  - Flag low-confidence items for review
"""

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Document, ExtractedMetric, ReviewQueueItem
from prompts import KPI_EXTRACTOR, GUIDANCE_EXTRACTOR
from schemas import ExtractedKPI, GuidanceItem
from services.llm_client import call_llm_json

logger = logging.getLogger(__name__)

REVIEW_THRESHOLD = 0.8  # confidence below this → review queue


# ─────────────────────────────────────────────────────────────────
# KPI extraction
# ─────────────────────────────────────────────────────────────────

async def extract_metrics(db: AsyncSession, document: Document, text: str) -> list[ExtractedMetric]:
    """
    Run the KPI extraction prompt against the document text,
    persist results, and flag low-confidence items.
    """
    prompt = KPI_EXTRACTOR.format(text=text)
    raw_items = call_llm_json(prompt)
    if not isinstance(raw_items, list):
        raw_items = [raw_items]

    metrics: list[ExtractedMetric] = []
    for item in raw_items:
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

    await db.commit()
    logger.info("Extracted %d metrics from document %s", len(metrics), document.id)
    return metrics


# ─────────────────────────────────────────────────────────────────
# Guidance extraction
# ─────────────────────────────────────────────────────────────────

async def extract_guidance(db: AsyncSession, document: Document, text: str) -> list[dict]:
    """
    Extract forward-looking guidance statements.
    Returns parsed items (not persisted as separate table in MVP —
    stored alongside metrics with segment='guidance').
    """
    prompt = GUIDANCE_EXTRACTOR.format(text=text)
    raw_items = call_llm_json(prompt)
    if not isinstance(raw_items, list):
        raw_items = [raw_items]

    guidance_records = []
    for item in raw_items:
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

    await db.commit()
    logger.info("Extracted %d guidance items from document %s", len(guidance_records), document.id)
    return guidance_records

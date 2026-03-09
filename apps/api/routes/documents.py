"""
Document endpoints (§8): upload, list, process, extract, compare.
"""

import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, Document, DocumentSection, ExtractedMetric, EventAssessment, ResearchOutput, ReviewQueueItem
from schemas import DocumentCreate, DocumentOut
from services.document_ingestion import ingest_document
from services.document_parser import process_document
from services.metric_extractor import extract_metrics, extract_guidance
from services.thesis_comparator import compare_thesis

router = APIRouter(tags=["documents"])


# ─────────────────────────────────────────────────────────────────
# List documents for a company
# ─────────────────────────────────────────────────────────────────
@router.get("/companies/{ticker}/documents", response_model=list[DocumentOut])
async def list_documents(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Document)
        .join(Company)
        .where(Company.ticker == ticker.upper())
        .order_by(Document.created_at.desc())
    )
    return result.scalars().all()


# ─────────────────────────────────────────────────────────────────
# Upload a document
# ─────────────────────────────────────────────────────────────────
@router.post("/companies/{ticker}/documents/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    ticker: str,
    file: UploadFile = File(...),
    document_type: str = Form(...),
    period_label: str = Form(...),
    title: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    # Look up company
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Save upload to temp file
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = await ingest_document(
            db=db,
            company_id=company.id,
            ticker=company.ticker,
            file_path=tmp_path,
            filename=file.filename or f"upload{suffix}",
            document_type=document_type,
            period_label=period_label,
            title=title,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc))

    return doc


# ─────────────────────────────────────────────────────────────────
# Upload + full auto pipeline
# ─────────────────────────────────────────────────────────────────
@router.post("/companies/{ticker}/documents/upload-and-process", status_code=200)
async def upload_and_process(
    ticker: str,
    file: UploadFile = File(...),
    document_type: str = Form("earnings_release"),
    period_label: str = Form(...),
    title: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    One-click pipeline: upload → parse → extract KPIs → compare thesis → generate all outputs.
    """
    import json as _json
    from services.surprise_detector import detect_surprises
    from services.output_generator import generate_briefing, generate_ir_questions, generate_thesis_drift_report

    # Look up company
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    pipeline_log = []

    # ── Step 1: Ingest ───────────────────────────────────────────
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = await ingest_document(
            db=db,
            company_id=company.id,
            ticker=company.ticker,
            file_path=tmp_path,
            filename=file.filename or f"upload{suffix}",
            document_type=document_type,
            period_label=period_label,
            title=title,
        )
        pipeline_log.append({"step": "upload", "status": "ok", "document_id": str(doc.id)})
    except ValueError as exc:
        raise HTTPException(409, str(exc))

    # ── Step 2: Parse ────────────────────────────────────────────
    try:
        parse_result = await process_document(db, doc, ticker=company.ticker)
        pipeline_log.append({"step": "parse", "status": "ok", "pages": parse_result["pages"], "tables": parse_result["tables_found"]})
    except Exception as e:
        pipeline_log.append({"step": "parse", "status": "error", "detail": str(e)[:200]})
        return {"pipeline": pipeline_log}

    # ── Step 3: Extract KPIs ─────────────────────────────────────
    try:
        from configs.settings import settings as _settings
        text_path = Path(_settings.storage_base_path) / "processed" / company.ticker / period_label / "parsed_text.json"
        pages = _json.loads(text_path.read_text())
        full_text = "\n\n".join(p["text"] for p in pages)

        metrics = await extract_metrics(db, doc, full_text)
        guidance = await extract_guidance(db, doc, full_text)
        pipeline_log.append({"step": "extract", "status": "ok", "metrics": len(metrics), "guidance": len(guidance)})
    except Exception as e:
        pipeline_log.append({"step": "extract", "status": "error", "detail": str(e)[:200]})

    # ── Step 4: Compare thesis ───────────────────────────────────
    try:
        comparison = await compare_thesis(db, company.id, doc.id, period_label)
        pipeline_log.append({"step": "compare", "status": "ok", "direction": comparison.thesis_direction})
    except ValueError:
        pipeline_log.append({"step": "compare", "status": "skipped", "detail": "No active thesis"})
    except Exception as e:
        pipeline_log.append({"step": "compare", "status": "error", "detail": str(e)[:200]})

    # ── Step 5: Detect surprises ─────────────────────────────────
    try:
        surprises = await detect_surprises(db, company.id, doc.id, period_label)
        pipeline_log.append({"step": "surprises", "status": "ok", "count": len(surprises)})
    except Exception as e:
        pipeline_log.append({"step": "surprises", "status": "error", "detail": str(e)[:200]})

    # ── Step 6: Generate outputs ─────────────────────────────────
    try:
        briefing = await generate_briefing(db, company.id, period_label)
        pipeline_log.append({"step": "briefing", "status": "ok"})
    except Exception as e:
        pipeline_log.append({"step": "briefing", "status": "error", "detail": str(e)[:200]})

    try:
        questions = await generate_ir_questions(db, company.id, period_label)
        pipeline_log.append({"step": "ir_questions", "status": "ok", "count": len(questions)})
    except Exception as e:
        pipeline_log.append({"step": "ir_questions", "status": "error", "detail": str(e)[:200]})

    try:
        drift = await generate_thesis_drift_report(db, company.id, period_label)
        pipeline_log.append({"step": "thesis_drift", "status": "ok"})
    except Exception as e:
        pipeline_log.append({"step": "thesis_drift", "status": "error", "detail": str(e)[:200]})

    return {"pipeline": pipeline_log}


# ─────────────────────────────────────────────────────────────────
# Get single document
# ─────────────────────────────────────────────────────────────────
@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


# ─────────────────────────────────────────────────────────────────
# Process (parse) a document
# ─────────────────────────────────────────────────────────────────
@router.post("/documents/{document_id}/process")
async def process_doc(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")
    # Load company to get ticker (avoid lazy load in async context)
    company_result = await db.execute(select(Company).where(Company.id == doc.company_id))
    company = company_result.scalar_one_or_none()
    ticker = company.ticker if company else "UNKNOWN"

    # Restore file from DB if missing on disk (Railway redeploys wipe filesystem)
    file_path = Path(doc.file_path)
    if not file_path.exists() and doc.file_content:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(doc.file_content)

    if not file_path.exists():
        raise HTTPException(400, "File not found on disk and no content stored in DB. Please re-upload.")

    summary = await process_document(db, doc, ticker=ticker)
    return summary


# ─────────────────────────────────────────────────────────────────
# Extract metrics from a document
# ─────────────────────────────────────────────────────────────────
@router.post("/documents/{document_id}/extract")
async def extract_doc(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Load company to get ticker (avoid lazy load in async context)
    company_result = await db.execute(select(Company).where(Company.id == doc.company_id))
    company = company_result.scalar_one_or_none()
    ticker = company.ticker if company else "UNKNOWN"

    # Read parsed text
    from configs.settings import settings
    proc_dir = Path(settings.storage_base_path) / "processed"
    text_path = proc_dir / ticker / (doc.period_label or "misc") / "parsed_text.json"

    if not text_path.exists():
        raise HTTPException(400, "Document has not been processed yet. Call /process first.")

    import json
    pages = json.loads(text_path.read_text())
    full_text = "\n\n".join(p["text"] for p in pages)

    metrics = await extract_metrics(db, doc, full_text)
    guidance = await extract_guidance(db, doc, full_text)

    return {
        "metrics_extracted": len(metrics),
        "guidance_items": len(guidance),
    }


# ─────────────────────────────────────────────────────────────────
# Compare document against thesis
# ─────────────────────────────────────────────────────────────────
@router.post("/documents/{document_id}/compare")
async def compare_doc(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    try:
        comparison = await compare_thesis(db, doc.company_id, doc.id, doc.period_label)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return comparison.model_dump()


# ─────────────────────────────────────────────────────────────────
# Delete a single document
# ─────────────────────────────────────────────────────────────────
@router.delete("/documents/{document_id}")
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Delete related records first
    await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id == document_id))
    await db.execute(delete(EventAssessment).where(EventAssessment.document_id == document_id))
    await db.execute(delete(ExtractedMetric).where(ExtractedMetric.document_id == document_id))
    await db.execute(delete(DocumentSection).where(DocumentSection.document_id == document_id))
    await db.delete(doc)
    await db.commit()
    return {"status": "deleted", "document_id": str(document_id)}


# ─────────────────────────────────────────────────────────────────
# Delete ALL documents for a company
# ─────────────────────────────────────────────────────────────────
@router.delete("/companies/{ticker}/documents")
async def delete_all_documents(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    try:
        docs = await db.execute(select(Document).where(Document.company_id == company.id))
        doc_ids = [d.id for d in docs.scalars().all()]

        if doc_ids:
            # Get metric and assessment IDs for review queue cleanup
            metrics = await db.execute(select(ExtractedMetric.id).where(ExtractedMetric.document_id.in_(doc_ids)))
            metric_ids = [m[0] for m in metrics.all()]

            assessments = await db.execute(select(EventAssessment.id).where(EventAssessment.document_id.in_(doc_ids)))
            assessment_ids = [a[0] for a in assessments.all()]

            # Delete review queue items (could reference metrics, assessments, or outputs)
            all_entity_ids = doc_ids + metric_ids + assessment_ids
            if all_entity_ids:
                await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(all_entity_ids)))

            # Delete in dependency order
            await db.execute(delete(EventAssessment).where(EventAssessment.document_id.in_(doc_ids)))
            await db.execute(delete(ExtractedMetric).where(ExtractedMetric.document_id.in_(doc_ids)))
            await db.execute(delete(DocumentSection).where(DocumentSection.document_id.in_(doc_ids)))
            await db.execute(delete(Document).where(Document.company_id == company.id))

        # Also clean up research outputs and their review items
        outputs = await db.execute(select(ResearchOutput.id).where(ResearchOutput.company_id == company.id))
        output_ids = [o[0] for o in outputs.all()]
        if output_ids:
            await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(output_ids)))
            await db.execute(delete(ResearchOutput).where(ResearchOutput.company_id == company.id))

        await db.commit()
        return {"status": "deleted", "documents_removed": len(doc_ids)}

    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f"Delete failed: {str(e)}")

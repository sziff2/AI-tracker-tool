"""
Document endpoints (§8): upload, list, process, extract, compare.
"""

import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, Document
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

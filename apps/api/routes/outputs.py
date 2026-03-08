"""
Output generation endpoints (§8): briefings, IR questions, thesis drift.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, ResearchOutput
from schemas import ResearchOutputOut
from services.output_generator import (
    generate_briefing,
    generate_ir_questions,
    generate_thesis_drift_report,
)

router = APIRouter(tags=["outputs"])


@router.get("/companies/{ticker}/outputs", response_model=list[ResearchOutputOut])
async def list_outputs(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ResearchOutput)
        .join(Company)
        .where(Company.ticker == ticker.upper())
        .order_by(ResearchOutput.created_at.desc())
    )
    return result.scalars().all()


@router.post("/companies/{ticker}/generate-briefing")
async def briefing(ticker: str, period_label: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    b = await generate_briefing(db, company.id, period_label)
    return b.model_dump()


@router.post("/companies/{ticker}/generate-ir-questions")
async def ir_questions(ticker: str, period_label: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    questions = await generate_ir_questions(db, company.id, period_label)
    return [q.model_dump() for q in questions]


@router.post("/companies/{ticker}/generate-thesis-drift")
async def thesis_drift(ticker: str, period_label: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    try:
        drift = await generate_thesis_drift_report(db, company.id, period_label)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return drift

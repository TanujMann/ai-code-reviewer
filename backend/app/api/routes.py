"""
routes.py
=========
All API endpoint definitions.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.models.schemas import ReviewRequest, ReviewResponse, HealthResponse
from app.services.llm_service import LLMService, get_llm_service
from app.core.config import settings

router = APIRouter()


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(service: LLMService = Depends(get_llm_service)):
    """Check if the API and model are ready."""
    return HealthResponse(
        status="healthy" if service.is_ready else "degraded",
        model_loaded=service.is_ready,
        model_backend=service.backend_name,
        version=settings.APP_VERSION,
    )


# ─────────────────────────────────────────────
# MAIN REVIEW ENDPOINT
# ─────────────────────────────────────────────

@router.post("/review", response_model=ReviewResponse, tags=["Review"])
async def review_code(
    request: ReviewRequest,
    service: LLMService = Depends(get_llm_service),
):
    """
    Submit code for AI review.
    
    Returns detailed analysis including:
    - Quality score (0-100)
    - Bugs with line numbers
    - Security vulnerabilities
    - Improvement suggestions
    """
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if len(request.code) > settings.MAX_CODE_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Code too long. Max {settings.MAX_CODE_LENGTH} characters."
        )
    
    try:
        logger.info(f"Review request | lang={request.language} | size={len(request.code)}")
        
        review = await service.review_code(
            code=request.code,
            language=request.language.value,
            filename=request.filename,
            focus_areas=request.focus_areas,
        )
        
        logger.info(f"Review complete | score={review.score} | time={review.review_time_ms}ms")
        return review
        
    except Exception as e:
        logger.error(f"Review failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Review failed: {str(e)}")


# ─────────────────────────────────────────────
# QUICK SCAN (faster, less detailed)
# ─────────────────────────────────────────────

@router.post("/quick-scan", tags=["Review"])
async def quick_scan(
    request: ReviewRequest,
    service: LLMService = Depends(get_llm_service),
):
    """
    Fast security-only scan.
    Returns only critical bugs and security issues.
    """
    review = await service.review_code(
        code=request.code,
        language=request.language.value,
        focus_areas=["security", "critical_bugs"]
    )
    
    # Return only critical info
    return {
        "score": review.score,
        "grade": review.grade,
        "critical_issues": [
            b for b in review.bugs if b.severity.value in ["critical", "high"]
        ],
        "security_issues": review.security_issues,
        "review_time_ms": review.review_time_ms,
    }


# ─────────────────────────────────────────────
# SUPPORTED LANGUAGES
# ─────────────────────────────────────────────

@router.get("/languages", tags=["System"])
async def get_supported_languages():
    """Returns list of supported programming languages."""
    return {
        "languages": [
            {"id": "python", "name": "Python", "extensions": [".py"]},
            {"id": "javascript", "name": "JavaScript", "extensions": [".js", ".mjs"]},
            {"id": "typescript", "name": "TypeScript", "extensions": [".ts", ".tsx"]},
            {"id": "java", "name": "Java", "extensions": [".java"]},
            {"id": "cpp", "name": "C++", "extensions": [".cpp", ".cc", ".cxx"]},
            {"id": "c", "name": "C", "extensions": [".c", ".h"]},
            {"id": "go", "name": "Go", "extensions": [".go"]},
            {"id": "rust", "name": "Rust", "extensions": [".rs"]},
        ]
    }

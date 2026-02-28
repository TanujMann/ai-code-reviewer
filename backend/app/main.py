"""
main.py
=======
FastAPI application entry point.
Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from loguru import logger

from app.core.config import settings
from app.api.routes import router
from app.services.llm_service import get_llm_service


# ─────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    logger.info("=" * 50)
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"   Model backend: {settings.MODEL_BACKEND}")
    logger.info("=" * 50)
    
    # Pre-load the LLM service
    service = get_llm_service()
    logger.info(f"✅ API ready! Docs at: http://localhost:{settings.PORT}/docs")
    
    yield  # App runs here
    
    logger.info("Shutting down...")


# ─────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## 🤖 AI Code Review Assistant API

Submit code snippets and receive:
- **Quality scores** (0-100)
- **Bug detection** with line numbers
- **Security vulnerability** analysis
- **Improvement suggestions** with code examples

### Backends
- `demo`: Rule-based (works without GPU)
- `openai`: GPT-4o-mini (needs API key)
- `local`: Fine-tuned CodeLlama (needs GPU)

Set `MODEL_BACKEND` in `.env` to switch.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────

# CORS — allows VS Code extension and browser to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

app.include_router(router, prefix="/api/v1")

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    return RedirectResponse(url="/docs")


# ─────────────────────────────────────────────
# RUN DIRECTLY
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

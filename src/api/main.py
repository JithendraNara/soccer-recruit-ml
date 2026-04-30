"""FastAPI application."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.utils.config import settings
from src.utils.logger import logger
from src.data.database import init_db
from src.api.routers import players, similarity, prediction
from src.api.schemas.models import HealthResponse


def _load_models():
    """Load persisted models from disk on startup for faster first request."""
    import os
    from src.api.routers import similarity as sim_router, prediction as pred_router

    models_dir = "./models"
    if not os.path.exists(models_dir):
        return

    try:
        sim_file = os.path.join(models_dir, "similarity_model.joblib")
        if os.path.exists(sim_file):
            import joblib
            sim_router._model_registry._model = joblib.load(sim_file)
            logger.info("Loaded similarity model from disk")
    except Exception as e:
        logger.warning(f"Could not load similarity model: {e}")

    try:
        pred_file = os.path.join(models_dir, "prediction_model.joblib")
        if os.path.exists(pred_file):
            import joblib
            pred_router._prediction_model = joblib.load(pred_file)
            logger.info("Loaded prediction model from disk")
    except Exception as e:
        logger.warning(f"Could not load prediction model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager — startup and shutdown events."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    init_db()
    logger.info("Database initialized")
    _load_models()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Soccer Player Similarity & Recruitment Modeling Platform",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(players.router, prefix=settings.api_prefix)
app.include_router(similarity.router, prefix=settings.api_prefix)
app.include_router(prediction.router, prefix=settings.api_prefix)


@app.get("/", tags=["root"])
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        database="sqlite",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

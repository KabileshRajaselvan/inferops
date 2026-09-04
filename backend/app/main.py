import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import batch_predict, deployments, evaluate, feedback, models, predict
from app.config import get_settings
from app.core.kafka import producer_client
from app.core.mlflow_client import configure_mlflow

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("inferops")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await producer_client.start()
    configure_mlflow()
    logger.info("InferOps API ready")
    yield
    await producer_client.stop()


app = FastAPI(
    title="InferOps",
    description="ML model serving platform: A/B-tested inference, drift detection, and automatic retraining.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(batch_predict.router)
app.include_router(models.router)
app.include_router(deployments.router)
app.include_router(feedback.router)
app.include_router(evaluate.router)

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed; /metrics disabled")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

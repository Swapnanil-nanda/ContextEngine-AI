import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.rag.utils import get_logger

logger = get_logger(__name__)

import asyncio
import urllib.request

async def _keep_alive_worker():
    url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("KEEP_ALIVE_URL")
    if not url:
        return
    health_url = url.rstrip("/") + "/health"
    logger.info(f"Keep-alive worker enabled for {health_url}")
    try:
        await asyncio.sleep(120)
    except asyncio.CancelledError:
        return
    while True:
        try:
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "ContextEngine-KeepAlive/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.info(f"Keep-alive ping sent to {health_url} (status {resp.status})")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Keep-alive ping notice: {e}")
        try:
            await asyncio.sleep(540)
        except asyncio.CancelledError:
            break

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG API Service...")
    keep_alive_task = asyncio.create_task(_keep_alive_worker())
    try:
        yield
    finally:
        keep_alive_task.cancel()
        logger.info("Shutting down RAG API Service...")

app = FastAPI(
    title="RAG API Service",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

media_path = os.path.join(os.path.dirname(__file__), "static", "media")
os.makedirs(media_path, exist_ok=True)
app.mount("/media", StaticFiles(directory=media_path), name="media")

app.include_router(router)

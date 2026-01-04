import uvicorn

# from backend import linux
from .api import app
from .config import PORT
import logging
import sys
from fastapi import Request
from . import runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        body = await request.json()
    except Exception:
        body = None

    logger.info(
        {
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host,
            "body": body,
        }
    )

    response = await call_next(request)
    logger.info(f"<-- {request.method} {request.url.path} {response.status_code}")
    return response

def get_backend():
    """
    Dynamically choose which backend should be used depending on the OS 
    """
    if sys.platform == "darwin":
        from .backend import mlx
        logger.info("Using MLX backend (MacOs)")
        return mlx
    elif sys.platform.startswith("linux"):
        from .backend import linux
        logger.info(f"Using linux backend {sys.platform}")
        return linux
    else:
        raise RuntimeError(f"Unsupported OS: {sys.platform}")

runtime.backend = get_backend()

def run():
    uvicorn.run(app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    run()

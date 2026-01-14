import sys
import logging

logger = logging.getLogger("app")

def get_backend():
    """
    Dynamically choose which backend should be used depending on the OS 
    """
    if sys.platform == "darwin":
        from .backend import mlx_backend
        return mlx_backend
    elif sys.platform.startswith("linux"):
        from .backend import linux
        return linux
    else:
        return None

backend = get_backend()

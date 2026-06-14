from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.analysis import router as analysis_router

router = APIRouter()
router.include_router(health_router)
router.include_router(analysis_router)

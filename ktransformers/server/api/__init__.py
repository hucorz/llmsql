from fastapi import APIRouter

from .openai import router as openai_router, post_db_creation_operations

router = APIRouter()
router.include_router(openai_router)

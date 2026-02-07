"""REST API for model information."""

from __future__ import annotations

from fastapi import APIRouter

from pi.ai import get_models, get_providers

router = APIRouter(prefix="/api/models", tags=["models"])


def create_models_router() -> APIRouter:

    @router.get("")
    async def list_models():
        result = []
        for provider_name in get_providers():
            models = get_models(provider_name)
            result.append({
                "name": provider_name,
                "models": [m.model_dump(by_alias=True) for m in models],
            })
        return result

    return router

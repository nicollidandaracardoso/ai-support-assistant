from fastapi import FastAPI

from app.schemas import HealthResponse

app = FastAPI(
    title="AI Support Assistant",
    description="API para um assistente de suporte baseado em documentação.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")

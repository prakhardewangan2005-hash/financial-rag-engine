import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes import router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise-Grade Financial Filings & Regulatory Compliance RAG Engine built from first principles."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response

app.include_router(router, prefix=settings.API_V1_PREFIX)
app.include_router(router)  # root mount for /health and /docs

@app.get("/")
def root_endpoint():
    return {
        "message": "Financial & Regulatory RAG Engine API is running.",
        "docs_url": "/docs",
        "health_check": "/health",
        "query_endpoint": f"{settings.API_V1_PREFIX}/query"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

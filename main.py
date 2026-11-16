"""FastAPI application entry point for Household Power Forecasting."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.routers import forecast

app = FastAPI(
    title="Household Power Forecasting API",
    description=(
        "Next-day (24-hour ahead) hourly household electricity consumption "
        "forecasting powered by a two-layer LSTM neural network."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include forecasting routes
app.include_router(forecast.router, prefix="/api")


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root access directly to Swagger interactive API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/api", tags=["General"])
async def api_info():
    """Basic API metadata and discovery information."""
    return {
        "name": "Household Power Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/forecast/health",
            "predict_csv": "/api/forecast/predict/csv",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

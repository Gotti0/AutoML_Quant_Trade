import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import routes

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AutoML Quant Trade API",
    description="Backend API for the Quantitative Trading Dashboard",
    version="1.0.0"
)

# Configuration for CORS to allow the Vite React frontend to access the API
origins = [
    "http://localhost:5173", # Vite default port
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main router
app.include_router(routes.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AutoML Quant Trade API is running"}

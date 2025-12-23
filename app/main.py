from fastapi import FastAPI
from app.api.health import router as health_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(health_router)
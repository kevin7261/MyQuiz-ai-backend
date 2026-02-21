from fastapi import FastAPI

from routers.zip import router as zip_router

app = FastAPI()

app.include_router(zip_router)


@app.get("/")
def read_root():
    return {"status": "Server is running"}


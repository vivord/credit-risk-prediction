from fastapi import FastAPI
from pydantic import BaseModel
from src.deployment.predict import predict_risk

app = FastAPI(title="Credit Risk API")

class LoanApplication(BaseModel):
    Duration_in_month: int
    Credit_amount: float
    Age_in_years: int
    Status_of_existing_account: str = "A11"
    Credit_history: str = "A32"
    Purpose: str = "A43"
    # Add other fields matching your features...
@app.get("/")
async def root():
    return {
        "message": "Welcome to Credit Risk Prediction API",
        "docs": "/docs",           # Swagger UI
        "health": "/health",
        "usage": "POST to /predict with loan data"
    }
@app.post("/predict")
def predict(loan: LoanApplication):
    return predict_risk(loan.dict())

@app.get("/health")
def health():
    return {"status": "healthy"}
from fastapi import FastAPI
from pydantic import BaseModel
from src.deployment.predict import predict_risk

app = FastAPI(title="Credit Risk API")

class LoanApplication(BaseModel):
    Duration_in_month: int
    Credit_amount: float
    Age_in_years: int
    # Add other fields matching your features...

@app.post("/predict")
def predict(loan: LoanApplication):
    return predict_risk(loan.dict())

@app.get("/health")
def health():
    return {"status": "healthy"}
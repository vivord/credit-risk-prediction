# Credit Risk Prediction ML Project

Production-ready credit risk model for banking (aligns with core banking + CBC workflows).

## Quick Start
See Setup above.

## Project Structure
(See below)

## Next Steps for Production
- Replace data with CBC/core extracts (via SQLAlchemy/PostgreSQL).
- Add DVC for data versioning, MLflow for tracking.
- Integrate with your data warehouse (star schema).
- Regulatory validation + monitoring (drift with Evidently).

Setup Instructions (From Zero)

- Create folder: credit-risk-prediction
- Copy all files below into it.
- cd credit-risk-prediction
- python -m venv venv && source venv/bin/activate (or Windows equivalent).
- pip install -r requirements.txt
- Run training: python src/train.py
- Run API: uvicorn src.deployment.app:app --reload
- Run Dashboard: set PYTHONPATH=.
- Run Dashboard then: streamlit run src/deployment/dashboard.py
- (Optional) docker build -t credit-risk . && docker run -p 8000:8000 -p 8501:8501 credit-risk
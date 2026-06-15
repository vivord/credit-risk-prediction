FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000 & streamlit run src.deployment/dashboard.py --server.port 8501"]
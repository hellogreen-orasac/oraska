FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ postgresql-client && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY oraska/ ./oraska/
COPY alembic/ ./alembic/
COPY alembic.ini .
RUN mkdir -p checkpoints
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn oraska.main:app --host 0.0.0.0 --port 8000"]


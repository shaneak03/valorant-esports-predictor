FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt api/requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

COPY api/ ./api/
COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 5000
CMD ["python", "api/app.py"]

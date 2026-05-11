FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model so it's baked into the image
# and doesn't need to download at runtime on the server
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('AITeamVN/Vietnamese_Embedding')"

# Copy source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

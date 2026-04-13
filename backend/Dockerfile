<<<<<<< HEAD
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
CMD ["gunicorn", "similarity_server:app", "--bind", "0.0.0.0:7860"]
=======
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Cloud Run uses PORT env var
ENV PORT=8080

EXPOSE 8080

# Start app with gunicorn
CMD ["gunicorn", "similarity_server:app", "--bind", "0.0.0.0:8080"]
>>>>>>> d80ee7d6c5754259e5e12a2e45bfa7cc0c30ad3c

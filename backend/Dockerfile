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

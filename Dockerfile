FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
EXPOSE 8080
CMD ["gunicorn", "similarity_server:app", "--bind", "0.0.0.0:8080"]

FROM python:3.11-slim

WORKDIR /app

# Install dependencies needed for compiling certain python packages (like psycopg2) if necessary, though psycopg2-binary usually handles it
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV PORT=8501
ENV HOST=0.0.0.0

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

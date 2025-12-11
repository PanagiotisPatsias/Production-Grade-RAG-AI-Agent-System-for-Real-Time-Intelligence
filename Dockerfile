# Lightweight Python image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps (χρειαζόμαστε αυτά για pypdf / build κ.λπ.)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (για να κάνει cache τα layers)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy όλο το project
COPY . .

# Streamlit config για Cloud Run / Docker
# Θα ακούει στην πόρτα 8080 και σε 0.0.0.0 (όχι μόνο localhost)
EXPOSE 8080

# Default command
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

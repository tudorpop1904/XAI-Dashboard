## ---- Builder stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps (only what is strictly needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# IMPORTANT: copy requirements FIRST for better caching
COPY requirements.txt .
COPY requirements-dev.txt .

# Upgrade pip once
RUN pip install --upgrade pip

# Install torch separately (rarely changes → good cache layer)
RUN pip install \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install -r requirements.txt


## ---- Runtime stage ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code LAST (maximizes Docker cache reuse)
COPY . .

# Streamlit config
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\n\n[browser]\ngatherUsageStats = false\n' > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
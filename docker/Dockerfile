## ---- Builder stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for compiling native extensions (scikit-learn, shap, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment so all packages live in one portable tree
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install torch + torchvision TOGETHER from the CPU-only index.
# This ensures their C++ ABIs match (prevents torchvision::nms errors).
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
       torch torchvision \
       --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt


## ---- Runtime stage (no build tools shipped) ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Only runtime C libs needed (OpenMP for sklearn, font rendering)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

# Streamlit config: disable telemetry, set server defaults
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\n\n[browser]\ngatherUsageStats = false\n' > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]

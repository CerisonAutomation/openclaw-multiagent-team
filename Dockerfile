# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for node+vercel CLI (optional path for `openclaw deploy`)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Install openclaw + both SDKs so any provider works out of the box
COPY pyproject.toml README.md ./
COPY openclaw ./openclaw
RUN pip install -e ".[all]"

# Optional: include Vercel CLI (uncomment to bake it in; adds ~150MB)
# RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
#     && apt-get install -y nodejs && npm i -g vercel

EXPOSE 8000
ENTRYPOINT ["openclaw"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

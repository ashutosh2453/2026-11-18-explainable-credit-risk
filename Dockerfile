FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CREDIT_DATA=/app/data/german_credit.csv CREDIT_ARTIFACT=/app/artifacts/model.joblib
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY credit_risk ./credit_risk
COPY data ./data
# Install dependencies, then train once at build time so containers start instantly.
RUN pip install --no-cache-dir . && python -m credit_risk.train --card /app/MODEL_CARD.md \
    && useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]

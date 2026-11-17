"""FastAPI service: ``uvicorn credit_risk.api:app``."""

from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .data import ApplicantError
from .service import Bundle, Scorer, load_or_train

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("credit_risk")


def create_app(bundle: Bundle | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        warnings.filterwarnings("ignore", category=UserWarning)
        app.state.scorer = Scorer(bundle or load_or_train())
        yield

    app = FastAPI(title="Explainable Credit-Risk Scorer", version=__version__, lifespan=lifespan)

    @app.exception_handler(ApplicantError)
    async def _invalid(_: Request, exc: ApplicantError):
        return JSONResponse(status_code=422, content={"detail": str(exc).split("; ")})

    def scorer(request: Request) -> Scorer:
        return request.app.state.scorer

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health(request: Request):
        m = scorer(request).b.model
        return {"status": "ok", "model": m.name, "version": __version__, "trained_at": m.trained_at}

    @app.get("/schema")
    def schema(request: Request):
        """Feature definitions, allowed values and an example applicant (drives the UI form)."""
        return scorer(request).schema()

    @app.post("/score")
    def score(request: Request, applicant: dict = Body(..., examples=[{"duration": 24, "credit_amount": 4000}]),
              recourse: bool = True):
        """Score one applicant: probability of default, decision, SHAP explanation, reason codes, recourse."""
        return scorer(request).score(applicant, with_recourse=recourse)

    @app.get("/model")
    def model(request: Request):
        """Metrics, threshold, global feature importance and fairness audit."""
        return scorer(request).model_info()

    return app


app = create_app()

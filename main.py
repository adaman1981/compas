import os
import secrets
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from fastapi.security import APIKeyHeader

app = FastAPI(title="KOMPAS Python Lab", version="0.1.0")

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)

def require_api_key(
    x_api_key: str | None = Depends(api_key_header)
) -> None:
    expected = os.getenv("KOMPAS_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured"
        )
    if not x_api_key or not secrets.compare_digest(
        x_api_key, expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

class MonteCarloRequest(BaseModel):
    simulations: int = Field(default=10000, ge=100, le=100000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    distribution: Literal["normal", "uniform"] = "normal"
    mean: float = Field(default=0, ge=-1e9, le=1e9)
    std: float = Field(default=1, gt=0, le=1e9)
    low: float = Field(default=0, ge=-1e9, le=1e9)
    high: float = Field(default=1, ge=-1e9, le=1e9)
    threshold: float = Field(default=0, ge=-1e9, le=1e9)

    @model_validator(mode="after")
    def validate_uniform_bounds(self):
        if self.distribution == "uniform" and self.low >= self.high:
            raise ValueError("low must be less than high")
        return self


@app.get("/health")
def health():
    return {"status": "ok", "service": "kompas-python-lab", "version": "0.1.0"}


@app.post("/v1/monte-carlo", dependencies=[Depends(require_api_key)])
def monte_carlo(request: MonteCarloRequest):
    rng = np.random.default_rng(request.seed)
    if request.distribution == "normal":
        samples = rng.normal(request.mean, request.std, request.simulations)
    else:
        samples = rng.uniform(request.low, request.high, request.simulations)
    quantiles = np.quantile(samples, [0.05, 0.5, 0.95])
    successes = int(np.count_nonzero(samples > request.threshold))
    p = successes / request.simulations
    se = float(np.sqrt(p * (1 - p) / request.simulations))
    return {
        "method": "monte_carlo",
        "distribution": request.distribution,
        "simulations": request.simulations,
        "seed": request.seed,
        "sample_mean": float(np.mean(samples)),
        "sample_std": float(np.std(samples, ddof=1)),
        "quantiles": {"p05": float(quantiles[0]), "p50": float(quantiles[1]), "p95": float(quantiles[2])},
        "threshold": request.threshold,
        "probability_above_threshold": p,
        "probability_standard_error": se,
        "note": "Simulation assumes the selected distribution and parameters; results are not empirical forecasts.",
    }

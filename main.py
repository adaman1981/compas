import os
import secrets
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, model_validator
from scipy import stats

app = FastAPI(title="KOMPAS Python Lab", version="0.2.0")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(x_api_key: str | None = Depends(api_key_header)) -> None:
    expected = os.getenv("KOMPAS_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="API key is not configured")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


class MonteCarloRequest(BaseModel):
    simulations: int = Field(default=10000, ge=100, le=100000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    distribution: Literal["normal", "uniform"] = "normal"
    mean: float = Field(default=0, ge=-1e9, le=1e9, allow_inf_nan=False)
    std: float = Field(default=1, gt=0, le=1e9, allow_inf_nan=False)
    low: float = Field(default=0, ge=-1e9, le=1e9, allow_inf_nan=False)
    high: float = Field(default=1, ge=-1e9, le=1e9, allow_inf_nan=False)
    threshold: float = Field(default=0, ge=-1e9, le=1e9, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_uniform_bounds(self):
        if self.distribution == "uniform" and self.low >= self.high:
            raise ValueError("low must be less than high")
        return self


class BinomialRequest(BaseModel):
    trials: int = Field(ge=1, le=100000)
    success_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    threshold: int = Field(ge=0, le=100000)


class SummaryRequest(BaseModel):
    values: list[float] = Field(min_length=2, max_length=10000)

    @model_validator(mode="after")
    def check_values(self):
        if any(not np.isfinite(v) or abs(v) > 1e9 for v in self.values):
            raise ValueError("All values must be finite and between -1e9 and 1e9")
        return self


class TTestRequest(SummaryRequest):
    hypothesized_mean: float = Field(default=0, ge=-1e9, le=1e9, allow_inf_nan=False)
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1, allow_inf_nan=False)


@app.get("/health")
def health():
    return {"status": "ok", "service": "kompas-python-lab", "version": "0.2.0"}


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
        "method": "monte_carlo", "distribution": request.distribution,
        "simulations": request.simulations, "seed": request.seed,
        "sample_mean": float(np.mean(samples)),
        "sample_std": float(np.std(samples, ddof=1)),
        "quantiles": {"p05": float(quantiles[0]), "p50": float(quantiles[1]), "p95": float(quantiles[2])},
        "threshold": request.threshold,
        "probability_above_threshold": p,
        "probability_standard_error": se,
        "note": "Simulation assumes the selected distribution and parameters; results are not empirical forecasts.",
    }


@app.post("/v1/probability/binomial", dependencies=[Depends(require_api_key)])
def binomial(request: BinomialRequest):
    n, p, k = request.trials, request.success_probability, request.threshold
    return {
        "method": "exact_binomial", "trials": n, "success_probability": p, "threshold": k,
        "probability_exactly_threshold": float(stats.binom.pmf(k, n, p)),
        "probability_at_least_threshold": float(stats.binom.sf(k - 1, n, p)),
        "probability_at_most_threshold": float(stats.binom.cdf(k, n, p)),
        "expected_successes": n * p,
        "standard_deviation": float(np.sqrt(n * p * (1 - p))),
        "note": "Assumes independent Bernoulli trials with identical success probability.",
    }


@app.post("/v1/statistics/summary", dependencies=[Depends(require_api_key)])
def summary(request: SummaryRequest):
    a = np.asarray(request.values, dtype=float)
    q = np.quantile(a, [0.25, 0.5, 0.75])
    return {
        "method": "descriptive_statistics", "count": len(a),
        "mean": float(np.mean(a)), "median": float(q[1]),
        "sample_std": float(np.std(a, ddof=1)),
        "sample_variance": float(np.var(a, ddof=1)),
        "minimum": float(np.min(a)), "maximum": float(np.max(a)),
        "quartiles": {"q25": float(q[0]), "q50": float(q[1]), "q75": float(q[2])},
        "note": "Describes supplied observations; does not establish causation or forecast future values.",
    }


@app.post("/v1/statistics/one-sample-t-test", dependencies=[Depends(require_api_key)])
def one_sample_t_test(request: TTestRequest):
    a = np.asarray(request.values, dtype=float)
    if np.ptp(a) == 0:
        raise HTTPException(status_code=422, detail="T-test requires observations with nonzero variability")
    result = stats.ttest_1samp(a, popmean=request.hypothesized_mean, alternative=request.alternative)
    if not np.isfinite(result.statistic) or not np.isfinite(result.pvalue):
        raise HTTPException(status_code=422, detail="Numerically unstable t-test input")
    ci = result.confidence_interval(confidence_level=request.confidence_level)
    return {
        "method": "one_sample_t_test", "count": len(a),
        "sample_mean": float(np.mean(a)), "hypothesized_mean": request.hypothesized_mean,
        "alternative": request.alternative, "t_statistic": float(result.statistic),
        "degrees_of_freedom": len(a) - 1, "p_value": float(result.pvalue),
        "confidence_level": request.confidence_level,
        "confidence_interval": {
            "lower": float(ci.low) if np.isfinite(ci.low) else None,
            "upper": float(ci.high) if np.isfinite(ci.high) else None,
        },
        "note": "One-sample Student t-test assumes independent observations and a suitable sampling model; p-value is not the probability that the null hypothesis is true.",
    }

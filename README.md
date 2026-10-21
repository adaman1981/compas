# ALIFE · KOMPAS Python Lab v0.2.0

A protected numerical microservice for KOMPAS. Endpoints:

- `GET /health` (public)
- `POST /v1/monte-carlo` (unchanged from v0.1.0)
- `POST /v1/probability/binomial`
- `POST /v1/statistics/summary`
- `POST /v1/statistics/one-sample-t-test`

All POST endpoints require `X-API-Key`, configured through Render's `KOMPAS_API_KEY` environment variable. Never commit the key.

Render build: `pip install -r requirements.txt`
Render start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Open `/docs`, click Authorize, and enter the API key for interactive tests. Documentation is public; computation requires authorization.

Sample binomial request:
```json
{"trials":10,"success_probability":0.5,"threshold":7}
```

Sample descriptive statistics request:
```json
{"values":[10,12,14,16,18]}
```

Sample t-test request:
```json
{"values":[11,12,13,14,15],"hypothesized_mean":10,"alternative":"greater","confidence_level":0.95}
```

Only statistical calculations are implemented here. Forecasting, Bayesian inference, optimization and game theory are planned, not yet implemented.

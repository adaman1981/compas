# KOMPAS Python Lab

Independent numerical API for the experimental KOMPAS orchestrator.

## Render Web Service settings

- **Language:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment variable:** `KOMPAS_API_KEY` — a long random secret; set it **only** in Render Environment, never commit it to GitHub.

Use an available plan you are comfortable with. A free plan, if offered, may sleep after inactivity.

## Health check

`GET https://YOUR-SERVICE.onrender.com/health`

Expected: `{"status":"ok","service":"kompas-python-lab","version":"0.1.0"}`

## Protected calculation

`POST /v1/monte-carlo`

Header: `X-API-Key: YOUR_SECRET`

JSON body:

```json
{
  "simulations": 10000,
  "seed": 42,
  "distribution": "normal",
  "mean": 100,
  "std": 15,
  "threshold": 120
}
```

Returns simulation statistics, quantiles and estimated probability above threshold.

Security: no arbitrary Python code execution, strict request validation, bounded simulations, secret API key.

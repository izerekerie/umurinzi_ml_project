# Umurinzi — Deployment plan

This document describes how Umurinzi is intended to be deployed. The target
platform is **Render** with **Docker**. The app is not yet deployed; the
sections below set out the architecture, rationale, and cost of the plan.

---

## 1. Architecture at a glance

```
┌─────────────────────────────────┐
│  Browser (citizen/manager/admin)│
└────────────────┬────────────────┘
                 │ HTTPS
                 ▼
┌─────────────────────────────────┐
│  Render web service (Docker)    │
│  ├── gunicorn workers           │
│  │   └── app_cadastral.py       │
│  │       ├── /citizen flow      │
│  │       ├── /manager flow      │
│  │       ├── /admin flow        │
│  │       └── /apidocs (Swagger) │
│  └── SQLite USERS + ALTERNATIVES │
└─────────────────────────────────┘
                 │
                 │ At request time none of the below are called.
                 │ Everything is precomputed at build time.
                 ▼
                  x  Google Earth Engine  (only at training time)
                  x  Hansen GFC            (only at training time)
                  x  External APIs         (no runtime dependency)
```

**Key property:** zero runtime external dependencies. The container ships
with the trained model, sectors, and Hansen-derived `sector_risk.json`
pre-loaded. This makes deployments simple, cheap, and reliable.

---

## 2. Why Render + Docker

| Requirement | How Render + Docker meets it |
|---|---|
| Auto-deploy on `git push` | Render listens to GitHub, rebuilds on every commit |
| Reproducible image | `Dockerfile` pins all system + Python deps |
| ≥ 1 GB RAM (boots at ~600 MB) | Render's Standard tier provides 2 GB |
| HTTPS + custom domain | Free on every tier |
| Logs / metrics | Native log streaming + basic CPU/mem charts |
| OCR-friendly timeout | gunicorn timeout 90 s covers Bugesera-scale PDFs |

---

## 3. Cost estimate

| Component | Plan | Cost / month |
|---|---|---|
| Render web service | **Standard** (2 GB RAM, always-on) | **USD 25** |
| Render web service | Alternative: Starter (512 MB, may OOM on boot) | USD 7 |
| Render web service | Alternative: Free (cold starts, 512 MB) | USD 0 |
| GitHub repo + Actions | Free academic tier | USD 0 |
| Docker Hub | Free for public images | USD 0 |
| Custom domain `umurinzi.rw` | `.rw` registrar via RICTA | ~USD 1 / month amortised (USD 12 / yr) |
| **Total — production demo** |   | **~USD 26 / month** |
| **Total — free tier only** |   | **USD 0** |

> A Flask app that loads a 47 MB ML model + 416 sector polygons boots at
> ~600 MB, which exceeds the typical free 512 MB limit. The honest options
> are either USD 0 with documented cold-start demos, or USD 25–26 / month
> for an always-on production demo.

---

## 4. Cheaper alternatives if the budget is tight

| Platform | Cost | Catch |
|---|---|---|
| **Fly.io** — Hobby VM, 1 GB | ~USD 3–5 / month | Smaller community; CLI-first; no auto-deploy GUI |
| **Railway** | $5 / month credit free | Free credit runs out fast with always-on |
| **Render Free + cold starts** | USD 0 | ~30-second cold start on first request after 15 min idle |
| **DigitalOcean Droplet** | USD 6 / month | Manual ops (SSH, systemd, certificates) |
| **Hetzner CPX11** | EUR 4 / month | Cheapest VM; same manual ops as DO |

For a low-cost demo, **Render Free** is sufficient (USD 0) with the cold-start
acknowledged. Render Standard ($25/mo) is the move if a citizen-facing pilot
ever launches.

---

## 5. Local Docker testing before pushing

```bash
# Build the image
docker build -t umurinzi:latest .

# Run it
docker run -p 5050:5050 -e PORT=5050 umurinzi:latest

# Open http://localhost:5050 in your browser — same as production
```

Image size: ~1.2 GB (Python 3.13 slim + tesseract + opencv + sklearn).
Build time: ~6 minutes first time, ~30 seconds with layer cache.

---

## 6. Files involved in deployment

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the production container image |
| `.dockerignore` | Keeps the build context small (skips notebooks/, drafts) |
| `render.yaml` | Render Infrastructure-as-Code blueprint |
| `Procfile` | Alt start command for Render buildpack / Heroku-style platforms |
| `requirements.txt` | Python deps including `gunicorn` |

All of these are tracked in Git; the deployment is reproducible from a
clean clone.

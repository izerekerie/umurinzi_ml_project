# Umurinzi — Rwanda Forest Risk Intelligence

> *Open-data satellite monitoring and localised risk assessment to help
> Rwandan citizens protect forest **before** cutting.*

---

## 1 · Description

Umurinzi (Kinyarwanda for *guardian / protector*) is a BSc Software
Engineering capstone (African Leadership University) that addresses the
core gap in Rwanda's smallholder deforestation: most clearings are below
the 1-hectare detection threshold of published global forest monitoring,
so they're invisible until after the trees are gone.

The system combines:

- A **Random Forest classifier** trained on 10,000 labelled pixels from
  the Nyungwe buffer zone using Sentinel-2 (optical) + Sentinel-1 (radar)
  + SRTM (terrain) features, with Hansen Global Forest Change labels.
- A **Flask web application** with three personas — Citizen, Forest
  Manager, Admin — each scoped to its real workflow.
- An **OpenAPI-documented REST backend** exposing 16 endpoints, browsable
  through interactive Swagger UI at `/apidocs`.

**Headline result**: F1 = 0.791 on a held-out 2,000-pixel test set,
beating the published global baseline (Ygorra et al. 2024, F1 = 0.71) by
**+0.08**. Recall stays in the 80–83 % band even at the 0.1–0.2 ha
smallholder patch size where global models typically degrade.

The system answers four research questions:

| RQ | Question | Status |
|---|---|---|
| RQ1 | Optimal combination of S2 / S1 / SRTM features? | ✅ Answered — `results/experiments/rq1_writeup.md` |
| RQ2 | Accuracy degradation at smallholder patch sizes? | ✅ Answered — `results/patch_size_analysis/` |
| RQ3 | Does 500 m neighbourhood improve over parcel-only analysis? | 🟡 Implemented in app; writeup pending |
| RQ4 | Out-of-sample validation across districts? | 🟡 Pending — needs RNLA real-coordinate sample |

---

## 2 · Repository

| Resource | URL |
|---|---|
| **GitHub repo** | https://github.com/izerekerie/umurinzi_ml_project |
| **Demo video** | https://youtu.be/L10J9Ie8IDE?si=BFBF2ZC2SGKSbF63 |
| Live demo URL | https://umurinzi-web.onrender.com *(once deployed; see DEPLOYMENT.md)* |
| Swagger UI | `https://umurinzi-web.onrender.com/apidocs` |
| Dissertation prose | `results/experiments/rq1_writeup.md` |

To clone:

```bash
git clone https://github.com/izerekerie/umurinzi_ml_project.git
cd umurinzi_ml_project
```

---

## 3 · Environment & project setup

### Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | 3.11 or 3.13 | Project tested on 3.13 |
| Tesseract OCR | 5.x | Citizen cadastral upload reads printed labels |
| Poppler | 23.x+ | PDF text extraction (pdfplumber) |
| Git | any | Cloning |
| Docker (optional) | 24+ | Reproducible deploy; see DEPLOYMENT.md |

#### macOS install

```bash
brew install python@3.13 tesseract poppler git
```

#### Ubuntu / Debian install

```bash
sudo apt update
sudo apt install -y python3.13 python3.13-venv \
                    tesseract-ocr poppler-utils \
                    libgl1 libglib2.0-0 git
```

### Project setup (4 commands)

```bash
# 1. Create + activate a virtualenv
python3.13 -m venv .venv
source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Seed the SQLite database with bcrypt-hashed demo accounts
python scripts/seed_users.py

# 4. Run the Flask app
python app_cadastral.py
```

Then open **http://localhost:5050**.

### Demo accounts

| Role | Email | Password | Sees |
|---|---|---|---|
| Admin | `admin@treesight.rw` | `admin` | Everything (all 416 sectors + user management) |
| Forest Manager | `manager.nyamasheke@treesight.rw` | `nyamasheke` | Nyamasheke district sectors only |
| Forest Manager | `manager.rusizi@treesight.rw` | `rusizi` | Rusizi district sectors only |
| Forest Manager | `manager.nyaruguru@treesight.rw` | `nyaruguru` | Nyaruguru district sectors only |

### Folder structure (each folder has ONE purpose)

```
umurinzi/
├── data/
│   ├── raw/          GEE exports (training_data.csv) + sample cadastral PDFs
│   ├── processed/    Cleaned training data
│   ├── geo/          Sector polygons + Hansen rasters
│   └── database/     SQLite + seed SQL
├── notebooks/        GEE script + 6 Jupyter notebooks
├── scripts/          7 reproducible pipeline scripts
├── models/           4 trained Random Forest models (rf_A..D.pkl)
├── results/
│   ├── eda/                       5 exploratory data analysis figures
│   ├── experiments/                4-experiment comparison (RQ1)
│   ├── hyperparameter_tuning/      96-combo grid search outputs
│   ├── metrics/                    F1, confusion matrix, audit JSON
│   ├── patch_size_analysis/        RQ2 figure + CSV
│   └── application/                Precomputed sector_risk.json
├── app_cadastral.py               Flask web app entry point
├── Dockerfile                      Production container (Render-ready)
├── render.yaml                     Render Infrastructure-as-Code
├── requirements.txt
├── README.md                       This file
└── DEPLOYMENT.md                   Step-by-step deploy guide
```

### Reproducibility chain

Running these scripts in order rebuilds the whole pipeline from the raw
GEE export onwards:

```bash
# Data prep (notebooks)
# notebooks/01_GEE_Export.js       runs in the GEE code editor
jupyter nbconvert --to notebook --execute notebooks/02_Clean_Data.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_Train_Model.ipynb

# Reproducible analysis scripts
python scripts/eda_visualisations.py            # → results/eda/*.png
python scripts/hyperparameter_tune.py            # → results/hyperparameter_tuning/
python scripts/evaluate_split_and_patchsize.py   # → results/metrics/ + patch_size_analysis/
python scripts/rq1_writeup.py                    # → results/experiments/rq1_*
python scripts/precompute_sector_risk.py         # → results/application/sector_risk.json
```

---

## 4 · Designs

### 4.1 Figma mockups

The visual design system was prototyped in Figma before any HTML was
written. The mockups cover all five user-facing views — landing, login,
citizen, forest manager, and admin.

**Figma file:** https://www.figma.com/design/mYF9We3btINQNbOsiuRl5I/Umurinzi?node-id=0-1

Design system in use (replicated 1:1 in the Flask templates):

```
Primary brand        #14532d   (forest green)
Hover                #166534
Background           #f7f8f5
Card background      #ffffff
Risk HIGH            #dc2626
Risk MEDIUM          #ea580c
Risk LOW             #16a34a
Muted text           #6b7280
```

### 4.2 Architecture diagrams

Five architecture diagrams documented in Chapter 3 of the dissertation:

| Fig | Description | File reference |
|---|---|---|
| 3.1 | Two-pipeline system architecture (offline training + online inference) | Chapter 3 §3.X |
| 3.2 | Use-case diagram (Citizen, Forest Manager, Admin) | Chapter 3 §3.X |
| 3.3 | Data-pipeline sequence (GEE → labels → features → model) | Chapter 3 §3.X |
| 3.4 | API request sequence (citizen upload → OCR → analysis) | Chapter 3 §3.X |
| 3.5 | Entity-Relationship Diagram (USERS / SECTORS / PARCEL_ANALYSES / ALTERNATIVES) | Chapter 3 §3.X |

Insert the exported diagram images below (from the Chapter 3 dissertation):

<!-- Fig 3.1 — e.g. ![System architecture](results/diagrams/fig_3_1_architecture.png) -->
<!-- Fig 3.2 — Use-case diagram -->
<!-- Fig 3.3 — Data-pipeline sequence -->
<!-- Fig 3.4 — API request sequence -->
<!-- Fig 3.5 — Entity-Relationship Diagram -->

These are not circuit diagrams (Umurinzi is a software-only project — no
hardware sensors), so the "circuit diagram" rubric line is interpreted as
the **data-flow + ERD** combination above.

### 4.3 App interface

A full walkthrough of all five views (landing, login, citizen, forest
manager, admin) is in the **[demo video](https://youtu.be/L10J9Ie8IDE?si=BFBF2ZC2SGKSbF63)**.

Screenshots:

<!-- insert screenshots here -->
<img width="1512" height="824" alt="Screenshot 2026-06-12 at 20 49 47" src="https://github.com/user-attachments/assets/09987a0a-ac6f-42ad-9f56-c89a53809d7e" />
<img width="1512" height="824" alt="Screenshot 2026-06-12 at 20 53 27" src="https://github.com/user-attachments/assets/1f696abb-0e76-472c-ae82-f055e2c62bef" />
<img width="1512" height="824" alt="Screenshot 2026-06-12 at 20 53 33" src="https://github.com/user-attachments/assets/0bba2a6d-a566-4934-8df0-8288f27bbccb" />
<img width="1512" height="824" alt="Screenshot 2026-06-12 at 20 53 49" src="https://github.com/user-attachments/assets/af883271-8fc1-4486-b995-a38ef8f8beb3" />
<img width="1512" height="824" alt="Screenshot 2026-06-12 at 20 53 56" src="https://github.com/user-attachments/assets/9f0375ba-74f9-4158-b37f-90019998b0a7" />
<img width="1512" height="824" alt="Screenshot 2026-06-12 at 20 54 08" src="https://github.com/user-attachments/assets/cae91233-e30d-4509-8c79-e64e7d3f8b11" />
<img width="1512" height="824" alt="Screenshot 2026-06-12 at 20 54 28" src="https://github.com/user-attachments/assets/c3686af8-8fd0-493f-8482-07453af97a87" />


---

## 5 · Deployment plan

Deployed on **Render** from the included `render.yaml` blueprint (Docker,
gunicorn × 4 workers, Frankfurt region — closest to Rwanda).

**Live URL:** https://umurinzi-web.onrender.com

Deploy steps:

1. Push to GitHub.
2. Render → **New → Blueprint** → connect this repo.
3. Render reads `render.yaml` and builds the service (~6–8 min first build).
4. Set the `SECRET_KEY` environment variable before going public.

Local Docker test:

```bash
docker build -t umurinzi .
docker run -p 5050:5050 -e PORT=5050 umurinzi
# → http://localhost:5050
```

Run locally **without Docker** (plain Python — see §3 for prerequisites):

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/seed_users.py       # one-time: seed demo accounts
python app_cadastral.py
# → http://localhost:5050
```

Full cost breakdown is in **`DEPLOYMENT.md`**.

---

## License

This project is part of an undergraduate research deliverable at
African Leadership University. All third-party tools used are
open-source (MIT, BSD, or Apache 2.0). Satellite imagery is provided
under the European Space Agency, USGS, and University of Maryland open
data licences for academic use.

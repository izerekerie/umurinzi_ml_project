"""Umurinzi — Cadastral Plan Extractor (browser demo)

Run:    .venv/bin/python app_cadastral.py
Open:   http://localhost:5050/

Upload a Rwanda land-title PDF or photo, see the extracted location on a map.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from functools import wraps
from pathlib import Path

import bcrypt

from flask import (Flask, jsonify, redirect, render_template,
                   render_template_string, request, session, url_for)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from extract_cadastral import extract  # type: ignore

# ── Lazy-loaded analysis dependencies (heavy imports done at server boot) ──
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FEATURE_COLS = ['EVI_train', 'NBR_train', 'NDVI_change', 'NDVI_test', 'NDVI_train',
                'NIR_train', 'RED_train', 'SWIR_test', 'SWIR_train', 'VH_VV_ratio',
                'VH_test', 'VH_train', 'VV_test', 'VV_train', 'aspect', 'elevation',
                'slope']

print("[boot] loading trained Random Forest model rf_D.pkl …")
_MODEL = pickle.load(open(Path(__file__).parent / "models" / "rf_D.pkl", "rb"))
print(f"[boot]   model has {_MODEL.n_features_in_} features, classes={list(_MODEL.classes_)}")

print("[boot] loading training data with geometry …")
_train_raw = pd.read_csv(Path(__file__).parent / "data" / "raw" / "training_data.csv")
_geos = _train_raw['.geo'].apply(lambda s: json.loads(s)['coordinates'])
_train_raw['lng'] = _geos.apply(lambda c: c[0])
_train_raw['lat'] = _geos.apply(lambda c: c[1])
# Build a KD-tree on (lat,lng) for fast nearest-neighbour lookup. Spherical
# distance approximation is fine inside one country; pyproj projection would
# be more accurate but slower for per-request use.
_TREE = cKDTree(_train_raw[['lat', 'lng']].values)
print(f"[boot]   training set: {len(_train_raw):,} pixels, classes={_train_raw['label'].value_counts().to_dict()}")

# ── CLEARED_NDVI reference for forward simulation ─────────────────────
# Literature-based value for freshly cleared land in tropical deforestation
# (bare soil + light residual stubble, NDVI ≈ 0.15–0.25 — Pettorelli 2013,
# Mugabowindekwe et al. 2024). We deliberately do NOT use the training-set
# median for class=1 pixels here because:
#   (a) Hansen 2020-22 "loss" pixels often partially re-vegetated by the
#       2023-24 test period (regrowth, planted crops) — median is ≈ 0.70,
#       which would make simulation incorrectly RAISE NDVI after a cut;
#   (b) Cloud Score+ median compositing smooths the immediate post-cut
#       signal anyway.
# Using 0.20 represents the worst-case post-cut NDVI a citizen would
# observe in the satellite imagery within the year after clearing.
_CLEARED_NDVI = 0.20
print(f"[boot]   CLEARED_NDVI reference = {_CLEARED_NDVI:.2f} (literature default)")

print("[boot] loading 416 sector polygons for click-to-analyse …")
import geopandas as gpd
_SECTORS = gpd.read_file(Path(__file__).parent / "data" / "geo" / "sectors_wgs84.geojson")
# Pre-compute centroid lat/lng — used as the analysis input when a manager
# clicks a sector on the choropleth.
# Compute centroids in a projected CRS (UTM-35S, EPSG:32735) so they're
# spatially correct, then project back to WGS-84 for storage. This avoids
# the GeoPandas "Geometry is in a geographic CRS" warning and gives accurate
# sector centres even near the equator where curvature still distorts.
_centroids_wgs = _SECTORS.geometry.to_crs(32735).centroid.to_crs(4326)
_SECTORS["centroid_lat"] = _centroids_wgs.y
_SECTORS["centroid_lng"] = _centroids_wgs.x
print(f"[boot]   {len(_SECTORS)} sectors loaded with centroids")


def find_nearest_pixels(lat: float, lng: float, k: int = 25):
    """K-nearest training pixels + the distance to the closest one (in km).
    The distance is the confidence signal: small = in-domain (Nyungwe), large
    = out-of-distribution (the model's prediction is less trustworthy)."""
    distances, idx = _TREE.query([lat, lng], k=k)
    # The KDTree was built on (lat, lng) so distances are in degrees.
    # Convert the nearest-neighbour distance to kilometres on a sphere.
    # 1° latitude ≈ 111 km; for Rwanda longitudes (≈ 30°E), 1° lng ≈ 111 × cos(2°) ≈ 110.9 km.
    deg_to_km = 111.0
    nearest_km = float(distances[0] * deg_to_km) if hasattr(distances, "__iter__") else float(distances * deg_to_km)
    return _train_raw.iloc[idx], nearest_km


def analyse_parcel(lat: float, lng: float, area_ha: float = None) -> dict:
    """Run the full Umurinzi analysis for one parcel.

    The model RUNS on any Rwanda location, but its prediction is only
    well-calibrated inside the Nyungwe training domain. We therefore return a
    `confidence` field based on distance from the nearest training pixel:

        ≤ 5 km   → HIGH   (in-domain)
        ≤ 50 km  → MEDIUM (near-domain, Western/Southern Rwanda)
        > 50 km  → LOW    (out-of-domain — surface result only)

    The 3-rule classifier still fires for every query so citizens get a
    HIGH/MEDIUM/LOW result everywhere, but the dissertation and UI
    communicate the calibration scope honestly.
    """
    nbrs, nearest_km = find_nearest_pixels(lat, lng, k=25)
    feats = nbrs[FEATURE_COLS].median().values.reshape(1, -1)
    prob = float(_MODEL.predict_proba(feats)[0][1])

    ndvi_current = float(nbrs['NDVI_test'].median())
    ndvi_train_avg = float(nbrs['NDVI_train'].median())
    ndvi_change = float(nbrs['NDVI_change'].median())
    tree_cover_pct = max(0.0, min(100.0, ndvi_current * 100.0))

    # 500-metre neighbourhood: K-nearest as a spatial proxy
    deforested_pct_500m = float(nbrs['label'].mean() * 100)
    avg_ndvi_500m = float(nbrs['NDVI_test'].mean())

    # ── 3-rule Risk Classifier ─────────────────────────────────────
    rule1_high = (prob > 0.65) or (tree_cover_pct < 30)
    rule2_high = (deforested_pct_500m > 50) and (ndvi_current < avg_ndvi_500m * 0.70)
    rule3_med  = (0.35 < prob <= 0.65) and (deforested_pct_500m > 0)
    if rule1_high or rule2_high:
        risk_level = 'HIGH'
        fired_rule = 'Rule 1 (parcel)' if rule1_high else 'Rule 2 (neighbourhood)'
    elif rule3_med:
        risk_level = 'MEDIUM'
        fired_rule = 'Rule 3 (intermediate)'
    else:
        risk_level = 'LOW'
        fired_rule = 'default'

    # Training-domain confidence based on KD-tree distance to nearest sample
    if nearest_km <= 5:
        confidence = 'HIGH'
        confidence_note = 'In training domain (Nyungwe buffer zone)'
    elif nearest_km <= 50:
        confidence = 'MEDIUM'
        confidence_note = f'Near training domain ({nearest_km:.0f} km from nearest sample)'
    else:
        confidence = 'LOW'
        confidence_note = (f'Outside training domain ({nearest_km:.0f} km from nearest '
                           f'Nyungwe sample) — production deployment should query GEE live')

    return {
        'risk_level':           risk_level,
        'rule_fired':           fired_rule,
        'deforestation_prob':   round(prob, 3),
        'ndvi_current':         round(ndvi_current, 3),
        'ndvi_2020':            round(ndvi_train_avg, 3),
        'ndvi_change':          round(ndvi_change, 3),
        'tree_cover_pct':       round(tree_cover_pct, 1),
        'neighbourhood_500m_avg_ndvi':      round(avg_ndvi_500m, 3),
        'neighbourhood_500m_deforested_pct': round(deforested_pct_500m, 1),
        'parcel_area_ha':       area_ha,
        'analysis_id':          abs(hash((round(lat, 5), round(lng, 5)))) % 10_000_000,
        'confidence':           confidence,
        'confidence_note':      confidence_note,
        'km_from_training':     round(nearest_km, 1),
    }


# Forest-manager / admin accounts live in the same SQLite DB as alternatives.
_USERS_DB = Path(__file__).parent / "data" / "database" / "umurinzi.db"


def _users_conn():
    con = sqlite3.connect(str(_USERS_DB))
    con.row_factory = sqlite3.Row
    return con


def lookup_user(email: str):
    """Find a USERS row by email (case-insensitive). Returns dict or None."""
    if not email:
        return None
    with _users_conn() as con:
        row = con.execute(
            "SELECT * FROM USERS WHERE LOWER(email) = LOWER(?) AND is_active = 1",
            (email.strip(),)
        ).fetchone()
    return dict(row) if row else None


def verify_password(plain: str, stored_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), stored_hash.encode())
    except Exception:
        return False


def record_last_login(email: str):
    with _users_conn() as con:
        con.execute(
            "UPDATE USERS SET last_login = CURRENT_TIMESTAMP "
            "WHERE LOWER(email) = LOWER(?)", (email,)
        )
        con.commit()


def current_user():
    """Fresh DB read for every request — avoids stale-cache surprises."""
    email = session.get("user_email")
    return lookup_user(email) if email else None


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user():
            return redirect(url_for("login_page"))
        return fn(*args, **kwargs)
    return wrapper


app = Flask(__name__)
# Secret key for signing the login session cookie. Fixed dev value for the
# capstone demo; set UMURINZI_SECRET in the environment for any real deployment.
app.secret_key = os.environ.get("UMURINZI_SECRET", "umurinzi-dev-secret-change-me")
# Allow up to 16 MB uploads (covers any phone photo + PDF)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ── Interactive API documentation (Swagger UI) ─────────────────────────
# Browse and try every endpoint live at  http://localhost:5050/apidocs/
# The raw OpenAPI 2.0 spec is served at   http://localhost:5050/apispec_1.json
from flasgger import Swagger

app.config["SWAGGER"] = {"title": "Umurinzi API", "uiversion": 3}
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "Umurinzi Rwanda — Deforestation Risk API",
        "description": (
            "Backend for the Umurinzi MVP. Given a land parcel (from an uploaded "
            "land-title PDF/photo, manual coordinates, or a lat/lng), it predicts "
            "deforestation risk with a tuned Random Forest (Experiment D, F1≈0.79), "
            "forward-simulates a proposed cut, and returns vetted alternatives. "
            "Final-year capstone — Umurinzi Rwanda, ALU."
        ),
        "version": "1.0.0",
        "contact": {"email": "twagirinno@gmail.com"},
    },
    "tags": [
        {"name": "Parcel input", "description": "Turn a document or coordinates into a location"},
        {"name": "Risk analysis", "description": "Predict and simulate deforestation risk"},
        {"name": "Guidance", "description": "Alternatives and sector-level dashboards"},
    ],
})


@app.get("/")
def landing():
    return render_template("landing.html")


@app.get("/citizen")
def citizen():
    return render_template("citizen.html")


@app.get("/login")
def login_page():
    return render_template("login.html")


@app.post("/api/login")
def api_login():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    pwd   = data.get("password") or ""
    user  = lookup_user(email)
    if not user or not verify_password(pwd, user["password_hash"]):
        return jsonify({"error": "Invalid email or password"}), 401
    session["user_email"] = email
    record_last_login(email)
    return jsonify({
        "ok": True,
        "email": email,
        "user": {
            "full_name":      user["full_name"],
            "role":           user["role"],
            "district_scope": user["district_scope"],
            "organisation":   user["organisation"],
            "language":       user["language"],
            "last_login":     user.get("last_login"),
        },
    })


@app.post("/api/logout")
def api_logout():
    session.pop("user_email", None)
    return jsonify({"ok": True})


@app.get("/api/me")
def api_me():
    user = current_user()
    if not user:
        return jsonify({"authenticated": False})
    safe = {k: v for k, v in user.items() if k not in ("password_hash",)}
    return jsonify({
        "authenticated": True,
        "email": session.get("user_email"),
        "user": safe,
    })


def _require_admin():
    """Tiny helper for admin-only endpoints. Returns None on success, or a
    Flask response tuple on failure so handlers can `return early`."""
    user = current_user()
    if not user:
        return jsonify({"error": "Authentication required"}), 401
    if user["role"] != "admin":
        return jsonify({"error": "Admin only"}), 403
    return None


@app.get("/admin")
@login_required
def admin_page():
    if current_user()["role"] != "admin":
        return redirect(url_for("login_page"))
    return render_template("admin.html")


@app.get("/api/users")
@login_required
def api_users():
    """List all USERS — admin only."""
    err = _require_admin()
    if err is not None:
        return err
    with _users_conn() as con:
        rows = con.execute(
            "SELECT user_id, email, full_name, role, organisation, "
            "district_scope, language, created_at, last_login, is_active "
            "FROM USERS ORDER BY user_id"
        ).fetchall()
    return jsonify({"users": [dict(r) for r in rows]})


@app.post("/api/users")
@login_required
def api_users_create():
    """Create a new USER — admin only.
    Body: { email, password, full_name, role, district_scope?, organisation?, language? }
    """
    err = _require_admin()
    if err is not None:
        return err
    data = request.get_json() or {}
    required = ["email", "password", "full_name", "role"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"missing fields: {', '.join(missing)}"}), 400
    if data["role"] not in ("admin", "forest_manager"):
        return jsonify({"error": "role must be admin or forest_manager"}), 400
    if data["role"] == "forest_manager" and not data.get("district_scope"):
        return jsonify({"error": "forest_manager requires district_scope"}), 400
    if data.get("language") and data["language"] not in ("rw", "en", "fr"):
        return jsonify({"error": "language must be rw, en, or fr"}), 400

    pw_hash = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt(rounds=10)).decode()
    try:
        with _users_conn() as con:
            cur = con.execute(
                "INSERT INTO USERS (email, password_hash, full_name, role, "
                "organisation, district_scope, language) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (data["email"].lower().strip(), pw_hash, data["full_name"],
                 data["role"], data.get("organisation"),
                 data.get("district_scope") if data["role"] == "forest_manager" else None,
                 data.get("language", "en"))
            )
            con.commit()
            new_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"error": "email already exists"}), 409
    return jsonify({"ok": True, "user_id": new_id})


@app.patch("/api/users/<int:user_id>")
@login_required
def api_users_update(user_id):
    """Update an existing USER (role, scope, organisation, language, is_active) —
    admin only. Use the password endpoint to change passwords."""
    err = _require_admin()
    if err is not None:
        return err
    data = request.get_json() or {}
    fields = {}
    for f in ("full_name", "role", "organisation", "district_scope",
              "language", "is_active"):
        if f in data:
            fields[f] = data[f]
    if not fields:
        return jsonify({"error": "no editable fields supplied"}), 400
    if "role" in fields and fields["role"] not in ("admin", "forest_manager"):
        return jsonify({"error": "role must be admin or forest_manager"}), 400
    if "language" in fields and fields["language"] not in ("rw", "en", "fr"):
        return jsonify({"error": "language must be rw, en, or fr"}), 400
    if "is_active" in fields:
        fields["is_active"] = 1 if fields["is_active"] else 0

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    params = list(fields.values()) + [user_id]
    with _users_conn() as con:
        cur = con.execute(f"UPDATE USERS SET {set_clause} WHERE user_id = ?", params)
        if cur.rowcount == 0:
            return jsonify({"error": f"user {user_id} not found"}), 404
        con.commit()
    return jsonify({"ok": True, "updated_fields": list(fields.keys())})


@app.post("/api/users/<int:user_id>/password")
@login_required
def api_users_password(user_id):
    """Admin sets a new password for a user. Body: { password: str }"""
    err = _require_admin()
    if err is not None:
        return err
    data = request.get_json() or {}
    new_pwd = data.get("password") or ""
    if len(new_pwd) < 4:
        return jsonify({"error": "password must be at least 4 characters"}), 400
    pw_hash = bcrypt.hashpw(new_pwd.encode(), bcrypt.gensalt(rounds=10)).decode()
    with _users_conn() as con:
        cur = con.execute(
            "UPDATE USERS SET password_hash = ? WHERE user_id = ?",
            (pw_hash, user_id)
        )
        if cur.rowcount == 0:
            return jsonify({"error": f"user {user_id} not found"}), 404
        con.commit()
    return jsonify({"ok": True})


@app.delete("/api/users/<int:user_id>")
@login_required
def api_users_delete(user_id):
    """Soft-delete a user (is_active=0). We never hard-delete because
    PARCEL_ANALYSES rows may reference this user_id."""
    err = _require_admin()
    if err is not None:
        return err
    if current_user()["user_id"] == user_id:
        return jsonify({"error": "cannot disable your own admin account"}), 400
    with _users_conn() as con:
        cur = con.execute(
            "UPDATE USERS SET is_active = 0 WHERE user_id = ?", (user_id,)
        )
        if cur.rowcount == 0:
            return jsonify({"error": f"user {user_id} not found"}), 404
        con.commit()
    return jsonify({"ok": True})


@app.get("/manager")
@login_required
def manager():
    return render_template("manager.html")



@app.post("/api/analyse-sector")
@login_required
def api_analyse_sector():
    """Run the full Umurinzi model on an arbitrary sector — used when the
    Forest Manager clicks a sector on the choropleth. Same pipeline as the
    citizen flow, just keyed by sector_id instead of GPS/draw/upload."""
    data = request.get_json() or {}
    sector_id = (data.get("sector_id") or "").strip()
    if not sector_id:
        return jsonify({"error": "sector_id required"}), 400
    row = _SECTORS[_SECTORS["sector_id"].astype(str) == sector_id]
    if row.empty:
        return jsonify({"error": f"Unknown sector_id {sector_id}"}), 404
    sec = row.iloc[0]

    # Enforce district scope so managers can't analyse outside their patch
    user = current_user()
    if user["district_scope"] and sec["district"] != user["district_scope"]:
        return jsonify({"error": "Sector outside your district scope"}), 403

    result = analyse_parcel(
        lat=float(sec["centroid_lat"]),
        lng=float(sec["centroid_lng"]),
        area_ha=None,
    )
    result["sector_id"]   = sector_id
    result["sector_name"] = str(sec["sector"])
    result["district"]    = str(sec["district"])
    result["province"]    = str(sec["province"])
    result["centroid_lat"] = float(sec["centroid_lat"])
    result["centroid_lng"] = float(sec["centroid_lng"])
    return jsonify(result)


@app.get("/api/sector-risk")
@login_required
def api_sector_risk():
    """Sector-level risk data scoped to the logged-in user's district.
    Admins see every sector; forest managers see only their assigned district.
    ---
    tags:
      - Guidance
    produces:
      - application/json
    responses:
      200:
        description: Per-sector aggregated risk scores, scoped to the caller's district.
      401:
        description: Authentication required.
      500:
        description: sector_risk.json has not been generated yet.
    """
    path = Path(__file__).parent / "results" / "application" / "sector_risk.json"
    if not path.exists():
        return jsonify({"error": "sector_risk.json not generated yet"}), 500
    data = json.loads(path.read_text())
    user = current_user()
    scope = user.get("district_scope")
    if scope:  # forest manager — keep only sectors in their district
        in_scope = [s for s in data["sectors"] if s["district"] == scope]
        risks = [s["risk_level"] for s in in_scope]
        data["sectors"] = in_scope
        data["summary"] = {
            "total_sectors":       len(in_scope),
            "assessed_sectors":    sum(1 for r in risks if r != "UNKNOWN"),
            "high_risk_sectors":   risks.count("HIGH"),
            "medium_risk_sectors": risks.count("MEDIUM"),
            "low_risk_sectors":    risks.count("LOW"),
        }
        data["scope"] = {"district": scope, "view": "district_only"}
    else:  # admin — pass through
        data["scope"] = {"district": "ALL", "view": "national"}
    data["user"] = {
        "full_name":      user["full_name"],
        "role":           user["role"],
        "district_scope": user["district_scope"],
    }
    return jsonify(data)


@app.get("/static/sectors.geojson")
def sectors_geojson():
    """Serve the bundled sector polygons (cached aggressively in the browser)."""
    path = Path(__file__).parent / "data" / "geo" / "sectors_wgs84.geojson"
    return path.read_text(), 200, {
        "Content-Type": "application/json",
        "Cache-Control": "public, max-age=86400",
    }


_CUT_RECENT = {}   # analysis_id → last full analyse result (for simulate to look up)

# ── Lookup the seeded ALTERNATIVES SQLite table ────────────────────────
import sqlite3
_DB_PATH = Path(__file__).parent / "data" / "database" / "umurinzi.db"
if not _DB_PATH.exists():
    print(f"[boot]   ⚠ {_DB_PATH} not found — alternatives endpoint will return empty")
else:
    print(f"[boot]   alternatives DB: {_DB_PATH}")


@app.post("/api/alternatives")
def api_alternatives():
    """Return vetted alternatives for a given cutting reason and risk level.
    Triggered when current OR simulated risk is HIGH/MEDIUM and the citizen
    selects a cutting reason; returns pre-written, source-verified suggestions
    (e.g. government programs) in the requested language.
    ---
    tags:
      - Guidance
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [analysis_id, reason]
          properties:
            analysis_id:
              type: integer
              description: ID returned by a prior /api/analyse call.
              example: 1
            reason:
              type: string
              enum: [firewood, timber, farming, income]
              description: Why the citizen wants to cut.
              example: firewood
            language:
              type: string
              enum: [rw, en, fr]
              default: en
              description: Language for the suggestion text.
    responses:
      200:
        description: Matching suggestions for (reason, risk_level, language).
        examples:
          application/json:
            analysis_id: 1
            reason: "firewood"
            risk_level: "HIGH"
            language: "en"
            suggestions:
              - suggestion_text: "Apply for the subsidised efficient-cookstove program."
                gov_program_url: "https://example.gov.rw/cookstoves"
                source_verified: 1
      400:
        description: analysis_id and reason required; reason must be a valid enum value.
    """
    data = request.get_json() or {}
    try:
        aid    = int(data["analysis_id"])
        reason = str(data["reason"]).lower()
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "analysis_id and reason required"}), 400
    if reason not in ("firewood", "timber", "farming", "income"):
        return jsonify({"error": "reason must be firewood, timber, farming, or income"}), 400
    language = data.get("language", "en")

    orig = _CUT_RECENT.get(aid)
    risk_level = orig["risk_level"] if orig else "HIGH"

    if not _DB_PATH.exists():
        return jsonify({"suggestions": [], "warning": "DB not initialised"})

    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT suggestion_text, gov_program_url, source_verified "
        "FROM ALTERNATIVES "
        "WHERE reason = ? AND risk_level = ? AND language = ?",
        (reason, risk_level, language)
    ).fetchall()
    con.close()

    return jsonify({
        "analysis_id":  aid,
        "reason":       reason,
        "risk_level":   risk_level,
        "language":     language,
        "suggestions":  [dict(r) for r in rows],
    })


@app.post("/api/simulate")
def api_simulate():
    """Forward-simulate a proposed cut on an already-analysed parcel.
    Estimates the parcel's NDVI, tree cover and neighbourhood state after
    clearing `cut_area_ha`, then re-runs the three-rule risk classifier so the
    citizen sees how the proposed cut would change the risk level.

    Math:  new_ndvi  = (1 - f) * orig_ndvi + f * CLEARED_NDVI
           where f = cut_area_ha / parcel_area_ha
           new_neigh_deforested_pct += (cut_area_ha / 78.5) * 100
    ---
    tags:
      - Risk analysis
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [analysis_id, cut_area_ha]
          properties:
            analysis_id:
              type: integer
              description: ID returned by a prior /api/analyse call.
              example: 1
            cut_area_ha:
              type: number
              description: Proposed clearing area in hectares (0 < cut ≤ parcel area).
              example: 0.02
    responses:
      200:
        description: Before / after / delta of the simulated cut.
        examples:
          application/json:
            before: {risk_level: "MEDIUM", tree_cover_pct: 41.0, ndvi_current: 0.41}
            after:  {risk_level: "HIGH", tree_cover_pct: 24.0, ndvi_current: 0.24, rule_fired: "Rule 1 (parcel)"}
            delta:  {tree_cover_pct: -17.0, risk_level_change: 1}
            recovery_years_estimate: "6–8 years"
      400:
        description: Missing fields, or cut_area_ha outside (0, parcel_area].
      404:
        description: analysis_id not found — run /api/analyse first.
    """
    data = request.get_json() or {}
    try:
        aid = int(data["analysis_id"])
        cut = float(data["cut_area_ha"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "analysis_id and cut_area_ha required"}), 400

    orig = _CUT_RECENT.get(aid)
    if orig is None:
        return jsonify({"error": "analysis_id not found — run /api/analyse first"}), 404
    parcel_area = orig.get("parcel_area_ha") or 0.05
    if cut <= 0 or cut > parcel_area:
        return jsonify({"error": f"cut_area_ha must be in (0, {parcel_area}]"}), 400

    f = cut / parcel_area
    new_ndvi  = (1 - f) * orig["ndvi_current"] + f * _CLEARED_NDVI
    new_tree_cover = max(0.0, min(100.0, new_ndvi * 100.0))
    new_neigh_def  = orig["neighbourhood_500m_deforested_pct"] + (cut / 78.5) * 100

    # Re-run the 3-rule classifier on the simulated state
    prob = orig["deforestation_prob"]   # keep model prob; cut affects context not pixel-level
    avg_ndvi_500m = orig["neighbourhood_500m_avg_ndvi"]
    rule1 = (prob > 0.65) or (new_tree_cover < 30)
    rule2 = (new_neigh_def > 50) and (new_ndvi < avg_ndvi_500m * 0.70)
    rule3 = (0.35 < prob <= 0.65) and (new_neigh_def > 0)
    if rule1 or rule2:
        new_risk = "HIGH"
        fired   = "Rule 1 (parcel)" if rule1 else "Rule 2 (neighbourhood)"
    elif rule3:
        new_risk = "MEDIUM"; fired = "Rule 3"
    else:
        new_risk = "LOW";    fired = "default"

    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    delta_risk = risk_order[new_risk] - risk_order[orig["risk_level"]]

    return jsonify({
        "before": {
            "risk_level":            orig["risk_level"],
            "tree_cover_pct":        orig["tree_cover_pct"],
            "ndvi_current":          orig["ndvi_current"],
            "neighbourhood_500m_deforested_pct": orig["neighbourhood_500m_deforested_pct"],
        },
        "after": {
            "risk_level":            new_risk,
            "tree_cover_pct":        round(new_tree_cover, 1),
            "ndvi_current":          round(new_ndvi, 3),
            "neighbourhood_500m_deforested_pct": round(new_neigh_def, 1),
            "rule_fired":            fired,
        },
        "delta": {
            "tree_cover_pct":        round(new_tree_cover - orig["tree_cover_pct"], 1),
            "ndvi_current":          round(new_ndvi - orig["ndvi_current"], 3),
            "neighbourhood_500m_deforested_pct": round(new_neigh_def - orig["neighbourhood_500m_deforested_pct"], 1),
            "risk_level_change":     delta_risk,   # +1 means risk worsened by one class
        },
        "recovery_years_estimate": "6–8 years",   # from Hansen 2020-22 → 2024 recovery trend
        "cleared_ndvi_reference":  round(_CLEARED_NDVI, 3),
        "cut_fraction":            round(f, 3),
    })


@app.post("/api/analyse")
def api_analyse():
    """Predict deforestation risk for a parcel at a given location.
    Looks up the nearest trained pixel, runs the tuned Random Forest, applies
    the three-rule risk classifier, and returns the parcel + neighbourhood state.
    The returned analysis_id can then be passed to /api/simulate and /api/alternatives.
    ---
    tags:
      - Risk analysis
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [lat, lng]
          properties:
            lat:
              type: number
              description: Latitude (WGS84).
              example: -2.4521
            lng:
              type: number
              description: Longitude (WGS84).
              example: 29.1043
            area_ha:
              type: number
              description: Optional parcel area in hectares.
              example: 0.05
    responses:
      200:
        description: Risk assessment for the parcel.
        examples:
          application/json:
            analysis_id: 1
            risk_level: "HIGH"
            deforestation_prob: 0.72
            tree_cover_pct: 41.0
            ndvi_current: 0.41
            parcel_area_ha: 0.05
      400:
        description: lat and lng are required and must be floats.
    """
    data = request.get_json() or {}
    try:
        lat = float(data["lat"])
        lng = float(data["lng"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "lat and lng are required floats"}), 400
    area_ha = data.get("area_ha")
    result = analyse_parcel(lat, lng, area_ha=area_ha)
    # Store so /api/simulate can reference this analysis by id later
    _CUT_RECENT[result["analysis_id"]] = result
    return jsonify(result)


@app.post("/api/manual-coords")
def api_manual_coords():
    """Manual coordinate entry — citizen reads the printed numbers from
    their certificate and types them in. Same output shape as /api/extract
    so the rest of the front-end pipeline doesn't care which path was used.
    ---
    tags:
      - Parcel input
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [easting, northing]
          properties:
            easting:
              type: number
              description: Rwanda local TM easting (500,000–600,000).
              example: 530412
            northing:
              type: number
              description: Rwanda local TM northing (4,700,000–4,900,000).
              example: 4793215
            area_sqm:
              type: number
              description: Optional parcel area in m²; if given, a square polygon is drawn.
              example: 512
    responses:
      200:
        description: Coordinates converted to WGS84 lat/lng (+ optional polygon).
        examples:
          application/json:
            easting: 530412
            northing: 4793215
            lat: -2.4521
            lng: 29.1043
            polygon_status: "manual"
            source: "manual_entry"
      400:
        description: Missing/invalid numbers, or coordinates outside Rwanda's range.
    """
    from pyproj import Transformer
    transformer = Transformer.from_crs(
        "+proj=tmerc +lat_0=0 +lon_0=30 +k=0.9999 +x_0=500000 +y_0=5000000 "
        "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs",
        "EPSG:4326", always_xy=True
    )
    data = request.get_json() or {}
    try:
        easting  = float(data["easting"])
        northing = float(data["northing"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Easting and Northing required as numbers"}), 400
    area_sqm = data.get("area_sqm")

    # Basic sanity: Rwanda local TM Easting is 5xx,xxx; Northing 47xx,xxx-48xx,xxx
    if not (500_000 <= easting <= 600_000):
        return jsonify({"error": f"Easting {easting} outside Rwanda's UTM range (500-600k)"}), 400
    if not (4_700_000 <= northing <= 4_900_000):
        return jsonify({"error": f"Northing {northing} outside Rwanda's UTM range"}), 400

    lng, lat = transformer.transform(easting, northing)

    # If the citizen provided an area, build an approximate square polygon for display
    polygon_wgs84 = None
    if area_sqm and area_sqm > 0:
        side_m = (float(area_sqm)) ** 0.5
        half = side_m / 2
        corners_utm = [
            (easting - half, northing + half),  # TL
            (easting + half, northing + half),  # TR
            (easting + half, northing - half),  # BR
            (easting - half, northing - half),  # BL
        ]
        polygon_wgs84 = []
        for e, n in corners_utm:
            cx, cy = transformer.transform(e, n)
            polygon_wgs84.append([cx, cy])
        polygon_wgs84.append(polygon_wgs84[0])

    return jsonify({
        "upi": None,
        "surface_sqm": area_sqm,
        "easting": easting,
        "northing": northing,
        "lat": lat,
        "lng": lng,
        "polygon_wgs84": polygon_wgs84,
        "extracted_area_m2": area_sqm,
        "polygon_status": "manual" if polygon_wgs84 else "manual_centroid_only",
        "source": "manual_entry",
    })


@app.post("/api/extract")
def api_extract():
    """Extract a parcel location from an uploaded land-title document.
    Runs OCR + cadastral parsing on a Rwanda land title (PDF or photo) and
    returns the parcel centroid, polygon, and UPI where detectable.
    ---
    tags:
      - Parcel input
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: A land-title PDF or photo (.pdf, .png, .jpg, .jpeg, .webp, .tiff). Max 16 MB.
    responses:
      200:
        description: Extracted parcel location and geometry.
        examples:
          application/json:
            upi: "1/05/10/05/7914"
            lat: -2.4521
            lng: 29.1043
            extracted_area_m2: 512.0
            polygon_status: "ok"
            source: "pdf"
      400:
        description: No file uploaded, or unsupported file type.
      500:
        description: Extraction failed (OCR or parsing error).
    """
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    suffix = "." + f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff"}:
        return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

    # Write to a project-local temp file (sandbox rules block /tmp tesseract reads)
    workdir = Path(__file__).resolve().parent / "data" / "raw" / "external" / "_uploads"
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = workdir / f"upload_{abs(hash(f.filename))}{suffix}"
    try:
        f.save(tmp)
        result = extract(tmp)
    except Exception as e:
        return jsonify({"error": f"Extraction failed: {e}"}), 500
    finally:
        if tmp.exists():
            tmp.unlink()

    return jsonify(result)


if __name__ == "__main__":
    print("\n🛡️  Umurinzi running")
    print("   Landing:   http://localhost:5050/")
    print("   Citizen:   http://localhost:5050/citizen")
    print("   Manager:   http://localhost:5050/manager")
    print("   Admin:     http://localhost:5050/admin\n")
    app.run(host="127.0.0.1", port=5050, debug=False)

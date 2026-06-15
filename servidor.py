"""
Anticlustering — Servidor web local
=====================================
Ejecutar:
    pip install flask umap-learn plotly numpy pandas scikit-learn
    python servidor.py

Luego abrir: http://localhost:5000
"""

import io, json, uuid, threading, traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import to_json
import umap
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, send_file, render_template_string

app = Flask(__name__)
JOBS = {}   # { job_id: { status, progress, message, result, error } }
SEED = 42

# ══════════════════════════════════════════════════════════════
# ESQUEMA POR DEFECTO (se infiere del CSV si no coincide)
# ══════════════════════════════════════════════════════════════

DEFAULT_SCHEMA = {
    "actividad":    "nominal",  # Ganadería, Agricultura, Contratistas, Mixto
    "tecnologia":   "ordinal",  # Adopción de nuevas tecnologías (ej. 1 al 5)
    "recambio":     "nominal",  # Recambio generacional (ej. "No iniciado", "En proceso", "Completado")
    "ambiente":     "ordinal",  # Foco en Ambiente/Sustentabilidad (ej. 1 al 5)
    "escala":       "ordinal",  # Escala productiva / Tamaño (ej. "Chica", "Mediana", "Grande")
    "edad":         "numeric",  # Edad del productor
}


# ══════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════

def infer_schema(df, group_col="grupo", id_col="id"):
    """Infiere tipo de cada columna si no está en DEFAULT_SCHEMA."""
    skip = {group_col, id_col}
    schema = {}
    for col in df.columns:
        if col in skip:
            continue
        if col in DEFAULT_SCHEMA:
            schema[col] = DEFAULT_SCHEMA[col]
            continue
        s = df[col]
        if s.dtype == object or s.nunique() < 6:
            schema[col] = "nominal" if s.nunique() > 2 else "binary"
        elif s.nunique() <= 7:
            schema[col] = "ordinal"
        else:
            schema[col] = "numeric"
    return schema


def encode_columns(df, schema):
    """Convierte columnas al tipo numérico que espera Gower."""
    df2 = df.copy()
    for col, ftype in schema.items():
        if ftype == "binary":
            vals = df2[col].astype(str)
            cats = vals.unique()
            df2[col] = vals.map({c: i for i, c in enumerate(cats)}).astype(float)
        elif ftype == "nominal":
            vals = df2[col].astype(str)
            cats = vals.unique()
            df2[col] = vals.map({c: i for i, c in enumerate(cats)}).astype(float)
        else:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0)
    return df2


def calibrate_weights(df, schema):
    weights = {}
    for col, ftype in schema.items():
        vals = df[col].values.astype(float)
        if ftype in ("numeric", "ordinal"):
            r = vals.max() - vals.min()
            weights[col] = ((vals - vals.min()) / r).var() if r > 0 else 0.0
        elif ftype == "binary":
            p = vals.mean()
            weights[col] = p * (1 - p)
        elif ftype == "nominal":
            counts = pd.Series(vals).value_counts(normalize=True)
            weights[col] = 1 - (counts ** 2).sum()
    total = sum(weights.values()) or 1
    p = len(weights)
    return {c: w * p / total for c, w in weights.items()}


def gower_matrix(df, schema, weights):
    cols = list(schema.keys())
    n = len(df)
    w_total = sum(weights[c] for c in cols)
    D = np.zeros((n, n))
    for col, ftype in schema.items():
        w = weights[col]
        vals = df[col].values.astype(float)
        if ftype in ("numeric", "ordinal"):
            r = vals.max() - vals.min()
            if r == 0:
                continue
            d_k = np.abs(vals[:, None] - vals[None, :]) / r
        else:
            d_k = (vals[:, None] != vals[None, :]).astype(float)
        D += w * d_k
    if w_total > 0:
        D /= w_total
    return D


def feasible_init(groups, K, table_size):
    n = len(groups)
    assignment = np.full(n, -1, dtype=int)
    mgc = [defaultdict(int) for _ in range(K)]
    mesa_size = np.zeros(K, dtype=int)

    unique, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)
    by_group = {}
    for g in unique:
        idxs = np.where(groups == g)[0].tolist()
        np.random.shuffle(idxs)
        by_group[g] = idxs

    for g in unique[order]:
        for person in by_group[g]:
            cands = np.where(mesa_size < table_size)[0]
            no_conflict = [m for m in cands if mgc[m][g] == 0]
            if no_conflict:
                mesa = max(no_conflict, key=lambda m: mesa_size[m])
            else:
                mesa = min(cands, key=lambda m: mgc[m][g])
            assignment[person] = mesa
            mgc[mesa][g] += 1
            mesa_size[mesa] += 1
    return assignment, mgc


def obj(assignment, D, K):
    total = 0.0
    for k in range(K):
        idx = np.where(assignment == k)[0]
        if len(idx) > 1:
            total += D[np.ix_(idx, idx)].sum() / 2
    return total


def delta_div(i, j, assignment, D):
    gi, gj = assignment[i], assignment[j]
    ii = np.where(assignment == gi)[0]
    jj = np.where(assignment == gj)[0]
    return ((D[i, jj[jj != j]].sum() + D[j, ii[ii != i]].sum()) -
            (D[i, ii[ii != i]].sum() + D[j, jj[jj != j]].sum()))


def classify(i, j, assignment, groups, mgc):
    gi, gj = assignment[i], assignment[j]
    if gi == gj:
        return "skip"
    group_i, group_j = groups[i], groups[j]
    cur = max(0, mgc[gi][group_i] - 1) + max(0, mgc[gj][group_j] - 1)
    new = max(0, mgc[gj][group_i]) + max(0, mgc[gi][group_j])
    delta = new - cur
    if delta > 0:
        return "illegal"
    return "legal" if delta < 0 else "neutral"


def do_swap(i, j, assignment, groups, mgc):
    gi, gj = assignment[i], assignment[j]
    mgc[gi][groups[i]] -= 1
    mgc[gj][groups[j]] -= 1
    assignment[i], assignment[j] = gj, gi
    mgc[gj][groups[i]] += 1
    mgc[gi][groups[j]] += 1


def exchange(D, groups, assignment, mgc, progress_cb=None):
    n = len(assignment)
    it = 0
    while True:
        improved = False
        it += 1
        for i in range(n):
            for j in range(i + 1, n):
                cat = classify(i, j, assignment, groups, mgc)
                if cat in ("skip", "illegal"):
                    continue
                dd = delta_div(i, j, assignment, D)
                if cat == "legal" or (cat == "neutral" and dd > 1e-10):
                    do_swap(i, j, assignment, groups, mgc)
                    improved = True
        if progress_cb:
            progress_cb(it)
        if not improved:
            break
    return assignment, mgc


def sa(D, groups, assignment, mgc, n_iter=60_000, progress_cb=None):
    n = len(assignment)
    # Calibrar T0
    deltas = []
    for _ in range(1500):
        i, j = np.random.randint(n), np.random.randint(n)
        if i == j or assignment[i] == assignment[j]:
            continue
        d = abs(delta_div(i, j, assignment, D))
        if d > 0:
            deltas.append(d)
    T = np.mean(deltas) / (-np.log(0.8)) if deltas else 0.01
    alpha = 0.9995
    cur = obj(assignment, D, len(np.unique(assignment)))
    best_a = assignment.copy()
    best_mgc = [defaultdict(int, m) for m in mgc]
    best = cur
    log_at = set(range(0, n_iter, n_iter // 8))

    for t in range(n_iter):
        i, j = np.random.randint(n), np.random.randint(n)
        if i == j:
            continue
        cat = classify(i, j, assignment, groups, mgc)
        if cat in ("skip", "illegal"):
            continue
        dd = delta_div(i, j, assignment, D)
        accept = cat == "legal" or (dd > 0) or (np.random.random() < np.exp(dd / T))
        if accept:
            do_swap(i, j, assignment, groups, mgc)
            cur += dd
            if cur > best:
                best = cur
                best_a = assignment.copy()
                best_mgc = [defaultdict(int, m) for m in mgc]
        T *= alpha
        if progress_cb and t in log_at:
            progress_cb(t, n_iter)

    assignment[:] = best_a
    for k in range(len(mgc)):
        mgc[k] = best_mgc[k]
    return assignment, mgc


# ══════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════

BG    = "#ffffff"
GRID  = "#e5e7eb"
TEXT  = "#4b5563"
TEXT2 = "#111827"
ACC   = "#000000"
ACC2  = "#374151"
ACC3  = "#111827"
MUTED = "#9ca3af"

LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="'DM Mono', monospace", color=TEXT, size=11),
    margin=dict(l=48, r=24, t=40, b=40),
    title=dict(font=dict(color=TEXT2, size=12, family="'Syne', sans-serif"), x=0.02, xanchor="left"),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(color=TEXT2, size=11)),
)

def ax(title="", **kw):
    """Eje estándar: grid sutil, ticks discretos."""
    d = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
             tickfont=dict(color=TEXT, size=9))
    if title:
        d["title"] = dict(text=title, font=dict(color=TEXT, size=10))
    return {**d, **kw}

def ax_blank(**kw):
    """Eje sin marcas, para embeddings donde las unidades no significan nada."""
    d = dict(showgrid=False, zeroline=False, showticklabels=False,
             ticks="", showline=False)
    return {**d, **kw}


# Escalas de color continuas y discretas, muted (B&W y grises)
SCALE_MESA  = [[0, "#000000"], [0.5, "#4b5563"], [1, "#d1d5db"]]
SCALE_HEAT  = [[0, "#ffffff"], [0.5, "#e5e7eb"], [1, "#000000"]]


def _layout(title_text):
    """Return a copy of LAYOUT with the title text set (avoids duplicate kwarg)."""
    L = dict(LAYOUT)
    L["title"] = {**LAYOUT["title"], "text": title_text}
    return L


def make_charts(df_raw, schema, weights, assignment, groups, K):
    n = len(assignment)
    cols = list(schema.keys())
    df_enc = encode_columns(df_raw[cols], schema)

    # UMAP
    D = gower_matrix(df_enc, schema, weights)
    reducer = umap.UMAP(n_components=2, metric="precomputed",
                        n_neighbors=15, min_dist=0.2, random_state=SEED)
    emb = reducer.fit_transform(D)

    # ── UMAP por mesa ──
    f1 = go.Figure(go.Scatter(
        x=emb[:, 0], y=emb[:, 1], mode="markers",
        marker=dict(
            color=assignment, colorscale=SCALE_MESA, size=7, opacity=0.85,
            line=dict(width=0.5, color="rgba(0,0,0,0.2)"),
            colorbar=dict(title=dict(text="Mesa", font=dict(color=TEXT, size=10)),
                          tickfont=dict(color=TEXT, size=9),
                          outlinewidth=0, thickness=8, len=0.6),
        ),
        customdata=assignment + 1,
        hovertemplate="Mesa %{customdata}<extra></extra>",
    ))
    f1.update_layout({**_layout("Mapa de similitud — color = mesa asignada"),
                     "xaxis": ax_blank(), "yaxis": ax_blank(),
                     "showlegend": False, "height": 380})

    # ── Heatmap grupos × mesas ──
    unique_groups_num = np.unique(groups)
    matrix = np.zeros((len(unique_groups_num), K), dtype=int)
    for gi, g in enumerate(unique_groups_num):
        for k in range(K):
            matrix[gi, k] = ((assignment == k) & (groups == g)).sum()

    group_labels = (
        [str(df_raw["grupo"].values[np.where(groups == g)[0][0]])
         for g in unique_groups_num]
        if "grupo" in df_raw.columns else [f"G{g}" for g in unique_groups_num]
    )
    zmax = max(2, matrix.max())

    f3 = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"{k+1}" for k in range(K)],
        y=group_labels,
        zmin=0, zmax=zmax,
        colorscale=SCALE_HEAT,
        showscale=False,
        xgap=3, ygap=3,
        hovertemplate="Mesa %{x} · %{y}<br>%{z} persona(s)<extra></extra>",
    ))
    f3.update_layout({**_layout("Distribución: grupos en cada mesa"),
                     "xaxis": ax("Mesa", tickfont=dict(size=8)),
                     "yaxis": ax(tickfont=dict(size=9)),
                     "height": max(240, len(unique_groups_num) * 22 + 80)})

    # ── Diversidad por mesa ──
    mesa_scores = [D[np.ix_(np.where(assignment == k)[0],
                             np.where(assignment == k)[0])].sum() / 2
                   for k in range(K)]
    median_div = np.median(mesa_scores)
    colors = ["#000000" if s >= median_div else "#4b5563" for s in mesa_scores]

    f4 = go.Figure(go.Bar(
        x=[f"{k+1}" for k in range(K)], y=mesa_scores,
        marker=dict(color=colors, line=dict(width=0), opacity=0.85),
        hovertemplate="Mesa %{x}<br>Diversidad: %{y:.3f}<extra></extra>",
    ))
    f4.add_hline(y=median_div, line_dash="dot", line_width=1, line_color=MUTED,
                 annotation_text=f"mediana {median_div:.2f}",
                 annotation_font_color=MUTED, annotation_font_size=9)
    f4.update_layout(**{**_layout("Diversidad por mesa"),
                     "xaxis": ax("Mesa", tickfont=dict(size=8)),
                     "yaxis": ax("Score"), "showlegend": False, "height": 260})

    return {
        "umap_mesa":    json.loads(to_json(f1)),
        "heatmap":      json.loads(to_json(f3)),
        "diversidad":   json.loads(to_json(f4)),
    }




# ══════════════════════════════════════════════════════════════
# WORKER
# ══════════════════════════════════════════════════════════════

def run_pipeline(job_id, csv_bytes, table_size, sa_iters):
    def upd(msg, pct):
        JOBS[job_id]["message"] = msg
        JOBS[job_id]["progress"] = pct

    try:
        np.random.seed(SEED)
        upd("Leyendo CSV…", 5)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        # Detectar columnas de id y grupo
        id_col    = next((c for c in df.columns if c.lower() == "id"), None)
        group_col = next((c for c in df.columns if c.lower() in ("grupo", "group")), None)
        if group_col is None:
            raise ValueError("El CSV necesita una columna llamada 'grupo' o 'group'.")

        groups_raw = df[group_col].values
        unique_g, counts_g = np.unique(groups_raw, return_counts=True)
        group_to_int = {g: i for i, g in enumerate(unique_g)}
        groups = np.array([group_to_int[g] for g in groups_raw])

        n = len(df)
        K = n // table_size
        if K < 1:
            raise ValueError(f"Con {n} personas y mesas de {table_size}, no hay suficientes datos.")

        upd("Inferiendo esquema…", 10)
        skip = {id_col, group_col} if id_col else {group_col}
        df_feat = df.drop(columns=[c for c in skip if c], errors="ignore")
        schema  = infer_schema(df_feat, group_col="", id_col="")
        df_enc  = encode_columns(df_feat[list(schema.keys())], schema)

        upd("Calibrando pesos…", 15)
        weights = calibrate_weights(df_enc, schema)

        upd("Calculando matriz de Gower…", 20)
        D = gower_matrix(df_enc, schema, weights)

        upd("Inicializando asignación…", 30)
        assignment, mgc = feasible_init(groups, K, table_size)

        upd("Exchange Algorithm…", 40)
        assignment, mgc = exchange(D, groups, assignment, mgc)

        upd("Simulated Annealing…", 60)
        assignment, mgc = sa(D, groups, assignment, mgc, n_iter=sa_iters)

        upd("Pulido final…", 80)
        assignment, mgc = exchange(D, groups, assignment, mgc)

        upd("Generando visualizaciones…", 88)
        charts = make_charts(df, schema, weights, assignment, groups, K)

        upd("Preparando resultados…", 95)

        # Stats
        mesa_scores = [D[np.ix_(np.where(assignment == k)[0],
                                 np.where(assignment == k)[0])].sum() / 2
                       for k in range(K)]
        total_conflicts = sum(
            max(0, sum(groups[np.where(assignment == k)[0]] == g) - 1)
            for k in range(K) for g in np.unique(groups)
        )
        inevitable = sum(max(0, c - K) for c in counts_g)
        avg_groups = np.mean([
            len(np.unique(groups[np.where(assignment == k)[0]])) for k in range(K)
        ])

        # CSV de asignación
        out_df = df.copy()
        out_df["mesa"] = assignment + 1
        out_df = out_df.sort_values("mesa")
        csv_out = out_df.to_csv(index=False)

        # Tabla de mesas para mostrar en pantalla
        id_col_name = id_col if id_col else None
        mesa_table = []
        for k in range(K):
            members = out_df[out_df["mesa"] == k + 1]
            member_list = []
            for _, row in members.iterrows():
                name = str(row[id_col_name]) if id_col_name and id_col_name in row else f"P{row.name}"
                grp = str(row[group_col]) if group_col in row else ""
                member_list.append({"nombre": name, "grupo": grp})
            mesa_table.append({"mesa": k + 1, "personas": member_list, "total": len(member_list)})

        JOBS[job_id] = {
            "status": "done",
            "progress": 100,
            "message": "Listo",
            "result": {
                "stats": {
                    "n":                n,
                    "K":                K,
                    "table_size":       table_size,
                    "n_groups":         len(unique_g),
                    "diversity_total":  round(sum(mesa_scores), 2),
                    "diversity_mean":   round(np.mean(mesa_scores), 4),
                    "diversity_std":    round(np.std(mesa_scores), 4),
                    "conflicts_real":   int(total_conflicts),
                    "conflicts_inev":   int(inevitable),
                    "avg_groups_table": round(float(avg_groups), 1),
                    "n_questions":      len(schema),
                },
                "charts":  charts,
                "csv_out": csv_out,
                "mesa_table": mesa_table,
                "group_summary": [
                    {"grupo": str(g), "n": int(c),
                     "inevitables": int(max(0, c - K))}
                    for g, c in zip(unique_g, counts_g)
                ],
            },
        }

    except Exception:
        JOBS[job_id] = {
            "status": "error",
            "progress": 0,
            "message": traceback.format_exc(),
            "result": None,
        }


# ══════════════════════════════════════════════════════════════
# RUTAS
# ══════════════════════════════════════════════════════════════

@app.route("/logo-crea.svg")
def logo_crea():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_file(os.path.join(base_dir, "logo-crea.svg"), mimetype="image/svg+xml")

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/procesar", methods=["POST"])
def procesar():
    f = request.files.get("archivo")
    if not f:
        return jsonify({"error": "No se recibió archivo"}), 400
    table_size = int(request.form.get("table_size", 10))
    sa_iters   = int(request.form.get("sa_iters", 60000))
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running", "progress": 0,
                    "message": "Iniciando…", "result": None}
    csv_bytes = f.read()
    t = threading.Thread(target=run_pipeline,
                         args=(job_id, csv_bytes, table_size, sa_iters),
                         daemon=True)
    t.start()
    return jsonify({"job_id": job_id})

@app.route("/estado/<job_id>")
def estado(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job no encontrado"}), 404
    r = {"status": job["status"], "progress": job["progress"],
         "message": job["message"]}
    if job["status"] == "done":
        res = job["result"]
        r["stats"]         = res["stats"]
        r["charts"]        = res["charts"]
        r["mesa_table"]    = res["mesa_table"]
        r["group_summary"] = res["group_summary"]
    elif job["status"] == "error":
        r["error"] = job["message"]
    return jsonify(r)

@app.route("/descargar/<job_id>")
def descargar(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return "No disponible", 404
    csv_out = job["result"]["csv_out"]
    buf = io.BytesIO(csv_out.encode("utf-8-sig"))
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name="asignacion_mesas.csv")


# ══════════════════════════════════════════════════════════════
# FRONTEND
# ══════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CREA · Anticlustering</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Syne:wght@600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:     #ffffff;
  --bg2:    #ffffff;
  --bg3:    #f9fafb;
  --border: #e5e7eb;
  --text:   #111827;
  --muted:  #4b5563;
  --acc:    #000000;
  --acc2:   #374151;
  --acc3:   #111827;
  --acc4:   #9ca3af;
  --radius: 6px;
  --mono:   'DM Mono', monospace;
  --display:'Syne', sans-serif;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  font-size: 13px;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── GRID BACKGROUND ── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background-image:
    linear-gradient(rgba(0,0,0,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,0,0,0.015) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}

.wrap { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 0 32px 80px; }

/* ── HEADER ── */
header {
  padding: 52px 0 40px;
  display: flex; align-items: center; justify-content: space-between; gap: 32px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 48px;
}
.brand-container {
  display: flex;
  align-items: center;
  gap: 20px;
}
.logo-img {
  height: 52px;
  width: auto;
  display: block;
}
.logo {
  font-family: var(--display);
  font-size: 32px; font-weight: 800;
  letter-spacing: -1px;
  color: var(--text);
}
.logo span { opacity: 0.3; }
.tagline { color: var(--muted); font-size: 12px; line-height: 1.6; text-align: right; }

/* ── PANELS ── */
.panel {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px 32px;
  margin-bottom: 24px;
}
.panel-title {
  font-family: var(--display);
  font-size: 15px; font-weight: 700;
  color: var(--text);
  margin-bottom: 20px;
  display: flex; align-items: center; gap: 10px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.panel-title .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--acc); flex-shrink: 0;
}

/* ── UPLOAD ZONE ── */
#drop-zone {
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  padding: 52px 24px;
  text-align: center;
  cursor: pointer;
  transition: all .2s;
  position: relative;
  background: var(--bg3);
}
#drop-zone:hover, #drop-zone.drag-over {
  border-color: var(--acc);
  background: #f3f4f6;
}
#drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.upload-icon { font-size: 36px; margin-bottom: 12px; filter: grayscale(100%); }
.upload-title { font-family: var(--display); font-size: 16px; font-weight: 700; margin-bottom: 6px; }
.upload-sub { color: var(--muted); font-size: 12px; }
#file-name {
  margin-top: 14px; padding: 8px 16px;
  background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
  color: var(--text); display: none; font-size: 12px;
  font-weight: bold;
}

/* ── CONFIG ── */
.config-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.field label { display: block; color: var(--muted); font-size: 11px;
               text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }
.field input, .field select {
  width: 100%;
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 4px; color: var(--text);
  font-family: var(--mono); font-size: 13px;
  padding: 10px 14px;
  transition: border-color .2s;
}
.field input:focus, .field select:focus {
  outline: none; border-color: var(--acc);
}
.field-hint { color: var(--muted); font-size: 11px; margin-top: 5px; }

/* ── BUTTON ── */
#run-btn {
  width: 100%; margin-top: 24px; padding: 14px;
  background: var(--acc);
  border: none; border-radius: 4px; cursor: pointer;
  font-family: var(--display); font-size: 14px; font-weight: 700;
  color: #fff; letter-spacing: .03em;
  transition: all .2s; position: relative; overflow: hidden;
  text-transform: uppercase;
}
#run-btn:hover:not(:disabled) { background: #1f2937; }
#run-btn:disabled { opacity: .3; cursor: not-allowed; transform: none; }

/* ── PROGRESS ── */
#progress-wrap { display: none; margin-top: 20px; }
.progress-bar-bg {
  background: var(--bg3); border-radius: 4px; height: 6px; overflow: hidden;
  border: 1px solid var(--border);
}
.progress-bar-fill {
  height: 100%;
  background: var(--acc);
  border-radius: 4px;
  transition: width .4s ease;
  width: 0%;
}
.progress-msg { color: var(--muted); font-size: 11px; margin-top: 8px; }

/* ── STATS ── */
#results { display: none; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; }
.stat {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 6px; padding: 18px 20px; text-align: center;
}
.stat-val { font-family: var(--display); font-size: 28px; font-weight: 800;
            margin: 8px 0 4px; }
.stat-label { color: var(--muted); font-size: 10px;
              text-transform: uppercase; letter-spacing: .1em; }
.stat-sub { color: var(--muted); font-size: 10px; margin-top: 3px; }

/* ── DOWNLOAD BTN ── */
#dl-btn {
  display: none; margin-top: 20px; padding: 12px 28px;
  background: var(--bg); border: 1px solid var(--acc);
  border-radius: 4px; color: var(--acc);
  font-family: var(--display); font-size: 13px; font-weight: 700;
  cursor: pointer; text-decoration: none;
  transition: all .2s; letter-spacing: .03em;
  text-transform: uppercase;
}
#dl-btn:hover { background: #f3f4f6; }

/* ── GROUP TABLE ── */
.group-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.group-table th { color: var(--muted); font-size: 10px; text-transform: uppercase;
                   letter-spacing: .08em; padding: 8px 12px; text-align: left;
                   border-bottom: 1px solid var(--border); }
.group-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
.group-table tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 10px; font-weight: 500; }
.badge-ok  { background: #f3f4f6; color: var(--text); border: 1px solid var(--border); }
.badge-warn { background: #fef2f2; color: #991b1b; border: 1px solid #fee2e2; }

/* ── CHARTS ── */
.chart-wrap {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 4px;
  margin-bottom: 20px; overflow: hidden;
}
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }

/* ── SECTION HEADER ── */
.section-header {
  font-family: var(--display); font-size: 13px; font-weight: 700;
  color: var(--muted); text-transform: uppercase; letter-spacing: .1em;
  margin: 40px 0 16px;
  display: flex; align-items: center; gap: 12px;
}
.section-header::after {
  content: ''; flex: 1; height: 1px; background: var(--border);
}

/* ── MESA CARDS ── */
.mesa-toolbar {
  display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
}
.mesa-search {
  flex: 1; max-width: 320px;
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text);
  font-family: var(--mono); font-size: 12px;
  padding: 9px 14px;
  transition: border-color .2s;
}
.mesa-search:focus { outline: none; border-color: var(--acc); }
.mesa-search::placeholder { color: var(--muted); }
.mesa-count { color: var(--muted); font-size: 11px; }

.mesa-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}
.mesa-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  transition: border-color .2s, transform .15s;
}
.mesa-card:hover {
  border-color: var(--acc);
  transform: translateY(-2px);
}
.mesa-card-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 14px; padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.mesa-card-num {
  font-family: var(--display); font-size: 18px; font-weight: 800;
  color: var(--acc);
}
.mesa-card-count {
  font-size: 11px; color: var(--muted);
  background: var(--bg3); padding: 3px 10px;
  border-radius: 4px;
  border: 1px solid var(--border);
}
.mesa-person {
  display: flex; align-items: center; justify-content: space-between;
  padding: 5px 0; font-size: 12px;
}
.mesa-person + .mesa-person { border-top: 1px solid var(--border); }
.mesa-person-name { color: var(--text); }
.mesa-person-group {
  font-size: 10px; color: var(--muted);
  background: var(--bg3); padding: 2px 8px;
  border-radius: 3px; max-width: 120px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  border: 1px solid var(--border);
}

/* ── ERROR ── */
#error-box {
  display: none; background: #fef2f2;
  border: 1px solid #fee2e2;
  border-radius: 8px; padding: 16px 20px;
  color: #991b1b; font-size: 12px;
  white-space: pre-wrap; margin-top: 16px;
}

@media (max-width: 700px) {
  .config-grid, .chart-grid, .mesa-grid { grid-template-columns: 1fr; }
  .wrap { padding: 0 16px 60px; }
  .logo { font-size: 28px; }
  header { flex-direction: column; align-items: flex-start; gap: 16px; }
  .tagline { text-align: left; }
}

/* ── TABS NAVIGATION ── */
.tabs-nav {
  display: flex;
  gap: 8px;
  margin-bottom: 32px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 1px;
}
.tab-btn {
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--muted);
  font-family: var(--display);
  font-size: 13px;
  font-weight: 700;
  padding: 12px 18px;
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.tab-btn:hover {
  color: var(--text);
}
.tab-btn.active {
  color: var(--acc);
  border-bottom-color: var(--acc);
}
.tab-content {
  animation: tabFadeIn 0.3s ease-in-out;
}
@keyframes tabFadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ── EXPLANATION TAB ── */
.expl-container {
  display: flex;
  flex-direction: column;
  gap: 36px;
}
.expl-hero {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 36px;
  text-align: left;
}
.expl-hero h1 {
  font-family: var(--display);
  font-size: 26px;
  font-weight: 800;
  margin-bottom: 12px;
  color: var(--text);
  letter-spacing: -0.5px;
}
.expl-hero p {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
  max-width: 900px;
}
.expl-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
}
.expl-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  position: relative;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.01);
}
.expl-card:hover {
  border-color: var(--acc);
  transform: translateY(-4px);
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.03);
}
.expl-card-num {
  font-family: var(--display);
  font-size: 40px;
  font-weight: 800;
  color: #f3f4f6;
  position: absolute;
  top: 10px;
  right: 20px;
  line-height: 1;
  z-index: 0;
}
.expl-card h3 {
  font-family: var(--display);
  font-size: 15px;
  font-weight: 700;
  margin-bottom: 10px;
  color: var(--text);
  position: relative;
  z-index: 1;
}
.expl-card p {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.6;
  position: relative;
  z-index: 1;
}
.expl-flow {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 32px 24px;
  text-align: center;
}
.expl-flow h3 {
  font-family: var(--display);
  font-size: 15px;
  font-weight: 700;
  margin-bottom: 28px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text);
}
.flow-steps {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.flow-step {
  flex: 1;
  padding: 12px;
}
.flow-step-icon {
  font-size: 32px;
  margin-bottom: 8px;
}
.flow-step h4 {
  font-family: var(--display);
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 6px;
  color: var(--text);
}
.flow-step p {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.5;
}
.flow-arrow {
  font-size: 24px;
  color: var(--border);
  font-family: var(--display);
  font-weight: 700;
}
@media (max-width: 768px) {
  .flow-steps {
    flex-direction: column;
  }
  .flow-arrow {
    transform: rotate(90deg);
    margin: 8px 0;
  }
}

/* ── BUILDER TAB ── */
.builder-container {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 24px;
}
@media (max-width: 850px) {
  .builder-container {
    grid-template-columns: 1fr;
  }
}
.questions-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-top: 16px;
}
.question-card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  position: relative;
  animation: qCardFadeIn 0.25s ease-out;
}
@keyframes qCardFadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.question-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.question-num {
  font-weight: bold;
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.remove-q-btn {
  background: none;
  border: none;
  color: #b91c1c;
  cursor: pointer;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: bold;
  text-transform: uppercase;
}
.remove-q-btn:hover {
  text-decoration: underline;
}
.question-fields {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 16px;
}
@media (max-width: 550px) {
  .question-fields {
    grid-template-columns: 1fr;
  }
}
.options-field {
  display: none;
}
.add-btn {
  background: var(--text);
  color: var(--bg);
  border: none;
  border-radius: 4px;
  padding: 8px 14px;
  font-family: var(--display);
  font-size: 11px;
  font-weight: 700;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  transition: all 0.2s;
}
.add-btn:hover {
  background: #374151;
}
.action-btn {
  width: 100%;
  padding: 12px 14px;
  border-radius: 4px;
  font-family: var(--display);
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  text-transform: uppercase;
  margin-bottom: 12px;
  transition: all 0.2s ease;
  text-align: center;
  letter-spacing: 0.03em;
}
.primary-btn {
  background: var(--acc);
  color: #fff;
  border: none;
}
.primary-btn:hover {
  background: #1f2937;
}
.secondary-btn {
  background: var(--bg);
  color: var(--acc);
  border: 1px solid var(--acc);
}
.secondary-btn:hover {
  background: #f3f4f6;
}
.success-box {
  background: #ecfdf5;
  border: 1px solid #a7f3d0;
  color: #065f46;
  border-radius: 4px;
  padding: 12px;
  font-size: 11px;
  margin-top: 12px;
  text-align: center;
}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="brand-container">
      <img class="logo-img" src="/logo-crea.svg" alt="CREA Logo">
    </div>
  </header>

  <div class="tabs-nav">
    <button class="tab-btn active" id="btn-tab-expl" onclick="switchTab('tab-expl')">📖 Cómo funciona</button>
    <button class="tab-btn" id="btn-tab-builder" onclick="switchTab('tab-builder')">🧪 Crear Datos Sintéticos</button>
    <button class="tab-btn" id="btn-tab-sim" onclick="switchTab('tab-sim')">⚙️ Simulador de Mesas</button>
  </div>

  <!-- TAB 1: EXPLANATION -->
  <div id="tab-expl" class="tab-content">
    <div class="expl-container">
      <div class="expl-hero">
        <h1>¿Qué es el Anticlustering?</h1>
        <p>El anticlustering es el proceso opuesto al "clustering" tradicional. En lugar de agrupar elementos similares, el anticlustering <strong>distribuye los elementos similares en grupos diferentes</strong>. El objetivo es crear mesas donde cada una sea un espacio diverso y representativo de todo el evento.</p>
      </div>
      
      <div class="expl-grid">
        <div class="expl-card">
          <div class="expl-card-num">01</div>
          <h3>Distancia de Gower</h3>
          <p>Para calcular la similitud entre participantes con datos mixtos (categóricos, numéricos y lógicos), el algoritmo calcula una matriz de distancias usando el coeficiente de Gower. Esto asegura que todas las variables influyan armónicamente.</p>
        </div>
        
        <div class="expl-card">
          <div class="expl-card-num">02</div>
          <h3>Calibración de Pesos</h3>
          <p>El algoritmo analiza la dispersión de las respuestas en el evento y calibra los pesos automáticamente. Las preguntas con mayor varianza tienen un mayor peso relativo, logrando una distribución equitativa de la diversidad en todas las mesas.</p>
        </div>
        
        <div class="expl-card">
          <div class="expl-card-num">03</div>
          <h3>Evitar Conflictos de Grupo</h3>
          <p>Una restricción dura del algoritmo es evitar que personas del mismo grupo de origen (por ejemplo, miembros de la misma empresa o delegación local) compartan la misma mesa, maximizando así la interacción con nuevos contactos.</p>
        </div>
        
        <div class="expl-card">
          <div class="expl-card-num">04</div>
          <h3>Optimización (Simulated Annealing)</h3>
          <p>Partiendo de una asignación inicial factible, el algoritmo evalúa continuamente millones de intercambios aleatorios entre mesas. Mediante "Templado Simulado" (Simulated Annealing), acepta temporalmente configuraciones peores para escapar de callejones sin salida y alcanzar el óptimo global.</p>
        </div>
      </div>

      <div class="expl-flow">
        <h3>El Flujo del Algoritmo</h3>
        <div class="flow-steps">
          <div class="flow-step">
            <div class="flow-step-icon">📄</div>
            <h4>Carga de Respuestas</h4>
            <p>Se sube el archivo CSV del formulario de inscripción.</p>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-step-icon">⚖️</div>
            <h4>Inferencia y Pesos</h4>
            <p>Se detectan los tipos de datos y se calibran sus pesos.</p>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-step-icon">🔄</div>
            <h4>Optimización</h4>
            <p>Simulated Annealing intercambia personas buscando la máxima diversidad.</p>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-step-icon">📊</div>
            <h4>Mesas y Reporte</h4>
            <p>Se generan las mesas óptimas y gráficos de diversidad.</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: SYNTHETIC DATA BUILDER -->
  <div id="tab-builder" class="tab-content" style="display:none;">
    <div class="builder-container">
      <div class="builder-sidebar">
        <div class="panel">
          <div class="panel-title"><span class="dot"></span> Configuración</div>
          <div class="field" style="margin-bottom:16px;">
            <label>Total de Personas</label>
            <input type="number" id="gen-people" value="80" min="10" max="500">
            <div class="field-hint">Asistentes sintéticos a generar</div>
          </div>
          <div class="field" style="margin-bottom:16px;">
            <label>Grupos de Origen</label>
            <input type="number" id="gen-groups" value="15" min="2" max="30">
            <div class="field-hint">Para la columna 'grupo'. 15 grupos en Córdoba Norte</div>
          </div>
          <div class="builder-actions" style="margin-top:20px;">
            <button id="gen-dl-btn" class="action-btn secondary-btn" onclick="downloadCSV()">Generar y Descargar CSV</button>
            <button id="gen-load-btn" class="action-btn primary-btn" onclick="loadInSimulator()">Cargar en el Simulador →</button>
          </div>
          <div id="gen-success-msg" class="success-box" style="display:none;">
            ✓ ¡Datos sintéticos cargados en el simulador!
          </div>
        </div>
      </div>
      
      <div class="builder-main">
        <div class="panel">
          <div class="panel-title" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span><span class="dot"></span> Formulario Mock</span>
            <button id="add-question-btn" class="add-btn" onclick="addQuestion()">+ Agregar Pregunta</button>
          </div>
          <div id="questions-list" class="questions-list">
            <!-- Questions rendered via JS -->
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 3: SIMULATOR (ORIGINAL SHIELDED PANELS) -->
  <div id="tab-sim" class="tab-content" style="display:none;">
    <!-- UPLOAD -->
  <div class="panel">
    <div class="panel-title"><span class="dot"></span> Cargar respuestas</div>
    <div id="drop-zone">
      <input type="file" id="file-input" accept=".csv">
      <div class="upload-icon">📂</div>
      <div class="upload-title">Arrastrá tu CSV acá</div>
      <div class="upload-sub">o hacé click para seleccionar el archivo</div>
      <div id="file-name"></div>
    </div>
    <div id="error-box"></div>
  </div>

  <!-- CONFIG -->
  <div class="panel">
    <div class="panel-title"><span class="dot" style="background:var(--acc3)"></span> Configuración</div>
    <div class="config-grid">
      <div class="field">
        <label>Personas por mesa</label>
        <input type="number" id="table-size" value="10" min="2" max="30">
        <div class="field-hint">Tamaño fijo de cada mesa</div>
      </div>
      <div class="field">
        <label>Iteraciones SA</label>
        <input type="number" id="sa-iters" value="60000" step="10000" min="10000" max="300000">
        <div class="field-hint">Más iteraciones = mejor resultado, más tiempo</div>
      </div>
    </div>
    <button id="run-btn" disabled>Calcular distribución de mesas</button>
    <div id="progress-wrap">
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" id="progress-fill"></div>
      </div>
      <div class="progress-msg" id="progress-msg">Iniciando…</div>
    </div>
  </div>

  <!-- RESULTS -->
  <div id="results">

    <!-- Stats -->
    <div class="section-header">Resumen</div>
    <div class="panel">
      <div class="stats-grid" id="stats-grid"></div>
      <a id="dl-btn" href="#" download="asignacion_mesas.csv">
        ↓ Descargar asignación (.csv)
      </a>
    </div>

    <!-- Mesa assignments -->
    <div class="section-header">Resultado por mesa</div>
    <div class="mesa-toolbar">
      <input type="text" class="mesa-search" id="mesa-search" placeholder="Buscar persona o grupo…">
      <span class="mesa-count" id="mesa-count"></span>
    </div>
    <div class="mesa-grid" id="mesa-grid"></div>

    <!-- Charts -->
    <div class="section-header">Visualizaciones</div>
    <div class="chart-wrap"><div id="chart-umap-mesa" style="height:400px"></div></div>
    <div class="chart-grid">
      <div class="chart-wrap"><div id="chart-heatmap"></div></div>
      <div class="chart-wrap"><div id="chart-diversidad" style="height:300px"></div></div>
    </div>

    <!-- Group table (collapsed) -->
    <div class="section-header">Detalle de grupos</div>
    <div class="panel" style="padding: 0; overflow: hidden;">
      <table class="group-table" id="group-table">
        <thead>
          <tr>
            <th>Grupo</th>
            <th>Asistentes</th>
            <th>Conflictos inevitables</th>
            <th>Estado</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>

  </div><!-- /results -->
  </div><!-- /tab-sim -->

</div><!-- /wrap -->

<script>
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const fileNameEl = document.getElementById('file-name');
const runBtn     = document.getElementById('run-btn');
const progressWrap = document.getElementById('progress-wrap');
const progressFill = document.getElementById('progress-fill');
const progressMsg  = document.getElementById('progress-msg');
const resultsEl    = document.getElementById('results');
const errorBox     = document.getElementById('error-box');

let selectedFile = null;
let currentJobId = null;
let pollTimer    = null;

// ── FILE HANDLING ──
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

function setFile(f) {
  selectedFile = f;
  fileNameEl.textContent = `📄 ${f.name}  (${(f.size/1024).toFixed(1)} KB)`;
  fileNameEl.style.display = 'block';
  runBtn.disabled = false;
  errorBox.style.display = 'none';
}

// ── RUN ──
runBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  clearPoll();
  runBtn.disabled = true;
  progressWrap.style.display = 'block';
  resultsEl.style.display = 'none';
  errorBox.style.display = 'none';
  setProgress(0, 'Enviando archivo…');

  const fd = new FormData();
  fd.append('archivo', selectedFile);
  fd.append('table_size', document.getElementById('table-size').value);
  fd.append('sa_iters', document.getElementById('sa-iters').value);

  let res;
  try {
    res = await fetch('/procesar', { method: 'POST', body: fd });
    res = await res.json();
  } catch(e) {
    showError('Error de red al enviar el archivo.');
    runBtn.disabled = false;
    return;
  }
  if (res.error) { showError(res.error); runBtn.disabled = false; return; }
  currentJobId = res.job_id;
  pollTimer = setInterval(pollStatus, 1200);
});

async function pollStatus() {
  if (!currentJobId) return;
  let res;
  try {
    res = await fetch(`/estado/${currentJobId}`);
    res = await res.json();
  } catch { return; }

  setProgress(res.progress, res.message);

  if (res.status === 'done') {
    clearPoll();
    showResults(res);
    runBtn.disabled = false;
  } else if (res.status === 'error') {
    clearPoll();
    showError(res.error || res.message);
    runBtn.disabled = false;
  }
}

function clearPoll() { clearInterval(pollTimer); pollTimer = null; }

function setProgress(pct, msg) {
  progressFill.style.width = pct + '%';
  progressMsg.textContent = msg;
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.style.display = 'block';
  progressWrap.style.display = 'none';
}

// ── RESULTS ──
let mesaData = [];

function showResults(data) {
  progressWrap.style.display = 'none';
  resultsEl.style.display = 'block';

  const s = data.stats;
  const statsGrid = document.getElementById('stats-grid');
  statsGrid.innerHTML = `
    ${stat(s.n, 'Asistentes', `${s.K} mesas · ${s.table_size} por mesa`, '#818cf8')}
    ${stat(s.conflicts_real + ' / ' + s.conflicts_inev, 'Conflictos', 'reales / inevitables',
           s.conflicts_real <= s.conflicts_inev ? '#34d399' : '#fb7185')}
    ${stat(s.diversity_mean.toFixed(3), 'Diversidad/mesa', `σ ${s.diversity_std}`, '#34d399')}
    ${stat(s.avg_groups_table, 'Grupos/mesa', `de ${s.n_groups} grupos`, '#fbbf24')}
  `;

  // Mesa cards
  mesaData = data.mesa_table || [];
  document.getElementById('mesa-count').textContent = `${mesaData.length} mesas`;
  renderMesaCards('');

  // Search
  const searchEl = document.getElementById('mesa-search');
  searchEl.value = '';
  searchEl.oninput = () => renderMesaCards(searchEl.value.trim().toLowerCase());

  // Group table
  const tbody = document.querySelector('#group-table tbody');
  tbody.innerHTML = (data.group_summary || []).sort((a,b) => b.n - a.n).map(g => `
    <tr>
      <td>${g.grupo}</td>
      <td>${g.n}</td>
      <td>${g.inevitables}</td>
      <td>${g.inevitables === 0
        ? '<span class="badge badge-ok">✓ OK</span>'
        : `<span class="badge badge-warn">⚠ ${g.inevitables}</span>`
      }</td>
    </tr>`).join('');

  // Download link
  const dlBtn = document.getElementById('dl-btn');
  dlBtn.href = `/descargar/${currentJobId}`;
  dlBtn.style.display = 'inline-block';

  // Charts
  const cfg = { displayModeBar: false, displaylogo: false };
  Plotly.react('chart-umap-mesa',  data.charts.umap_mesa.data,  data.charts.umap_mesa.layout,  cfg);
  Plotly.react('chart-heatmap',    data.charts.heatmap.data,    data.charts.heatmap.layout,    cfg);
  Plotly.react('chart-diversidad', data.charts.diversidad.data, data.charts.diversidad.layout, cfg);

  // Scroll suave a resultados
  setTimeout(() => resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

function renderMesaCards(filter) {
  const grid = document.getElementById('mesa-grid');
  let count = 0;
  grid.innerHTML = mesaData.map(m => {
    const personas = m.personas.filter(p =>
      !filter ||
      p.nombre.toLowerCase().includes(filter) ||
      p.grupo.toLowerCase().includes(filter)
    );
    if (filter && personas.length === 0) return '';
    count++;
    return `<div class="mesa-card">
      <div class="mesa-card-header">
        <span class="mesa-card-num">Mesa ${m.mesa}</span>
        <span class="mesa-card-count">${m.total} personas</span>
      </div>
      ${personas.map(p => `
        <div class="mesa-person">
          <span class="mesa-person-name">${p.nombre}</span>
          <span class="mesa-person-group">${p.grupo}</span>
        </div>
      `).join('')}
    </div>`;
  }).join('');
  document.getElementById('mesa-count').textContent = filter
    ? `${count} de ${mesaData.length} mesas`
    : `${mesaData.length} mesas`;
}

function stat(val, label, sub, color='#e2e8f0') {
  return `<div class="stat">
    <div class="stat-label">${label}</div>
    <div class="stat-val" style="color:${color}">${val}</div>
    <div class="stat-sub">${sub}</div>
  </div>`;
}

// ── TABS NAVIGATION LOGIC ──
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.getElementById(tabId).style.display = 'block';
  
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  
  if (tabId === 'tab-expl') document.getElementById('btn-tab-expl').classList.add('active');
  if (tabId === 'tab-builder') document.getElementById('btn-tab-builder').classList.add('active');
  if (tabId === 'tab-sim') document.getElementById('btn-tab-sim').classList.add('active');
  
  window.dispatchEvent(new Event('resize'));
}

// ── MOCK QUESTION BUILDER LOGIC ──
let questions = [
  { id: 1, name: '¿Te interesa la adopción de nuevas tecnologías?', type: 'binary', options: '' },
  { id: 2, name: 'Actividad principal en el campo', type: 'nominal', options: 'Ganadería, Agricultura, Contratista, Mixto' },
  { id: 3, name: 'Foco en sustentabilidad / ambiente', type: 'ordinal', options: '' },
  { id: 4, name: 'Edad del productor', type: 'numeric', options: '' }
];
let nextQId = 5;

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function renderQuestions() {
  const container = document.getElementById('questions-list');
  if (!container) return;
  container.innerHTML = '';
  
  questions.forEach((q, idx) => {
    const card = document.createElement('div');
    card.className = 'question-card';
    card.dataset.id = q.id;
    
    const isNominal = q.type === 'nominal';
    
    card.innerHTML = `
      <div class="question-card-header">
        <span class="question-num">Pregunta ${idx + 1}</span>
        <button class="remove-q-btn" onclick="removeQuestion(${q.id})">Eliminar</button>
      </div>
      <div class="question-fields">
        <div class="field">
          <label>Pregunta / Enunciado</label>
          <input type="text" class="q-name" value="${escapeHtml(q.name)}" oninput="updateQuestionName(${q.id}, this.value)" placeholder="Ej. ¿Cuántas hectáreas produce?">
        </div>
        <div class="field">
          <label>Tipo de Respuesta</label>
          <select class="q-type" onchange="updateQuestionType(${q.id}, this.value)">
            <option value="binary" ${q.type === 'binary' ? 'selected' : ''}>Binario (Sí/No)</option>
            <option value="nominal" ${q.type === 'nominal' ? 'selected' : ''}>Nominal (Categorías)</option>
            <option value="ordinal" ${q.type === 'ordinal' ? 'selected' : ''}>Ordinal (Escala 1-5)</option>
            <option value="numeric" ${q.type === 'numeric' ? 'selected' : ''}>Numérico (Valor libre)</option>
          </select>
        </div>
      </div>
      <div class="field options-field" id="options-${q.id}" style="display: ${isNominal ? 'block' : 'none'}">
        <label>Opciones (separadas por coma)</label>
        <input type="text" class="q-options" value="${escapeHtml(q.options)}" oninput="updateQuestionOptions(${q.id}, this.value)" placeholder="Ej. Opción A, Opción B, Opción C">
        <div class="field-hint">Ingresá las categorías que tomará la respuesta aleatoria.</div>
      </div>
    `;
    container.appendChild(card);
  });
}

function updateQuestionName(id, val) {
  const q = questions.find(item => item.id === id);
  if (q) q.name = val;
}

function updateQuestionType(id, val) {
  const q = questions.find(item => item.id === id);
  if (q) {
    q.type = val;
    const optField = document.getElementById(`options-${id}`);
    if (optField) {
      optField.style.display = val === 'nominal' ? 'block' : 'none';
    }
  }
}

function updateQuestionOptions(id, val) {
  const q = questions.find(item => item.id === id);
  if (q) q.options = val;
}

function addQuestion() {
  questions.push({
    id: nextQId++,
    name: '',
    type: 'binary',
    options: ''
  });
  renderQuestions();
}

function removeQuestion(id) {
  questions = questions.filter(item => item.id !== id);
  renderQuestions();
}

// ── SYNTHETIC DATA GENERATION ──
function generateSyntheticCSV(questions, numPeople, numGroups) {
  let headers = ['id', 'grupo'];
  questions.forEach((q, idx) => {
    headers.push(q.name.trim() || `Pregunta ${idx + 1}`);
  });
  
  let rows = [headers.join(',')];
  
  for (let i = 0; i < numPeople; i++) {
    let row = [];
    row.push(`Participante ${i + 1}`);
    
    let grpName = numGroups <= 26 
      ? `Grupo ${String.fromCharCode(65 + (i % numGroups))}`
      : `Grupo ${i % numGroups + 1}`;
    row.push(grpName);
    
    questions.forEach((q) => {
      let val = '';
      if (q.type === 'binary') {
        val = Math.random() > 0.5 ? 'Sí' : 'No';
      } else if (q.type === 'nominal') {
        let opts = q.options.split(',').map(s => s.trim()).filter(Boolean);
        if (opts.length === 0) opts = ['Opción A', 'Opción B', 'Opción C'];
        val = opts[Math.floor(Math.random() * opts.length)];
      } else if (q.type === 'ordinal') {
        val = Math.floor(Math.random() * 5) + 1;
      } else if (q.type === 'numeric') {
        val = 20 + Math.floor(Math.random() * 51); // 20 a 70
      }
      
      if (typeof val === 'string' && (val.includes(',') || val.includes('"') || val.includes('\n'))) {
        val = '"' + val.replace(/"/g, '""') + '"';
      }
      row.push(val);
    });
    
    rows.push(row.join(','));
  }
  
  return rows.join('\n');
}

function downloadCSV() {
  const numPeople = parseInt(document.getElementById('gen-people').value, 10) || 80;
  const numGroups = parseInt(document.getElementById('gen-groups').value, 10) || 8;
  
  const csvContent = generateSyntheticCSV(questions, numPeople, numGroups);
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', 'respuestas_sinteticas.csv');
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Global setFile function reference
const originalSetFile = setFile;

function loadInSimulator() {
  const numPeople = parseInt(document.getElementById('gen-people').value, 10) || 80;
  const numGroups = parseInt(document.getElementById('gen-groups').value, 10) || 8;
  
  const csvContent = generateSyntheticCSV(questions, numPeople, numGroups);
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  
  const file = new File([blob], 'respuestas_sinteticas.csv', { type: 'text/csv' });
  originalSetFile(file);
  
  const successBox = document.getElementById('gen-success-msg');
  successBox.style.display = 'block';
  
  setTimeout(() => {
    successBox.style.display = 'none';
    switchTab('tab-sim');
  }, 1000);
}

// Initial render
renderQuestions();
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("=" * 52)
    print("  CREA · Anticlustering para eventos")
    print("=" * 52)
    print("  → http://localhost:5000")
    print("  Presioná Ctrl+C para detener el servidor")
    print("=" * 52)
    app.run(debug=False, port=5000, threaded=True)
"""
================================================================================
 PIPELINE — Field Cancerization via DNA Methylation (IWBBIO 2026 paper)
================================================================================

Riorganizzazione in funzioni del notebook originale `05-4iwbbio-results.ipynb`.

Ogni STEP della pipeline e' una funzione che:
  - riceve in input gli oggetti prodotti dagli step precedenti (o li carica da
    disco se sono gia' stati salvati in una run precedente),
  - esegue esattamente la stessa logica della cella originale del notebook,
  - salva i suoi output intermedi su disco (cartella ARTIFACTS_DIR),
  - ritorna un dict con gli oggetti necessari agli step successivi.

SCOPO DI QUESTO SCRIPT
-----------------------
Permettere di variare facilmente i parametri dello STEP 3 (score statistico e
score biologico: W_BIO, W_IP, EPS_IP, STABILITY_SIGN_MIN, K_FOLDS, R_REPEATS, ...)
e rieseguire SOLO la parte di pipeline che dipende da quei parametri, senza
dover ripetere da capo i passaggi costosi e deterministici che vengono prima
(caricamento beta, ComBat, ecc.).

Per farlo:
  1) esegui `run_until_step3_inputs()` UNA VOLTA: produce tutto cio' che serve
     come input allo STEP 3 (M_train, y_train, cpg_cols_current, bio_weight_norm,
     X_tum_combat, ...) e lo salva su disco (cache).
  2) poi chiama `run_step3_and_downstream(params)` dentro un ciclo for, variando
     i parametri che vuoi: questa funzione ricarica la cache (rapido) e riesegue
     SOLO dallo STEP 3 in poi (RSKF, region anchoring, correlation clustering,
     diversificazione, knapsack, SVM, FPI).

Vedi la funzione `main()` in fondo al file per un esempio completo, e la
funzione `example_parameter_sweep()` per l'esempio di ciclo for sui parametri
dello score stat/bio.

NOTE IMPORTANTI SU FEDELTA' AL NOTEBOOK ORIGINALE
---------------------------------------------------
- STEP 3 usa la versione GPU/CuPy del notebook (Mann-Whitney vettorizzato su
  GPU). Questo script DEVE essere eseguito su una macchina con GPU NVIDIA e
  CuPy installato (es. Kaggle con accelerator GPU attivo). Non c'e' fallback
  CPU per scelta esplicita.
- Per il calcolo finale del Field Progression Index (FPI) e dello score
  biologico s_bio sulle CpG finali, il notebook originale conteneva DUE celle
  quasi identiche ma non equivalenti: una gestisce correttamente la conversione
  M-value -> beta per il tessuto Tumour prima di calcolare la deviazione
  assoluta, l'altra (l'ultima cella del notebook) salta questo passaggio e
  produrrebbe un FPI artificialmente vicino a 0. Questo script usa la versione
  CORRETTA.
- E' stato individuato e corretto un bug del notebook originale nella cella
  "STEP 6: Train SVM sulle CpG di final_cpgs": il dizionario `col_to_j` usato
  per indicizzare `X_tr_combat`/`X_te_combat` (che vivono nello spazio delle
  485k CpG comuni, `common_cpgs`) veniva in realta' costruito sulle ~270k CpG
  post-Edgar/variance-filter (`cpg_cols_current`), cioe' in uno spazio di
  colonne diverso. Questo script usa SEMPRE una mappa costruita su
  `common_cpgs` per indicizzare `X_tr_combat`/`X_te_combat`/`X_te_combat`.

================================================================================
"""

from __future__ import annotations

import os
import re
import gc
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu
from scipy.special import ndtr

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    classification_report,
    RocCurveDisplay,
    silhouette_score,
)
from sklearn.utils import resample

import pulp

warnings.filterwarnings("ignore")

# CuPy e' obbligatorio: lo STEP 3 di questo script usa SOLO la versione GPU.
# Se non e' disponibile, lo script si ferma subito con un errore chiaro,
# invece di proseguire silenziosamente con un fallback CPU.
import cupy as cp


# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class PathsConfig:
    """Path di input (dataset Kaggle) e di output. Modifica questi path
    secondo dove girerai lo script (Kaggle, o macchina locale con gli stessi
    file scaricati)."""

    # --- Beta matrices (parquet) ---
    BETA_PATHS: dict = field(default_factory=lambda: {
        "GSE225845": "/kaggle/input/datasets/rovieramariella/gse225845-parquet/GSE225845.parquet",
        "GSE287331": "/kaggle/input/datasets/rovieramariella/3-gse287331-parquet/GSE287331_clean_imputed.parquet",
    })

    # --- Phenotype (parquet) ---
    PHENO_PATHS: dict = field(default_factory=lambda: {
        "GSE225845": "/kaggle/input/datasets/rovieramariella/pheno-gse225845-with-true-age-bins/pheno_GSE225845_with_true_age_bins.parquet",
        "GSE287331": "/kaggle/input/datasets/rovieramariella/pheno-gse287331-with-age-bin/pheno_GSE287331_with_age_bin.parquet",
    })

    # --- Age (Horvath DNAmAge, csv) ---
    AGE_PATHS: dict = field(default_factory=lambda: {
        "GSE287331": "/kaggle/input/datasets/rovieramariella/output-gse287331-horvathdnamage-normal-adj-only-cs/Output_GSE287331_HorvathDNAmAge_normal_adj_only.csv",
    })

    # --- Blacklist CpG (csv, una per dataset) ---
    BLACKLIST_PATHS: dict = field(default_factory=lambda: {
        "GSE225845": "/kaggle/input/datasets/rovieramariella/removed-cpgs-all-filters-summary-gse225845/removed_cpgs_all_filters_summary_gse225845.csv",
        "GSE287331": "/kaggle/input/datasets/rovieramariella/removed-cpgs-all-filters-summary-gse287331/removed_cpgs_all_filters_summary_gse287331.csv",
    })
    BLACKLIST_CPG_COL: str = "CpG_ID"

    # --- Manifest Illumina EPIC (CpG -> gene/island/region) ---
    MANIFEST_PATH: str = "/kaggle/input/datasets/rovieramariella/manifest-infinium-methylationepic-cpg-to-gene/infinium-methylationepic-v-1-0-b5-manifest-file.csv"

    # --- COSMIC Cancer Gene Census ---
    CGC_PATH: str = "/kaggle/input/datasets/rovieramariella/cosmic-cancergenecensus-v103-grch38-tsv/Cosmic_CancerGeneCensus_v103_GRCh38.tsv"

    # --- Output ---
    OUTPUT_DIR: str = "/kaggle/working/"
    ARTIFACTS_DIR: str = "/kaggle/working/artifacts"  # cache per la pipeline a step


@dataclass
class GlobalConfig:
    """Costanti globali identiche al notebook originale."""
    SAMPLE_ID: str = "id_tissue"
    LABEL_COL: str = "label"
    RANDOM_STATE: int = 42
    REPRODUCE_NOTEBOOK_BUG: bool = True
    FPI_USE_CORRECTED_VERSION: bool = False


@dataclass
class Step1Config:
    """STEP 1 — Edgar(beta) feature selection (train-only variability filter)."""
    R_BETA_CUTOFF: float = 0.05
    LABELS_USE: tuple = (0, 1)


@dataclass
class Step2Config:
    """STEP 2 — beta -> M transform + variance filter (train-only)."""
    EPS_M: float = 1e-6
    VAR_DROP_FRAC: float = 0.05
    APPLY_VAR_FILTER: bool = True


@dataclass
class Step3Config:
    """STEP 3 — Repeated Stratified K-Fold stability ranking (score stat + bio).

    QUESTI sono i parametri che vuoi variare per vedere come cambiano i
    risultati finali (CpG selezionate, performance SVM, FPI, ...).
    """
    K_FOLDS: int = 5
    R_REPEATS: int = 10
    STABILITY_SIGN_MIN: float = 0.75
    PREFILTER_TOP_DM: int = 50_000          # tutte le CpG (nessun prefilter reale)
    MIN_CANDIDATES_TARGET: int = 5_000
    CHUNK_SIZE: int = 10_000                # colonne per chunk GPU (evita OOM)

    # --- pesi dello score finale (score stat vs score bio) ---
    W_BIO: float = 0.50      # peso del blocco "score_bio" nello score finale
    W_IP: float = 0.50       # dentro score_bio: peso di IP vs bio_weight genomico
    EPS_IP: float = 1e-4     # epsilon per evitare divisioni per ~0 nel calcolo di IP


@dataclass
class Step3bConfig:
    """STEP 3b — Region anchoring su CpG islands."""
    ANCHOR_CONTEXT: frozenset = frozenset({"Island"})
    FRACTION_STABLE_MIN: float = 0.60
    MIN_CPGS_PER_REGION: int = 2
    TOP_M_PER_REGION: int = 3
    MANIFEST_CPG_COL: str = "IlmnID"
    MANIFEST_ISLAND_NAME_COL: str = "UCSC_CpG_Islands_Name"
    MANIFEST_REL_COL: str = "Relation_to_UCSC_CpG_Island"


@dataclass
class Step4Config:
    """STEP 4 — Correlation clustering (Pool A / Pool B separati)."""
    CORR_CLUSTER_THR: float = 0.85


@dataclass
class Step5Config:
    """STEP 5 — Diversificazione greedy con vincoli soft + relaxation."""
    K_TARGET: int = 5000
    CHR_MAX_FRAC: float = 0.08
    CTX_MAX_FRAC: float = 0.65
    WIN_BP: int = 500_000
    MAX_PER_WIN: int = 15
    MANIFEST_CPG_COL: str = "IlmnID"
    MANIFEST_CHR_COL: str = "CHR"
    MANIFEST_POS_COL: str = "MAPINFO"
    MANIFEST_CTX_COL: str = "Relation_to_UCSC_CpG_Island"


@dataclass
class Step6Config:
    """STEP 6 — Selezione finale K CpG (presa diretta dall'ordine di STEP 5)."""
    K_FINAL: int = 5000


@dataclass
class SVMConfig:
    """Training/valutazione SVM sulle CpG di final_cpgs."""
    C_GRID: tuple = (0.001, 0.01, 0.1, 1.0, 10.0)
    CV_FOLDS: int = 5
    CALIBRATION_CV: int = 3
    MAX_ITER: int = 5000
    N_BOOT: int = 2000
    ALPHA: float = 0.05


@dataclass
class KnapsackConfig:
    """STEP knapsack — selezione MILP bi-obiettivo con sweep su mu (v4 paper)."""
    K: int = 50
    N_POOL: int = 5000
    CHR_MAX: int = 10
    CORR_THR: float = 0.85
    ETA: float = 0.5

    W_COEF: float = 0.5
    W_DM: float = 0.2
    W_SCORE: float = 0.3
    W_COSMIC: float = 0.3
    W_COSMIC_GRID: tuple = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

    MAX_PER_GENE: int = 2
    MU_GRID: tuple = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    K_PARETO_GRID: tuple = (20, 30, 40, 50, 60, 70, 80)
    N_FOLDS_STAB: int = 5
    CBC_TIME_LIMIT: int = 600

    EPS_M: float = 1e-6


@dataclass
class FPIConfig:
    """STEP finale — Field Progression Index (Eq. 5) + score biologico s_bio (Eq. 4)."""
    EPS_FPI: float = 1e-6
    EPS_M: float = 1e-6
    K_FOR_FPI: int = 30   # quante CpG finali (dal Pareto del knapsack) usare per l'FPI


@dataclass
class PipelineConfig:
    """Aggregatore di tutta la configurazione. Passa un'istanza di questa
    classe (eventualmente con dei campi modificati) alle funzioni `run_*`."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    glob: GlobalConfig = field(default_factory=GlobalConfig)
    step1: Step1Config = field(default_factory=Step1Config)
    step2: Step2Config = field(default_factory=Step2Config)
    step3: Step3Config = field(default_factory=Step3Config)
    step3b: Step3bConfig = field(default_factory=Step3bConfig)
    step4: Step4Config = field(default_factory=Step4Config)
    step5: Step5Config = field(default_factory=Step5Config)
    step6: Step6Config = field(default_factory=Step6Config)
    svm: SVMConfig = field(default_factory=SVMConfig)
    knapsack: KnapsackConfig = field(default_factory=KnapsackConfig)
    fpi: FPIConfig = field(default_factory=FPIConfig)


# ==============================================================================
# HELPERS GENERICI (identici al notebook, CELLA 2 / "Helpers")
# ==============================================================================

_NON_CPG = {"id_tissue", "label", "sample_id", "barcode", "index"}


def require_cols(df, cols, tag):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{tag} missing columns: {missing}")


def parse_idat_basename(idat):
    if idat is None or (isinstance(idat, float) and pd.isna(idat)):
        return (None, None)
    s = str(idat).strip()
    if not s:
        return (None, None)
    m = re.search(r"(\d{8,})\s*[_-]?\s*(R\d{2}C\d{2})", s)
    if not m:
        return (None, None)
    return (m.group(1), m.group(2))


def read_horvath_age_csv(path):
    df = pd.read_csv(path).copy()
    cols_lc = {c.lower(): c for c in df.columns}
    if "id_tissue" not in df.columns:
        if "sampleid" in cols_lc:
            df = df.rename(columns={cols_lc["sampleid"]: "id_tissue"})
        else:
            raise KeyError(f"Age file {path} missing id column.")
    if "age" not in df.columns:
        if "dnamage" in cols_lc:
            df = df.rename(columns={cols_lc["dnamage"]: "age"})
        else:
            raise KeyError(f"Age file {path} missing age column.")
    df["id_tissue"] = df["id_tissue"].astype(str)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df[["id_tissue", "age"]].drop_duplicates("id_tissue")


def get_cpg_cols(df: pl.DataFrame) -> list:
    return [c for c in df.columns if c not in _NON_CPG]


def read_illumina_manifest_csv(path: str) -> pd.DataFrame:
    """Legge il manifest Illumina EPIC saltando le righe di header non-CSV
    iniziali (cerca la riga che contiene la colonna 'IlmnID')."""
    header_idx = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            parts = [p.strip() for p in line.split(",")]
            if "IlmnID" in parts:
                header_idx = i
                break
    if header_idx is None:
        raise ValueError("Manifest header not found (no 'IlmnID' line detected).")

    df = pd.read_csv(
        path, sep=",", header=header_idx, engine="python", on_bad_lines="skip"
    )
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    return df


def load_manifest_with_real_header(path: str) -> pd.DataFrame:
    """Variante usata negli step di region-anchoring / diversificazione."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        start_row = None
        for i, line in enumerate(f):
            if line.startswith("IlmnID"):
                start_row = i
                break
    if start_row is None:
        raise ValueError(f"Non trovo header 'IlmnID' in: {path}")
    return pd.read_csv(path, skiprows=start_row, low_memory=False)


def beta_to_m(beta_arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    b = np.clip(beta_arr, 0.0, 1.0)
    return np.log2((b + eps) / (1.0 - b + eps))


def m_to_beta(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inversa di beta_to_m: M = log2((b+eps)/(1-b+eps))  =>
    b = (2^M*(1+eps) - eps) / (1+2^M)"""
    m = np.asarray(m, dtype=np.float64)
    p2m = np.power(2.0, m)
    b = (p2m * (1.0 + eps) - eps) / (1.0 + p2m)
    return np.clip(b, 0.0, 1.0)


def apply_thesis_style(use_tex: bool = True,
                        legend_position: str = "upper right",
                        legend_outside: bool = False):
    """Stile uniforme (Matplotlib + Seaborn) coerente con la tesi/paper.
    Definisce anche `place_legend()` nel modulo per posizionare le legende.
    Identica al notebook originale (CELLA "THESIS STYLE FOR PLOT")."""

    def _normalize_pos(pos: str) -> str:
        if not isinstance(pos, str):
            return "upper right"
        key = pos.strip().lower().replace("top", "upper").replace("bottom", "lower")
        mapping = {
            "upper right": "upper right", "upper left": "upper left",
            "lower right": "lower right", "lower left": "lower left",
            "center": "center", "best": "best",
        }
        return mapping.get(key, "upper right")

    _default_loc = _normalize_pos(legend_position)

    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "text.usetex": bool(use_tex),
        "figure.figsize": (6.8, 4.5),
        "font.size": 8.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9.5,
        "legend.title_fontsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.facecolor": "white",
        "axes.edgecolor": "#D0D0D0",
        "axes.linewidth": 0.8,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#D0D0D0",
        "legend.loc": _default_loc,
        "legend.framealpha": 1.0,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.6,
        "legend.borderpad": 0.4,
        "legend.borderaxespad": 0.8,
    })


DS_PALETTE = {"GSE225845": "#1b9e77", "GSE287331": "#d95f02"}
DS_PALETTE_VIRIDIS = {0: plt.cm.viridis(0.15), 1: plt.cm.viridis(0.75)}


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ==============================================================================
# STEP 0 — Build meta per i due dataset + Train/Test split stratificato
# ==============================================================================

def step0_build_meta(cfg: PipelineConfig) -> dict:
    """CELLA 3 — Build meta per tutti i dataset.
    Ritorna dict con: meta (dict per ds), meta_225, meta_287, meta_tumor.
    """
    P, G = cfg.paths, cfg.glob
    meta = {}
    meta_tumor_list = []

    for ds in ["GSE225845", "GSE287331"]:
        print("\n" + "=" * 80)
        print(f"BUILD META — {ds}")
        print("=" * 80)

        beta_meta = (
            pl.scan_parquet(P.BETA_PATHS[ds])
            .select([G.SAMPLE_ID, G.LABEL_COL])
            .collect()
            .to_pandas()
        )
        beta_meta[G.SAMPLE_ID] = beta_meta[G.SAMPLE_ID].astype(str)

        pheno = pd.read_parquet(P.PHENO_PATHS[ds]).copy()
        require_cols(pheno, [G.SAMPLE_ID], tag=f"[{ds}] pheno")
        pheno[G.SAMPLE_ID] = pheno[G.SAMPLE_ID].astype(str)

        if "idat_basename" in pheno.columns:
            tech = pheno["idat_basename"].apply(
                lambda x: pd.Series(parse_idat_basename(x), index=["SentrixID", "ChipPosition"])
            )
            pheno = pd.concat([pheno, tech], axis=1)
        else:
            if "sentrix_id" in pheno.columns:
                pheno["SentrixID"] = pheno["sentrix_id"].astype(str)
            elif "slide_id" in pheno.columns:
                pheno["SentrixID"] = pheno["slide_id"].astype(str)
            else:
                pheno["SentrixID"] = None
            pheno["ChipPosition"] = None
            pheno["idat_basename"] = None

        if ds == "GSE225845":
            require_cols(pheno, ["age_at_surgery"], tag=f"[{ds}] pheno (age)")
            pheno["age"] = pd.to_numeric(pheno["age_at_surgery"], errors="coerce")
        else:
            age_df = read_horvath_age_csv(P.AGE_PATHS[ds])
            pheno = pheno.merge(age_df, on="id_tissue", how="left")

        keep = ["id_tissue", "idat_basename", "SentrixID", "ChipPosition", "age"]
        pheno_small = pheno[[c for c in keep if c in pheno.columns]].drop_duplicates("id_tissue")

        meta_df = beta_meta.merge(pheno_small, on="id_tissue", how="left")

        meta_tumor_ds = meta_df[meta_df[G.LABEL_COL] == 2].copy()
        meta_tumor_ds["dataset_origin"] = ds
        meta_tumor_list.append(meta_tumor_ds)

        meta_df = meta_df[meta_df[G.LABEL_COL].isin([0, 1])].copy()
        meta_df.reset_index(drop=True, inplace=True)

        age_median = meta_df["age"].median()
        n_age_na = meta_df["age"].isna().sum()
        meta_df["age"] = meta_df["age"].fillna(age_median)

        meta_df["dataset_origin"] = ds

        n = len(meta_df)
        print(f"  n samples (N+A): {n}")
        print(f"  label dist:      {meta_df[G.LABEL_COL].value_counts().to_dict()}")
        print(f"  age NA imputati: {n_age_na}/{n}  (mediana={age_median:.1f})")
        print(f"  SentrixID unici: {meta_df['SentrixID'].nunique()}")

        meta[ds] = meta_df

    meta_225 = meta["GSE225845"]
    meta_287 = meta["GSE287331"]
    meta_tumor = pd.concat(meta_tumor_list, ignore_index=True)

    print("\nDONE — meta_225, meta_287 in scope.")
    print(f"Tumor samples: {len(meta_tumor)}")
    print(meta_tumor["dataset_origin"].value_counts())

    return {"meta": meta, "meta_225": meta_225, "meta_287": meta_287, "meta_tumor": meta_tumor}


def step0_train_test_split(cfg: PipelineConfig, meta_225: pd.DataFrame,
                            meta_287: pd.DataFrame) -> dict:
    """CELLA 4 — STEP 0: Train/Test split stratificato (PRIMA DI TUTTO)."""
    G, P = cfg.glob, cfg.paths

    meta_all = pd.concat([meta_225, meta_287], ignore_index=True)
    meta_all["stratum"] = meta_all["dataset_origin"] + "_" + meta_all[G.LABEL_COL].map({0: "N", 1: "A"})

    print("Distribuzione strati:")
    print(meta_all["stratum"].value_counts().sort_index())

    idx_all = np.arange(len(meta_all))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.30,
        stratify=meta_all["stratum"].values,
        random_state=G.RANDOM_STATE,
    )

    meta_train = meta_all.iloc[idx_train].reset_index(drop=True)
    meta_test = meta_all.iloc[idx_test].reset_index(drop=True)

    print(f"\nTrain: {len(meta_train)} samples")
    print(meta_train["stratum"].value_counts().sort_index())
    print(f"\nTest:  {len(meta_test)} samples")
    print(meta_test["stratum"].value_counts().sort_index())

    os.makedirs(P.OUTPUT_DIR, exist_ok=True)
    meta_train.to_parquet(os.path.join(P.OUTPUT_DIR, "meta_train.parquet"), index=False)
    meta_test.to_parquet(os.path.join(P.OUTPUT_DIR, "meta_test.parquet"), index=False)

    train_ids = set(meta_train[G.SAMPLE_ID].tolist())
    test_ids = set(meta_test[G.SAMPLE_ID].tolist())
    assert len(train_ids & test_ids) == 0, "LEAKAGE: campioni in comune tra train e test!"
    print("\nSplit OK — nessun leakage.")

    return {"meta_train": meta_train, "meta_test": meta_test}


# ==============================================================================
# STEP 1 — Intersezione CpG comuni (3-way) + blacklist
# ==============================================================================

def step1_common_cpgs(cfg: PipelineConfig) -> dict:
    """CELLA 5 — STEP 1: Intersezione CpG comuni tra i dataset, meno blacklist."""
    P = cfg.paths

    print("Lettura schema beta con polars lazy...")
    cpg_sets = {}
    for ds, path in P.BETA_PATHS.items():
        all_cols = pl.scan_parquet(path).columns
        cpg_sets[ds] = {c for c in all_cols if c not in _NON_CPG}
        print(f"  {ds}: {len(cpg_sets[ds]):,} CpG")

    common_cpgs_set = cpg_sets["GSE225845"] & cpg_sets["GSE287331"]
    print(f"\nCpG comuni (3-way, pre-blacklist): {len(common_cpgs_set):,}")

    blacklist_union = set()
    for ds, bl_path in P.BLACKLIST_PATHS.items():
        bl = pl.read_csv(bl_path).select(
            pl.col(P.BLACKLIST_CPG_COL).cast(pl.Utf8)
        ).unique().to_series().to_list()
        bl_set = set(bl)
        overlap = bl_set & common_cpgs_set
        print(f"  blacklist {ds}: {len(bl_set):,} totali, {len(overlap):,} nell'intersezione")
        blacklist_union |= bl_set
    print(f"\nBlacklist union: {len(blacklist_union):,} CpG unici")

    cpgs_after_blacklist = common_cpgs_set - blacklist_union
    print(f"CpG dopo rimozione blacklist: {len(cpgs_after_blacklist):,}")
    print(f"CpG rimossi dalla blacklist : {len(common_cpgs_set) - len(cpgs_after_blacklist):,}")

    ref_order = [c for c in pl.scan_parquet(P.BETA_PATHS["GSE225845"]).columns
                 if c not in _NON_CPG]
    common_cpgs = [c for c in ref_order if c in cpgs_after_blacklist]
    print(f"\nCpG finali (ordine GSE225845): {len(common_cpgs):,}")

    os.makedirs(P.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(P.OUTPUT_DIR, "common_cpgs_3way_filtered.txt"), "w") as f:
        f.write("\n".join(common_cpgs))
    print("Salvato: common_cpgs_3way_filtered.txt")

    return {"common_cpgs": common_cpgs}


# ==============================================================================
# STEP 1b — Carica beta ristrette ai CpG comuni, allinea a train/test/tumor
# ==============================================================================

def step1b_load_beta(cfg: PipelineConfig, common_cpgs: list,
                      meta_train: pd.DataFrame, meta_test: pd.DataFrame,
                      meta_tumor: pd.DataFrame) -> dict:
    """CELLA 6 — STEP 1b: Carica beta ristrette ai CpG comuni e allinea ai meta."""
    P, G = cfg.paths, cfg.glob

    def load_and_align_beta(ds: str, ids_ordered: list) -> np.ndarray:
        beta = pl.read_parquet(P.BETA_PATHS[ds]).select(["id_tissue"] + common_cpgs)
        beta = beta.with_columns(pl.col("id_tissue").cast(pl.Utf8))

        order_df = pl.DataFrame({
            "id_tissue": ids_ordered,
            "_order": list(range(len(ids_ordered)))
        })
        beta = (
            beta.join(order_df, on="id_tissue", how="inner")
            .sort("_order")
            .drop(["id_tissue", "_order"])
        )
        assert beta.shape[0] == len(ids_ordered), \
            f"[{ds}] attesi {len(ids_ordered)} campioni, trovati {beta.shape[0]}"
        return beta.to_numpy().astype(np.float32)

    def get_ids(meta_df, ds):
        return meta_df[meta_df["dataset_origin"] == ds][G.SAMPLE_ID].tolist()

    print("Caricamento beta train...")
    X_train = {ds: load_and_align_beta(ds, get_ids(meta_train, ds))
               for ds in ["GSE225845", "GSE287331"]}

    print("Caricamento beta test...")
    X_test = {ds: load_and_align_beta(ds, get_ids(meta_test, ds))
              for ds in ["GSE225845", "GSE287331"]}

    for ds in ["GSE225845", "GSE287331"]:
        print(f"  {ds} — train: {X_train[ds].shape}, test: {X_test[ds].shape}")

    print("Caricamento beta tumor (reference-only per IP score)...")
    X_tumor = {ds: load_and_align_beta(ds, get_ids(meta_tumor, ds))
               for ds in ["GSE225845", "GSE287331"]}
    print(f"  GSE225845 tumor: {X_tumor['GSE225845'].shape}")
    print(f"  GSE287331 tumor: {X_tumor['GSE287331'].shape}")

    return {"X_train": X_train, "X_test": X_test, "X_tumor": X_tumor}


# ==============================================================================
# STEP 2+3 (preprocessing) — Deviazione assoluta rispetto a mean Normal (train)
# ==============================================================================

def step2_absolute_deviation(cfg: PipelineConfig, meta_train: pd.DataFrame,
                              X_train: dict, X_test: dict, X_tumor: dict) -> dict:
    """CELLA 7 — STEP 2+3: Calcola mean Normal sul train -> trasforma in
    deviazione assoluta (train, test, tumor)."""
    G = cfg.glob
    mean_normal_train = {}

    for ds in ["GSE225845", "GSE287331"]:
        labels_tr = meta_train[meta_train["dataset_origin"] == ds][G.LABEL_COL].values
        X_tr = X_train[ds]
        normal_mask = labels_tr == 0
        assert normal_mask.sum() > 0, f"[{ds}] nessun campione Normal nel train!"
        mean_normal_train[ds] = X_tr[normal_mask].mean(axis=0)
        print(f"  {ds}: mean Normal calcolata su {normal_mask.sum()} campioni Normal train")

    X_train_abs = {ds: np.abs(X_train[ds] - mean_normal_train[ds])
                   for ds in ["GSE225845", "GSE287331"]}
    X_test_abs = {ds: np.abs(X_test[ds] - mean_normal_train[ds])
                  for ds in ["GSE225845", "GSE287331"]}

    print("\nTrasformazione deviazione assoluta completata.")
    print("Verifica: mean deviation Adjacent > Normal nel train?")
    for ds in ["GSE225845", "GSE287331"]:
        labels_tr = meta_train[meta_train["dataset_origin"] == ds][G.LABEL_COL].values
        mean_adj = X_train_abs[ds][labels_tr == 1].mean()
        mean_nor = X_train_abs[ds][labels_tr == 0].mean()
        print(f"  {ds}: mean_dev(Adjacent)={mean_adj:.4f}  mean_dev(Normal)={mean_nor:.4f}  "
              f"OK={mean_adj > mean_nor}")

    X_tumor_abs = {ds: np.abs(X_tumor[ds] - mean_normal_train[ds])
                   for ds in ["GSE225845", "GSE287331"]}

    return {
        "mean_normal_train": mean_normal_train,
        "X_train_abs": X_train_abs,
        "X_test_abs": X_test_abs,
        "X_tumor_abs": X_tumor_abs,
    }


# ==============================================================================
# STEP ComBat — batch correction (neuroCombat) su pool train, poi su test/tumor
# ==============================================================================

def step_combat(cfg: PipelineConfig, common_cpgs: list, meta_train: pd.DataFrame,
                 X_train_abs: dict, X_test_abs: dict, X_tumor_abs: dict) -> dict:
    """CELLA 9 — ComBat sul pool train + CELLA 10 — applica a test e tumor.

    Richiede il package `neuroCombat` installato (vedi README/istruzioni di
    setup). Definisce anche `neuroCombatFromTraining_fixed`, identica alla
    patch presente nel notebook originale.
    """
    G = cfg.glob
    from neuroCombat import neuroCombat, neuroCombatFromTraining
    import neuroCombat.neuroCombat as _nc_mod

    def neuroCombatFromTraining_fixed(dat, batch, estimates):
        print("[neuroCombatFromTraining] In development ...\n")
        batch = np.array(batch, dtype="str")
        old_levels = np.array(estimates['batches'], dtype="str")
        missing_levels = np.setdiff1d(np.unique(batch), old_levels)
        if missing_levels.shape[0] != 0:
            raise ValueError(f"The batches {missing_levels} are not part of the training dataset")

        wh = [int(np.where(old_levels == x)[0][0]) if x in old_levels else None for x in batch]

        var_pooled = estimates['var.pooled']
        stand_mean = estimates['stand.mean'][:, 0]
        mod_mean = estimates['mod.mean']
        gamma_star = estimates['gamma.star']
        delta_star = estimates['delta.star']
        n_array = dat.shape[1]
        stand_mean = stand_mean + mod_mean.mean(axis=1)
        stand_mean = np.transpose([stand_mean, ] * n_array)
        bayesdata = np.subtract(dat, stand_mean) / np.sqrt(var_pooled)
        gamma = np.transpose(gamma_star[wh, :])
        delta = np.transpose(delta_star[wh, :])
        bayesdata = np.subtract(bayesdata, gamma) / np.sqrt(delta)
        bayesdata = bayesdata * np.sqrt(var_pooled) + stand_mean
        return {'data': bayesdata, 'estimates': estimates}

    _nc_mod.neuroCombatFromTraining = neuroCombatFromTraining_fixed
    neuroCombatFromTraining = neuroCombatFromTraining_fixed

    tissue_labels_tr = np.concatenate([
        meta_train[meta_train["dataset_origin"] == ds][G.LABEL_COL].values
        for ds in ["GSE225845", "GSE287331"]
    ])

    print("\n[4] ComBat su pool train")
    X_tr_pool = np.vstack([X_train_abs["GSE225845"], X_train_abs["GSE287331"]])
    batch_pool = (
        ["GSE225845"] * len(X_train_abs["GSE225845"]) +
        ["GSE287331"] * len(X_train_abs["GSE287331"])
    )
    label_pool = np.concatenate([tissue_labels_tr]).astype(int)

    covars_train = pd.DataFrame({"batch": batch_pool, "label": label_pool})
    print(f"    Pool totale: {X_tr_pool.shape}")
    print(covars_train["batch"].value_counts())

    result = neuroCombat(
        dat=X_tr_pool.T, covars=covars_train, batch_col="batch",
        categorical_cols=["label"], mean_only=False,
    )

    n_225 = len(X_train_abs["GSE225845"])
    n_287 = len(X_train_abs["GSE287331"])

    X_tr_combat = result["data"].T[:n_225 + n_287]
    combat_estimates = result["estimates"]

    print(f"\nComBat completato.")
    print(f"  X_tr_combat      : {X_tr_combat.shape}")

    # ── applica a test ──
    X_te_pool = np.vstack([X_test_abs[ds] for ds in ["GSE225845", "GSE287331"]])
    ds_labels_te = (
        ["GSE225845"] * len(X_test_abs["GSE225845"]) +
        ["GSE287331"] * len(X_test_abs["GSE287331"])
    )
    X_te_combat = neuroCombatFromTraining(
        dat=X_te_pool.T, batch=ds_labels_te, estimates=combat_estimates,
    )["data"].T
    print(f"  X_te_combat      : {X_te_combat.shape}")

    # ── applica a tumor ──
    X_tum_pool_raw = np.vstack([X_tumor_abs[ds] for ds in ["GSE225845", "GSE287331"]])
    ds_labels_tum = (
        ["GSE225845"] * len(X_tumor_abs["GSE225845"]) +
        ["GSE287331"] * len(X_tumor_abs["GSE287331"])
    )
    X_tum_combat = neuroCombatFromTraining(
        dat=X_tum_pool_raw.T, batch=ds_labels_tum, estimates=combat_estimates,
    )["data"].T
    print(f"  X_tum_combat     : {X_tum_combat.shape}")

    return {
        "tissue_labels_tr": tissue_labels_tr,
        "ds_labels_te": ds_labels_te,
        "ds_labels_tum": ds_labels_tum,
        "X_tr_combat": X_tr_combat,
        "X_te_combat": X_te_combat,
        "X_tum_combat": X_tum_combat,
        "combat_estimates": combat_estimates,
    }


def step_tissue_labels_te(cfg: PipelineConfig, meta_test: pd.DataFrame) -> dict:
    """CELLA 28 (parte) — tissue_labels_te (0=Normal, 1=Adjacent) sul test pool,
    nello stesso ordine [GSE225845, GSE287331] usato per X_te_combat."""
    G = cfg.glob
    tissue_labels_te = np.concatenate([
        meta_test[meta_test["dataset_origin"] == ds][G.LABEL_COL].values
        for ds in ["GSE225845", "GSE287331"]
    ])
    return {"tissue_labels_te": tissue_labels_te}


# ==============================================================================
# STEP 5 (Edgar) — Feature selection train-only su variabilita' (r_beta)
# ==============================================================================

def step_edgar_feature_selection(cfg: PipelineConfig, common_cpgs: list,
                                  X_tr_combat: np.ndarray, X_te_combat: np.ndarray,
                                  tissue_labels_tr: np.ndarray,
                                  tissue_labels_te: np.ndarray) -> dict:
    """CELLA 12 — STEP 5: Feature Selection (Edgar beta).
    r_beta = Q90 - Q10 (train-only), filtro r_beta >= R_BETA_CUTOFF."""
    cfg1 = cfg.step1
    P = cfg.paths

    def compute_r_beta_edgar(X: np.ndarray, labels: np.ndarray, cutoff: float):
        q90 = np.quantile(X, 0.90, axis=0)
        q10 = np.quantile(X, 0.10, axis=0)
        r_beta = q90 - q10
        mask_keep = r_beta >= cutoff
        return r_beta, mask_keep

    print("=" * 90)
    print(f"Edgar(beta) train-only  cutoff r_beta >= {cfg1.R_BETA_CUTOFF}")
    print("=" * 90)

    r_beta, mask_edgar = compute_r_beta_edgar(X_tr_combat, tissue_labels_tr, cfg1.R_BETA_CUTOFF)

    cpg_cols_edgar = [common_cpgs[i] for i in range(len(common_cpgs)) if mask_edgar[i]]
    selected_cpg_indices = np.where(mask_edgar)[0]
    selected_cpgs = cpg_cols_edgar

    print(f"CpGs prima di Edgar : {len(common_cpgs):,}")
    print(f"CpGs dopo Edgar     : {len(selected_cpgs):,}  (rimossi {len(common_cpgs) - len(selected_cpgs):,})")

    B_train = X_tr_combat[:, selected_cpg_indices].astype(np.float32)
    B_test = X_te_combat[:, selected_cpg_indices].astype(np.float32)
    y_train = tissue_labels_tr
    y_test = tissue_labels_te

    print(f"B_train: {B_train.shape}")
    print(f"B_test : {B_test.shape}")

    os.makedirs(P.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(P.OUTPUT_DIR, "cpgs_edgar_selected.txt"), "w") as f:
        f.write("\n".join(selected_cpgs))
    print("Done.")

    return {
        "B_train": B_train, "B_test": B_test,
        "y_train": y_train, "y_test": y_test,
        "selected_cpgs": selected_cpgs,
        "cpg_cols_edgar": np.array(selected_cpgs, dtype=object),
    }


# ==============================================================================
# STEP 2 — beta -> M transform + variance filter (train-only)
# ==============================================================================

def step2_m_transform_variance_filter(cfg: PipelineConfig, B_train: np.ndarray,
                                       B_test: np.ndarray, selected_cpgs: list) -> dict:
    """CELLA 36 — STEP 2: beta->M (post-Edgar), drop lowest-variance % in TRAIN
    only. Versione operativa (la cella "diagnostica" che suggerisce un
    VAR_DROP_FRAC data-driven e' disponibile come `diagnose_var_cutoff()` qui
    sotto, da chiamare manualmente se vuoi confrontare con il valore scelto)."""
    cfg2 = cfg.step2

    cpg_cols_edgar = np.array(selected_cpgs, dtype=object)

    M_train = beta_to_m(B_train, eps=cfg2.EPS_M).astype(np.float32)
    M_test = beta_to_m(B_test, eps=cfg2.EPS_M).astype(np.float32)
    print(f"After M-transform: M_train={M_train.shape} | M_test={M_test.shape}")

    def soft_sd_filter(X, cols, drop_frac):
        sd = X.std(axis=0, ddof=1)
        thr = np.quantile(sd, drop_frac)
        keep_mask = sd >= thr
        return X[:, keep_mask], cols[keep_mask], sd[keep_mask], keep_mask

    if cfg2.APPLY_VAR_FILTER and (cfg2.VAR_DROP_FRAC > 0):
        M_train_f, cpg_cols_f, sd_f, keep_mask = soft_sd_filter(
            M_train, cpg_cols_edgar, drop_frac=cfg2.VAR_DROP_FRAC)
        M_test_f = M_test[:, keep_mask]

        print(f"Variance filter: drop_frac={cfg2.VAR_DROP_FRAC:.2%}")
        print(f"CpGs before: {M_train.shape[1]:,} | after: {M_train_f.shape[1]:,}  "
              f"(dropped {M_train.shape[1]-M_train_f.shape[1]:,})")

        M_train, M_test, cpg_cols_edgar = M_train_f, M_test_f, cpg_cols_f
        B_train = B_train[:, keep_mask]
        B_test = B_test[:, keep_mask]
    else:
        print("Variance filter OFF (no CpGs removed).")

    cpg_cols_current = cpg_cols_edgar.copy()
    print(f"Current aligned: B_train={B_train.shape} | M_train={M_train.shape} | "
          f"CpGs={len(cpg_cols_current):,}")

    return {
        "M_train": M_train, "M_test": M_test,
        "B_train": B_train, "B_test": B_test,
        "cpg_cols_current": cpg_cols_current,
    }


def diagnose_var_cutoff(M_train: np.ndarray, candidate_fracs=(0.01, 0.02, 0.05, 0.10),
                         show_plot: bool = True):
    """CELLA 35 (diagnostica) — Mostra il cutoff variance data-driven (prima
    valle della distribuzione SD) e quante CpG vengono rimosse per ogni
    frazione candidata. Utile per scegliere VAR_DROP_FRAC, NON obbligatoria
    nella pipeline principale."""
    from scipy.signal import find_peaks

    sd = M_train.std(axis=0, ddof=1)

    hist_vals, bin_edges = np.histogram(sd, bins=300)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valleys, props = find_peaks(-hist_vals, prominence=hist_vals.max() * 0.05)

    if len(valleys) > 0:
        valley_sd = float(bin_centers[valleys[0]])
        frac_at_valley = float((sd < valley_sd).mean())
    else:
        valley_sd = None
        frac_at_valley = None

    print("=" * 60)
    print("BLOCK B — Variance filter cutoff diagnostics")
    print("=" * 60)
    print(f"SD(M): min={sd.min():.4f}  max={sd.max():.4f}  median={np.median(sd):.4f}")
    print()

    if valley_sd is not None:
        n_rem = int((sd < valley_sd).sum())
        n_keep = int((sd >= valley_sd).sum())
        print(f"  -> Cutoff DATA-DRIVEN (prima valle): SD={valley_sd:.4f}")
        print(f"    Equivale a drop_frac={frac_at_valley:.3%}")
        print(f"    Rimosse: {n_rem:,}  |  Tenute: {n_keep:,}  ({100*n_keep/sd.size:.1f}%)")
    else:
        print("  -> Nessuna valle trovata — distribuzione unimodale")

    for f in candidate_fracs:
        thr = float(np.quantile(sd, f))
        n_rem = int((sd < thr).sum())
        n_keep = sd.size - n_rem
        print(f"    drop_frac={f:.2%}  SD_thr={thr:.4f}  rimosse={n_rem:,}  tenute={n_keep:,}")

    if show_plot:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(sd, bins=200, density=True, alpha=0.8, color="#7FC97F",
                edgecolor="white", linewidth=0.3, label="SD(M) train")
        if valley_sd is not None:
            ax.axvline(valley_sd, linestyle="-", linewidth=2.0, color="red",
                        label=f"data-driven (valle) {valley_sd:.4f}")
        colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(candidate_fracs)))
        for f, col in zip(candidate_fracs, colors):
            thr = float(np.quantile(sd, f))
            ax.axvline(thr, linestyle="--", linewidth=1.2, color=col,
                        label=f"manual {f:.0%} -> {thr:.4f}")
        ax.set_xlabel("SD(M) train")
        ax.set_ylabel("Density")
        ax.set_title("BLOCK B — Variance filter: data-driven vs manual")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    return frac_at_valley


# ==============================================================================
# STEP 2.5 — Bio-weight (CpG island / promoter-proximal)
# ==============================================================================

def step2_5_bio_weight(cfg: PipelineConfig, cpg_cols_current: np.ndarray) -> dict:
    """CELLA 38 — STEP 2.5: peso genomico bio_weight (DA METTERE TRA STEP2 e STEP3)."""
    P = cfg.paths

    def load_manifest_bio(path: str) -> pd.DataFrame:
        header_idx = None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if "IlmnID" in line.split(","):
                    header_idx = i
                    break
        if header_idx is None:
            raise ValueError("Header manifest non trovato (colonna IlmnID).")

        df = pd.read_csv(
            path, header=header_idx, engine="python", on_bad_lines="skip",
            usecols=["IlmnID", "Relation_to_UCSC_CpG_Island", "UCSC_RefGene_Group"],
        )
        df.columns = [c.strip() for c in df.columns]
        df["IlmnID"] = df["IlmnID"].astype(str).str.strip()
        return df

    manifest_bio = load_manifest_bio(P.MANIFEST_PATH)

    island_cpgs = set(manifest_bio.loc[
        manifest_bio["Relation_to_UCSC_CpG_Island"] == "Island", "IlmnID"])
    promoter_cpgs = set(manifest_bio.loc[
        manifest_bio["UCSC_RefGene_Group"].fillna("").astype(str).str.contains("TSS", na=False),
        "IlmnID"])

    cpg_cols_current_arr = np.array(cpg_cols_current, dtype=object)

    bio_weight = np.array([
        1.00 if (c in island_cpgs and c in promoter_cpgs) else
        0.70 if (c in island_cpgs) else
        0.50 if (c in promoter_cpgs) else
        0.20
        for c in cpg_cols_current_arr
    ], dtype=np.float32)

    bio_weight_norm = (bio_weight - bio_weight.min()) / (bio_weight.max() - bio_weight.min())

    n_top = int((bio_weight == 1.00).sum())
    n_island = int((bio_weight == 0.70).sum())
    n_prom = int((bio_weight == 0.50).sum())
    n_other = int((bio_weight == 0.20).sum())

    print(f"[BIO] CpG in isola CpG    : {len(island_cpgs):,}")
    print(f"[BIO] CpG in promotore TSS: {len(promoter_cpgs):,}")
    print(f"[BIO] Entrambi            : {len(island_cpgs & promoter_cpgs):,}")
    print(f"\n[BIO] Distribuzione peso biologico su {len(cpg_cols_current_arr):,} CpG (post STEP2):")
    print(f"  isola+TSS (1.00): {n_top:,}   ({100*n_top/len(cpg_cols_current_arr):.1f}%)")
    print(f"  isola     (0.70): {n_island:,}   ({100*n_island/len(cpg_cols_current_arr):.1f}%)")
    print(f"  promotore (0.50): {n_prom:,}   ({100*n_prom/len(cpg_cols_current_arr):.1f}%)")
    print(f"  altro     (0.20): {n_other:,}   ({100*n_other/len(cpg_cols_current_arr):.1f}%)")

    return {"bio_weight": bio_weight, "bio_weight_norm": bio_weight_norm}


# ==============================================================================
# Allinea X_tum_combat alle stesse CpG di M_train (cpg_cols_current)
# ==============================================================================

def step_align_tumor_to_current(cfg: PipelineConfig, X_tum_combat: np.ndarray,
                                 common_cpgs: list, cpg_cols_current: np.ndarray) -> dict:
    """CELLA 39 — Allinea X_tum_combat (spazio common_cpgs, post-ComBat) alle
    stesse CpG di M_train (cpg_cols_current, post-Edgar+variance-filter), poi
    trasforma in M-space. Ritorna X_tum_combat SOVRASCRITTO in M-space
    (coerente col notebook, che fa lo stesso "rename" concettuale)."""
    cfg2 = cfg.step2

    tum_cpg_cols = np.array(common_cpgs, dtype=object)
    tum_col_map = {c: j for j, c in enumerate(tum_cpg_cols)}

    cpg_cols_current_arr = np.array(cpg_cols_current, dtype=object)
    idx_tum = np.array(
        [tum_col_map[c] for c in cpg_cols_current_arr if c in tum_col_map], dtype=int
    )
    missing = [c for c in cpg_cols_current_arr if c not in tum_col_map]
    if missing:
        print(f"\u26a0 {len(missing)} CpG di cpg_cols_current non trovate in X_tum_combat — escluse")

    X_tum_filtered = X_tum_combat[:, idx_tum].astype(np.float32)
    X_tum_M = beta_to_m(X_tum_filtered, eps=cfg2.EPS_M)

    print(f"X_tum_M shape: {X_tum_M.shape}  (atteso: n_tumor x {len(cpg_cols_current_arr)})")

    return {"X_tum_combat_mspace": X_tum_M}


# ==============================================================================
# STEP 3 — Repeated Stratified K-Fold stability ranking (score stat + bio) [GPU]
# ==============================================================================
#
# QUESTO E' LO STEP DA VARIARE per l'esperimento richiesto: cambia i campi di
# `cfg.step3` (K_FOLDS, R_REPEATS, STABILITY_SIGN_MIN, W_BIO, W_IP, EPS_IP,
# MIN_CANDIDATES_TARGET, CHUNK_SIZE) e richiama questa funzione (+ tutte le
# successive) per vedere come cambia il pannello finale di CpG.
#
# Richiede GPU NVIDIA + CuPy (vedi import in testa al file). Nessun fallback
# CPU per scelta esplicita.
# ==============================================================================

def _ranks_ascending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.int32)
    return ranks


def _ranks_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.int32)
    return ranks


def _mannwhitney_gpu_chunked(X0: np.ndarray, X1: np.ndarray, idx_pf: np.ndarray,
                              chunk_size: int) -> np.ndarray:
    """Mann-Whitney U con approssimazione normale (identica a
    method='asymptotic') vettorizzata su GPU con CuPy, con chunking per
    colonne per evitare OOM. Identica alla cella GPU del notebook."""
    n1 = X0.shape[0]
    n2 = X1.shape[0]
    n1n2 = float(n1 * n2)
    mean_U = n1n2 / 2.0
    std_U = float(np.sqrt(n1n2 * (n1 + n2 + 1) / 12.0))

    pvals_out = np.ones(len(idx_pf), dtype=np.float64)

    for start in range(0, len(idx_pf), chunk_size):
        end = min(start + chunk_size, len(idx_pf))
        idx_chunk = idx_pf[start:end]

        A = cp.asarray(X0[:, idx_chunk], dtype=cp.float32)
        B = cp.asarray(X1[:, idx_chunk], dtype=cp.float32)

        A_exp = A[:, cp.newaxis, :]
        B_exp = B[cp.newaxis, :, :]

        U1 = (A_exp > B_exp).sum(axis=(0, 1)).astype(cp.float64) + \
             0.5 * (A_exp == B_exp).sum(axis=(0, 1)).astype(cp.float64)

        z = (U1 - mean_U) / std_U
        z_cpu = cp.asnumpy(z)
        pvals_out[start:end] = 2.0 * ndtr(-np.abs(z_cpu))

        del A, B, A_exp, B_exp, U1, z
        cp.get_default_memory_pool().free_all_blocks()

    return pvals_out


def step3_rskf_score(cfg: PipelineConfig, M_train: np.ndarray, y_train: np.ndarray,
                      cpg_cols_current: np.ndarray, bio_weight_norm: np.ndarray,
                      X_tum_combat_mspace: np.ndarray) -> dict:
    """STEP 3 — RSKF (K_FOLDS x R_REPEATS) + IP score dentro il loop + cutoff
    |Delta M| data-driven (elbow cord distance) + Mann-Whitney GPU (CuPy).

    Score finale:
      score      = (1 - W_BIO) * score_stat + W_BIO * score_bio
      score_stat = normalizzato(0.5*rank_p + 0.5*rank_|DeltaM|)
      score_bio  = W_IP*(1-IP_norm) + (1-W_IP)*(1-bio_weight_norm)
      IP_norm    = media sui fold di clip(DeltaM(Adj-Norm) / DeltaM(Tum-Norm), 0, 1)
      IP_raw     = media sui fold di DeltaM(Adj-Norm) / DeltaM(Tum-Norm)  (non clippato)

    Selezione candidati:
      stability_sign >= STABILITY_SIGN_MIN
      |DeltaM| >= elbow data-driven (cord distance sulla curva |DeltaM|-rank)
      nessun taglio arbitrario a N fisso (fallback a p90 solo se troppo pochi)
    """
    c3 = cfg.step3
    G = cfg.glob
    P = cfg.paths

    X_tumor_combat = X_tum_combat_mspace

    print("\n" + "=" * 90)
    print(f"STEP 3 — RSKF (K={c3.K_FOLDS}, R={c3.R_REPEATS}) + IP score dentro loop "
          f"+ cutoff |DeltaM| data-driven [GPU]")
    print("=" * 90)

    try:
        cp.cuda.Device(0).use()
        print(f"GPU disponibile: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    except Exception as e:
        raise RuntimeError(f"GPU non disponibile: {e}")

    X = M_train
    y = y_train
    cols = np.array(cpg_cols_current, dtype=object)
    n, p = X.shape

    print(f"Train: n={n} | p={p:,} | splits={c3.K_FOLDS*c3.R_REPEATS} "
          f"(K={c3.K_FOLDS}, R={c3.R_REPEATS})")
    print(f"W_BIO={c3.W_BIO} | W_IP={c3.W_IP} (dentro score_bio)")

    assert X_tumor_combat.shape[1] == p, \
        f"X_tumor_combat ha {X_tumor_combat.shape[1]} feature, attese {p}"

    mean_tum_global = X_tumor_combat.mean(axis=0)

    rkf = RepeatedStratifiedKFold(
        n_splits=c3.K_FOLDS, n_repeats=c3.R_REPEATS, random_state=G.RANDOM_STATE
    )

    rank_p_sum = np.zeros(p, dtype=np.float64)
    rank_dm_sum = np.zeros(p, dtype=np.float64)
    sign_agree_count = np.zeros(p, dtype=np.int32)
    sign_valid_count = np.zeros(p, dtype=np.int32)
    ip_norm_sum = np.zeros(p, dtype=np.float64)
    ip_raw_sum = np.zeros(p, dtype=np.float64)
    ip_valid_count = np.zeros(p, dtype=np.int32)
    valid_splits_for_score = 0

    n_total_splits = c3.K_FOLDS * c3.R_REPEATS

    for split_id, (train_idx, val_idx) in enumerate(rkf.split(X, y), start=1):
        X_tr = X[train_idx]; y_tr = y[train_idx]
        X_va = X[val_idx]; y_va = y[val_idx]

        tr0 = (y_tr == 0); tr1 = (y_tr == 1)
        va0 = (y_va == 0); va1 = (y_va == 1)
        if tr0.sum() == 0 or tr1.sum() == 0 or va0.sum() == 0 or va1.sum() == 0:
            continue

        X0_tr = X_tr[tr0]; X1_tr = X_tr[tr1]
        X0_va = X_va[va0]; X1_va = X_va[va1]

        dM_tr = X1_tr.mean(axis=0) - X0_tr.mean(axis=0)
        abs_dM_tr = np.abs(dM_tr)
        dM_va = X1_va.mean(axis=0) - X0_va.mean(axis=0)

        sign_tr = np.sign(dM_tr).astype(np.int8)
        sign_va = np.sign(dM_va).astype(np.int8)

        nonzero = (sign_tr != 0) & (sign_va != 0)
        sign_valid_count += nonzero.astype(np.int32)
        sign_agree_count += (nonzero & (sign_tr == sign_va)).astype(np.int32)

        # ── IP calcolato sul fold train ──
        mean_norm_fold = X0_tr.mean(axis=0)
        mean_adj_fold = X1_tr.mean(axis=0)

        num_ip_fold = mean_adj_fold - mean_norm_fold
        denom_ip_fold = mean_tum_global - mean_norm_fold
        denom_safe_fold = np.where(
            np.abs(denom_ip_fold) < c3.EPS_IP,
            np.sign(denom_ip_fold + 1e-10) * c3.EPS_IP,
            denom_ip_fold
        )
        IP_raw_fold = num_ip_fold / denom_safe_fold
        IP_norm_fold = np.clip(IP_raw_fold, 0.0, 1.0).astype(np.float32)

        ip_raw_sum += IP_raw_fold
        ip_norm_sum += IP_norm_fold
        ip_valid_count += 1

        # ── p-value GPU su tutte le CpG ──
        idx_pf = np.arange(p)
        pvals = np.ones(p, dtype=np.float64)
        pvals_pf = _mannwhitney_gpu_chunked(X0_tr, X1_tr, idx_pf, c3.CHUNK_SIZE)
        pvals[idx_pf] = pvals_pf

        rank_p_sum += _ranks_ascending(pvals)
        rank_dm_sum += _ranks_descending(abs_dM_tr)
        valid_splits_for_score += 1

        print(f"  done split {split_id:>3}/{n_total_splits} | p={p:,} | "
              f"valid={valid_splits_for_score}")

    if valid_splits_for_score == 0:
        raise RuntimeError("No valid splits — controlla labels/stratificazione.")

    # ── Aggregazione score ──
    mean_rank_p = rank_p_sum / valid_splits_for_score
    mean_rank_dm = rank_dm_sum / valid_splits_for_score

    score_stat_raw = 0.5 * mean_rank_p + 0.5 * mean_rank_dm
    score_stat = (score_stat_raw - score_stat_raw.min()) / \
                 (score_stat_raw.max() - score_stat_raw.min())

    IP_raw = (ip_raw_sum / np.maximum(ip_valid_count, 1)).astype(np.float32)
    IP_norm = (ip_norm_sum / np.maximum(ip_valid_count, 1)).astype(np.float32)

    print(f"\n[IP] Distribuzione IP_raw (media sui fold, prima del clip):")
    print(f"  min={IP_raw.min():.3f}  max={IP_raw.max():.3f}  median={np.median(IP_raw):.3f}")
    print(f"  IP in [0,1]          : {int(((IP_raw>=0)&(IP_raw<=1)).sum()):,}  "
          f"({100*((IP_raw>=0)&(IP_raw<=1)).mean():.1f}%)")
    print(f"  IP > 1 (Adj > Tum)   : {int((IP_raw>1).sum()):,}  — clippati a 1")
    print(f"  IP < 0 (dir. opposta): {int((IP_raw<0).sum()):,}  — clippati a 0")

    score_bio = c3.W_IP * (1.0 - IP_norm) + (1.0 - c3.W_IP) * (1.0 - bio_weight_norm)
    score = (1.0 - c3.W_BIO) * score_stat + c3.W_BIO * score_bio

    print(f"\n[SCORE] score_stat range : [{score_stat.min():.4f}, {score_stat.max():.4f}]")
    print(f"[SCORE] score_bio  range : [{score_bio.min():.4f}, {score_bio.max():.4f}]")
    print(f"[SCORE] score      range : [{score.min():.4f}, {score.max():.4f}]")

    # ── Stability ──
    stability_sign = np.zeros(p, dtype=np.float64)
    nz = sign_valid_count > 0
    stability_sign[nz] = sign_agree_count[nz] / sign_valid_count[nz]

    dM_full = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    abs_dM_full = np.abs(dM_full)

    # ── Cutoff |DeltaM| data-driven (elbow cord distance) ──
    def elbow_cord(ranks, values):
        x = np.asarray(ranks, dtype=float)
        v = np.asarray(values, dtype=float)
        x_n = (x - x.min()) / (x.max() - x.min())
        y_n = (v - v.min()) / (v.max() - v.min())
        dx = x_n[-1] - x_n[0]
        dy = y_n[-1] - y_n[0]
        norm = np.sqrt(dx**2 + dy**2)
        dist = np.abs(dy * x_n - dx * y_n + x_n[-1]*y_n[0] - y_n[-1]*x_n[0]) / norm
        return int(np.argmax(dist))

    dm_stable = abs_dM_full[stability_sign >= c3.STABILITY_SIGN_MIN]
    dm_sorted = np.sort(dm_stable)[::-1]
    ranks_dm = np.arange(1, len(dm_sorted) + 1)

    elbow_idx = elbow_cord(ranks_dm, dm_sorted)
    DELTA_M_THR_MAIN = float(dm_sorted[elbow_idx])
    DELTA_M_THR_FALLBACK = float(np.percentile(abs_dM_full, 90))

    n_main = int((dm_stable >= DELTA_M_THR_MAIN).sum())
    n_fallback = int((abs_dM_full[stability_sign >= c3.STABILITY_SIGN_MIN]
                       >= DELTA_M_THR_FALLBACK).sum())

    print(f"\n[|DeltaM|] Cutoff data-driven (elbow cord): {DELTA_M_THR_MAIN:.4f}  -> {n_main:,} CpG")
    print(f"[|DeltaM|] Cutoff fallback (p90)          : {DELTA_M_THR_FALLBACK:.4f}  -> {n_fallback:,} CpG")

    # ── Filtro candidati ──
    thr = DELTA_M_THR_MAIN
    mask_cand = (stability_sign >= c3.STABILITY_SIGN_MIN) & (abs_dM_full >= thr)
    n_cand = int(mask_cand.sum())
    print(f"\nCandidati stability>={c3.STABILITY_SIGN_MIN} & |DeltaM|>={thr:.4f}: {n_cand:,}")

    if n_cand < c3.MIN_CANDIDATES_TARGET:
        print(f"\u26a0 Troppo pochi candidati ({n_cand}), uso fallback p90")
        thr = DELTA_M_THR_FALLBACK
        mask_cand = (stability_sign >= c3.STABILITY_SIGN_MIN) & (abs_dM_full >= thr)
        n_cand = int(mask_cand.sum())
        print(f"Fallback |DeltaM|>={thr:.4f}: candidati={n_cand:,}")

    if n_cand == 0:
        raise RuntimeError("Step 3: 0 candidati anche dopo fallback.")

    cand_idx = np.where(mask_cand)[0]
    order = np.argsort(score[cand_idx], kind="mergesort")
    cand_idx_sorted = cand_idx[order]
    cand_idx_top = cand_idx_sorted

    cpg_candidates = cols[cand_idx_top]

    print(f"\nCpG candidati selezionati : {len(cpg_candidates):,}")
    print(f"  threshold |DeltaM|      : {thr:.4f} "
          f"({'elbow data-driven' if thr == DELTA_M_THR_MAIN else 'fallback p90'})")
    print(f"  stability threshold     : >= {c3.STABILITY_SIGN_MIN}")
    print(f"  W_BIO={c3.W_BIO} | W_IP={c3.W_IP}")
    print("  esempio CpG:", cpg_candidates[:5].tolist())

    step3_stats = {
        "score": score[cand_idx_top],
        "score_stat": score_stat[cand_idx_top],
        "score_bio": score_bio[cand_idx_top],
        "IP_norm": IP_norm[cand_idx_top],
        "IP_raw": IP_raw[cand_idx_top],
        "mean_rank_p": mean_rank_p[cand_idx_top],
        "mean_rank_abs_dM": mean_rank_dm[cand_idx_top],
        "stability_sign": stability_sign[cand_idx_top],
        "deltaM_full": dM_full[cand_idx_top],
        "abs_deltaM_full": abs_dM_full[cand_idx_top],
        "used_deltaM_thr": thr,
        "used_deltaM_thr_source": "elbow_cord" if thr == DELTA_M_THR_MAIN else "fallback_p90",
        "valid_splits_for_score": valid_splits_for_score,
        "sign_valid_splits": sign_valid_count[cand_idx_top],
    }

    OUTDIR = Path(P.OUTPUT_DIR) / "fs_outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    save_pickle(cpg_candidates.tolist(), OUTDIR / "cpg_candidates.pkl")
    save_pickle(step3_stats, OUTDIR / "step3_stats.pkl")
    print("\nStep3 variables saved.")

    return {
        "cpg_candidates": cpg_candidates,
        "step3_stats": step3_stats,
        "score_map": {c: s for c, s in zip(cpg_candidates, step3_stats["score"])},
    }


# ==============================================================================
# STEP 3b — Region anchoring su CpG islands
# ==============================================================================

def step3b_region_anchoring(cfg: PipelineConfig, cpg_candidates: np.ndarray,
                             step3_stats: dict) -> dict:
    """CELLA 46 — STEP 3b: region_score = median(score) x fraction_stable.
    fraction_stable >= 0.6, >= 2 CpG per regione. top-m per regione -> Pool A
    [anchored], resto -> Pool B [non-anchored]."""
    c3b = cfg.step3b
    c3 = cfg.step3
    P = cfg.paths

    print("\n" + "=" * 90)
    print("STEP 3b — Region anchoring su CpG islands")
    print("=" * 90)

    cand = np.array(cpg_candidates, dtype=object)
    cand_score = np.array(step3_stats["score"], dtype=float)
    cand_stab = np.array(step3_stats["stability_sign"], dtype=float)
    cand_absdM = np.array(step3_stats["abs_deltaM_full"], dtype=float)
    thr_used = float(step3_stats["used_deltaM_thr"])

    if not (cand.shape[0] == cand_score.shape[0] == cand_stab.shape[0] == cand_absdM.shape[0]):
        raise RuntimeError("STEP 3b: candidate arrays are misaligned (cpg_candidates vs step3_stats).")

    stable_cpg_flag = (cand_stab >= c3.STABILITY_SIGN_MIN) & (cand_absdM >= thr_used)

    print(f"Candidates: {cand.size:,}")
    print(f"DeltaM threshold used in STEP 3: {thr_used:.2f}")
    print(f"Stable CpGs inside candidates (stab>={c3.STABILITY_SIGN_MIN} AND |DeltaM|>=thr): "
          f"{int(stable_cpg_flag.sum()):,}")

    mf = load_manifest_with_real_header(P.MANIFEST_PATH)
    mf = mf[[c3b.MANIFEST_CPG_COL, c3b.MANIFEST_ISLAND_NAME_COL, c3b.MANIFEST_REL_COL]].copy()
    mf.columns = ["cpg", "island", "rel"]
    mf["cpg"] = mf["cpg"].astype(str)
    mf["island"] = mf["island"].fillna("").astype(str).str.strip()
    mf["rel"] = mf["rel"].fillna("").astype(str).str.strip()

    mf_island = mf[(mf["rel"].isin(list(c3b.ANCHOR_CONTEXT))) & (mf["island"] != "")]
    cpg_to_island = dict(zip(mf_island["cpg"], mf_island["island"]))

    island_id = np.array([cpg_to_island.get(c, None) for c in cand], dtype=object)
    in_island = island_id != None
    print(f"Mapped to named CpG islands (context={c3b.ANCHOR_CONTEXT}): {int(in_island.sum()):,}")

    island_to_idx = {}
    for i, rid in enumerate(island_id):
        if rid is None:
            continue
        island_to_idx.setdefault(rid, []).append(i)

    region_rows = []
    poolA_idx = []

    for rid, idxs in island_to_idx.items():
        idxs = np.array(idxs, dtype=int)
        if idxs.size < c3b.MIN_CPGS_PER_REGION:
            continue

        frac_stable = float(stable_cpg_flag[idxs].mean())
        if frac_stable < c3b.FRACTION_STABLE_MIN:
            continue

        med_score = float(np.median(cand_score[idxs]))
        region_score = med_score * frac_stable

        order = np.argsort(cand_score[idxs], kind="mergesort")
        chosen = idxs[order[:min(c3b.TOP_M_PER_REGION, idxs.size)]]
        poolA_idx.extend(chosen.tolist())

        region_rows.append((rid, idxs.size, frac_stable, med_score, region_score, chosen.size))

    poolA_idx = np.array(sorted(set(poolA_idx)), dtype=int)
    poolA_cpgs = cand[poolA_idx]

    maskA = np.zeros(cand.size, dtype=bool)
    maskA[poolA_idx] = True
    poolB_cpgs = cand[~maskA]

    print(f"Selected regions (pass constraints): {len(region_rows):,}")
    print(f"Pool A (anchored) CpGs: {poolA_cpgs.size:,}  (top-m={c3b.TOP_M_PER_REGION} per region)")
    print(f"Pool B (non-anchored) CpGs: {poolB_cpgs.size:,}")

    region_summary = (
        pl.DataFrame(
            region_rows,
            schema=["island_id", "n_cpgs", "fraction_stable", "median_score", "region_score", "picked_m"]
        ).sort("region_score")
    )

    print("Top 5 regions by region_score (lower is better):")
    print(region_summary.head(5))

    # ── salvataggi (CELLE 47-48) ──
    FS_OUTDIR = Path(P.OUTPUT_DIR) / "fs_outputs"
    FS_OUTDIR.mkdir(parents=True, exist_ok=True)
    save_pickle(poolA_cpgs.astype(str).tolist(), FS_OUTDIR / "poolA_cpgs.pkl")
    save_pickle(poolB_cpgs.astype(str).tolist(), FS_OUTDIR / "poolB_cpgs.pkl")
    pd.Series(poolA_cpgs.astype(str), name="CpG").to_csv(FS_OUTDIR / "poolA_cpgs.csv", index=False)
    pd.Series(poolB_cpgs.astype(str), name="CpG").to_csv(FS_OUTDIR / "poolB_cpgs.csv", index=False)

    region_summary_out = (
        region_summary
        .select(["island_id", "n_cpgs", "fraction_stable", "median_score", "region_score", "picked_m"])
        .with_columns([
            pl.col("island_id").cast(pl.Utf8),
            pl.col("n_cpgs").cast(pl.Int64),
            pl.col("picked_m").cast(pl.Int64),
            pl.col("fraction_stable").cast(pl.Float64),
            pl.col("median_score").cast(pl.Float64),
            pl.col("region_score").cast(pl.Float64),
        ])
        .sort(["region_score", "island_id"])
    )
    region_summary_out.write_csv(FS_OUTDIR / "region_summary.csv")
    region_summary_out.write_parquet(FS_OUTDIR / "region_summary.parquet")
    print(f"Saved PoolA/PoolB and region_summary to: {FS_OUTDIR.resolve()}")

    return {"poolA_cpgs": poolA_cpgs, "poolB_cpgs": poolB_cpgs, "region_summary": region_summary}


# ==============================================================================
# STEP 4 — Correlation clustering separato (Pool A / Pool B)
# ==============================================================================

def step4_correlation_clustering(cfg: PipelineConfig, M_train: np.ndarray,
                                  cpg_cols_current: np.ndarray,
                                  poolA_cpgs: np.ndarray, poolB_cpgs: np.ndarray,
                                  score_map: dict) -> dict:
    """CELLA 50 — STEP 4: Pool A: |r|>=thr -> connected components -> 1 CpG/cluster
    (best STEP3 score). Stesso procedimento per Pool B. Output: unione dei
    rappresentanti (A prima di B, unique)."""
    c4 = cfg.step4

    print("\n" + "=" * 90)
    print(f"STEP 4 — Correlation clustering separato (Pool A, Pool B) |r|>={c4.CORR_CLUSTER_THR}")
    print("=" * 90)

    cols_all = np.array(cpg_cols_current, dtype=object)
    col_to_j = {c: j for j, c in enumerate(cols_all)}

    def _build_Z(pool_cpgs: np.ndarray):
        pool = np.array([c for c in pool_cpgs if c in col_to_j], dtype=object)
        if pool.size == 0:
            return np.empty((M_train.shape[0], 0), dtype=np.float32), pool
        jj = np.array([col_to_j[c] for c in pool], dtype=int)
        Xp = M_train[:, jj].astype(np.float32, copy=False)
        mu = Xp.mean(axis=0)
        sd = Xp.std(axis=0, ddof=1) + 1e-12
        Z = (Xp - mu) / sd
        return Z, pool

    def _connected_components_from_corr(Z: np.ndarray, thr: float):
        p = Z.shape[1]
        if p == 0:
            return []
        if p == 1:
            return [[0]]
        n = Z.shape[0]
        C = (Z.T @ Z) / (n - 1)
        A = (np.abs(C) >= thr)
        np.fill_diagonal(A, True)

        visited = np.zeros(p, dtype=bool)
        comps = []
        for i in range(p):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                nbrs = np.where(A[u])[0]
                for v in nbrs:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(comp)
        return comps

    def _pick_rep_by_best_score(pool_order: np.ndarray, comps):
        reps = []
        for comp in comps:
            members = pool_order[np.array(comp, dtype=int)]
            scores = np.array([score_map.get(c, np.inf) for c in members], dtype=float)
            best = members[int(np.argmin(scores))]
            reps.append(best)
        return np.array(reps, dtype=object)

    def cluster_pool(pool_name: str, pool_cpgs: np.ndarray, thr: float):
        Z, pool_order = _build_Z(pool_cpgs)
        print(f"{pool_name}: CpGs in pool={pool_order.size:,}")
        if pool_order.size == 0:
            return np.array([], dtype=object), np.array([], dtype=int)
        comps = _connected_components_from_corr(Z, thr=thr)
        reps = _pick_rep_by_best_score(pool_order, comps)
        sizes = np.array([len(c) for c in comps], dtype=int)
        print(f"{pool_name}: clusters={len(comps):,} | reps={reps.size:,} | "
              f"median_cluster_size={int(np.median(sizes))} | max_cluster_size={int(sizes.max())}")
        return reps, sizes

    reps_A, sizes_A = cluster_pool("Pool A (anchored)", poolA_cpgs, thr=c4.CORR_CLUSTER_THR)
    reps_B, sizes_B = cluster_pool("Pool B (non-anchored)", poolB_cpgs, thr=c4.CORR_CLUSTER_THR)

    reps_union = np.array(list(dict.fromkeys(np.concatenate([reps_A, reps_B]).tolist())), dtype=object)
    print(f"Union representatives: {reps_union.size:,} (A={reps_A.size:,}, B={reps_B.size:,})")

    return {"reps_A": reps_A, "reps_B": reps_B, "reps_union": reps_union}


# ==============================================================================
# STEP 5 — Diversificazione soft + greedy (chr / context / spatial)
# ==============================================================================

def step5_diversification(cfg: PipelineConfig, reps_union: np.ndarray,
                           score_map: dict) -> dict:
    """CELLA 52 — STEP 5: greedy ordinato per STEP3 score, con vincoli soft
    (budget su chr e contesto, calcolati su K_TARGET) e vincolo spaziale hard
    (max CpG / finestra), con relaxation a stadi: spaziale -> contesto -> chr."""
    c5 = cfg.step5
    P = cfg.paths

    print("\n" + "=" * 90)
    print("STEP 5 — Diversification greedy with soft constraints + relaxation "
          "(spatial->context->chr)")
    print("=" * 90)

    reps = np.array(reps_union, dtype=object)
    if reps.size == 0:
        raise RuntimeError("STEP 5: reps_union is empty.")

    mf_pd = load_manifest_with_real_header(P.MANIFEST_PATH)
    mf = pl.from_pandas(mf_pd).select([
        pl.col(c5.MANIFEST_CPG_COL).cast(pl.Utf8).alias("cpg"),
        pl.col(c5.MANIFEST_CHR_COL).cast(pl.Utf8).alias("chr"),
        pl.col(c5.MANIFEST_POS_COL).cast(pl.Int64).alias("pos"),
        pl.col(c5.MANIFEST_CTX_COL).cast(pl.Utf8).alias("ctx"),
    ]).with_columns([
        pl.col("cpg").fill_null("").str.strip_chars(),
        pl.col("chr").fill_null("").str.strip_chars(),
        pl.col("ctx").fill_null("OpenSea").str.strip_chars(),
    ])

    mf = mf.filter(pl.col("cpg").is_in(reps.tolist()))

    cpg_to_chr = dict(zip(mf["cpg"].to_list(), mf["chr"].to_list()))
    cpg_to_pos = dict(zip(mf["cpg"].to_list(), mf["pos"].to_list()))
    cpg_to_ctx = dict(zip(mf["cpg"].to_list(), mf["ctx"].to_list()))

    mapped = []
    for c in reps:
        ch = cpg_to_chr.get(c, "")
        ps = cpg_to_pos.get(c, None)
        if ch == "" or ps is None:
            continue
        mapped.append(c)
    mapped = np.array(mapped, dtype=object)

    print(f"Reps union: {reps.size:,}")
    print(f"Mapped reps (chr+pos available): {mapped.size:,}")
    if mapped.size == 0:
        raise RuntimeError("STEP 5: No CpGs have manifest chr/pos mapping.")

    scores = np.array([score_map.get(c, np.inf) for c in mapped], dtype=float)
    order = np.argsort(scores, kind="mergesort")
    cand = mapped[order]
    cand_scores = scores[order]

    def diversify_greedy_budgets(cand_cpgs, cand_scores, k_target,
                                  enforce_spatial, enforce_context, enforce_chr):
        selected = []
        chr_counts, ctx_counts, win_counts = {}, {}, {}

        chr_budget = max(int(np.floor(c5.CHR_MAX_FRAC * k_target)), 1)
        ctx_budget = max(int(np.floor(c5.CTX_MAX_FRAC * k_target)), 1)

        def _ok_chr(ch):
            return True if not enforce_chr else (chr_counts.get(ch, 0) + 1) <= chr_budget

        def _ok_ctx(ctx):
            return True if not enforce_context else (ctx_counts.get(ctx, 0) + 1) <= ctx_budget

        def _ok_spatial(ch, pos):
            if not enforce_spatial:
                return True
            win_id = int(pos // c5.WIN_BP)
            return (win_counts.get((ch, win_id), 0) + 1) <= c5.MAX_PER_WIN

        for c, s in zip(cand_cpgs, cand_scores):
            if len(selected) >= k_target:
                break
            ch = cpg_to_chr.get(c, "")
            pos = cpg_to_pos.get(c, None)
            ctx = cpg_to_ctx.get(c, "OpenSea")
            if ch == "" or pos is None:
                continue
            if not _ok_spatial(ch, pos):
                continue
            if not _ok_ctx(ctx):
                continue
            if not _ok_chr(ch):
                continue
            selected.append(c)
            chr_counts[ch] = chr_counts.get(ch, 0) + 1
            ctx_counts[ctx] = ctx_counts.get(ctx, 0) + 1
            win_id = int(pos // c5.WIN_BP)
            win_counts[(ch, win_id)] = win_counts.get((ch, win_id), 0) + 1

        return np.array(selected, dtype=object), chr_counts, ctx_counts, win_counts, chr_budget, ctx_budget

    stages = [
        ("strict", True, True, True),
        ("relax_spatial", False, True, True),
        ("relax_context", False, False, True),
        ("relax_chr", False, False, False),
    ]

    selected_final = None
    stage_used = None
    diag = None
    k_eff = min(c5.K_TARGET, cand.size)

    for name, e_sp, e_ctx, e_chr in stages:
        sel, chr_ct, ctx_ct, win_ct, chr_budget, ctx_budget = diversify_greedy_budgets(
            cand, cand_scores, k_target=k_eff,
            enforce_spatial=e_sp, enforce_context=e_ctx, enforce_chr=e_chr,
        )
        print(f"Stage={name:>13} | selected={sel.size:,} / {k_eff:,} | "
              f"chr_budget={chr_budget} | ctx_budget={ctx_budget}")
        selected_final = sel
        stage_used = name
        diag = (chr_ct, ctx_ct, win_ct)
        if sel.size >= k_eff:
            break

    selected_diverse = selected_final
    print(f"Final selected after diversification: {selected_diverse.size:,} | stage_used={stage_used}")

    chr_ct, ctx_ct, win_ct = diag
    top_chr = sorted(chr_ct.items(), key=lambda x: -x[1])[:5]
    top_ctx = sorted(ctx_ct.items(), key=lambda x: -x[1])[:5]
    print("Top 5 chr counts:", top_chr)
    print("Top 5 ctx counts:", top_ctx)

    return {"selected_diverse": selected_diverse, "cpg_to_chr": cpg_to_chr,
            "cpg_to_pos": cpg_to_pos, "cpg_to_ctx": cpg_to_ctx}


# ==============================================================================
# STEP 6 — Final K selection
# ==============================================================================

def step6_final_k(cfg: PipelineConfig, selected_diverse: np.ndarray) -> dict:
    """CELLA 53 — STEP 6: K = K_FINAL (default 5000), prende i primi K
    dall'ordine di output di STEP 5."""
    c6 = cfg.step6
    P = cfg.paths

    final_cpgs = np.array(selected_diverse, dtype=object)[:min(c6.K_FINAL, len(selected_diverse))]

    print("\n" + "=" * 90)
    print(f"STEP 6 — Final CpG set: K={c6.K_FINAL} (got {final_cpgs.size:,})")
    print("=" * 90)

    # ── export (CELLA 57) ──
    if final_cpgs.size == 0:
        raise RuntimeError("final_cpgs is empty. Check STEP 6.")
    if len(np.unique(final_cpgs)) != final_cpgs.size:
        raise RuntimeError("final_cpgs contains duplicates.")

    OUTDIR = Path(P.OUTPUT_DIR) / "fs_outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = OUTDIR / f"final_cpgs_K{final_cpgs.size}_{ts}.txt"
    with open(txt_path, "w") as f:
        for c in final_cpgs:
            f.write(f"{c}\n")

    csv_path = OUTDIR / f"final_cpgs_K{final_cpgs.size}_{ts}.csv"
    pl.DataFrame({"CpG_ID": final_cpgs}).write_csv(csv_path)

    print("Saved CpGs TXT:", txt_path)
    print("Saved CpGs CSV:", csv_path)
    print("K =", final_cpgs.size)
    print("Example:", final_cpgs[:5].tolist())

    return {"final_cpgs": final_cpgs}


# ==============================================================================
# SVM finale su final_cpgs — training, valutazione, bootstrap CI
# ==============================================================================

def step_svm_train(cfg: PipelineConfig, common_cpgs: list, cpg_cols_current: np.ndarray,
                    final_cpgs: np.ndarray, X_tr_combat: np.ndarray,
                    X_te_combat: np.ndarray, tissue_labels_tr: np.ndarray) -> dict:
    """CELLA 58 — STEP 6 (modeling): Train SVM sulle CpG di final_cpgs.

    Il comportamento e' controllato da cfg.glob.REPRODUCE_NOTEBOOK_BUG:
      - True  -> riproduce il comportamento originale del notebook (mappa
                 costruita su cpg_cols_current).
      - False -> usa la mappa corretta (common_cpgs).
    """
    csvm = cfg.svm
    G = cfg.glob

    if getattr(G, "REPRODUCE_NOTEBOOK_BUG", False):
        col_to_j_common = {c: j for j, c in enumerate(np.asarray(cpg_cols_current, dtype=object).tolist())}
    else:
        col_to_j_common = {c: j for j, c in enumerate(common_cpgs)}

    final_cpgs_indices = np.array(
        [col_to_j_common[c] for c in final_cpgs if c in col_to_j_common], dtype=int
    )
    n_missing = sum(1 for c in final_cpgs if c not in col_to_j_common)
    if n_missing > 0:
        print(f"⚠ {n_missing} CpG di final_cpgs non in common_cpgs — escluse.")

    X_tr_fs = X_tr_combat[:, final_cpgs_indices]
    X_te_fs = X_te_combat[:, final_cpgs_indices]

    print(f"CpG in final_cpgs        : {len(final_cpgs)}")
    print(f"CpG trovate in common_cpgs: {len(final_cpgs_indices)}")
    print(f"X_tr_fs: {X_tr_fs.shape}")
    print(f"X_te_fs: {X_te_fs.shape}")

    best_auc, best_C = -1, 1.0
    for C in csvm.C_GRID:
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(C=C, max_iter=csvm.MAX_ITER),
                                            cv=csvm.CALIBRATION_CV)),
        ])
        scores = cross_val_score(
            pipe, X_tr_fs, tissue_labels_tr,
            cv=StratifiedKFold(n_splits=csvm.CV_FOLDS, shuffle=True, random_state=G.RANDOM_STATE),
            scoring="roc_auc",
        )
        print(f"  C={C:.3f}  AUC={scores.mean():.3f} \u00b1 {scores.std():.3f}")
        if scores.mean() > best_auc:
            best_auc, best_C = scores.mean(), C

    print(f"\nC ottimale: {best_C}  (CV AUC={best_auc:.3f})")

    svm_final = Pipeline([
        ("sc", StandardScaler()),
        ("clf", CalibratedClassifierCV(LinearSVC(C=best_C, max_iter=csvm.MAX_ITER),
                                        cv=csvm.CALIBRATION_CV)),
    ])
    svm_final.fit(X_tr_fs, tissue_labels_tr)
    print("SVM trainato.")

    return {
        "X_tr_fs": X_tr_fs, "X_te_fs": X_te_fs,
        "svm_final": svm_final, "best_C": best_C,
        "col_to_j_common": col_to_j_common,
    }


def step_svm_eval(cfg: PipelineConfig, svm_final, X_te_fs: np.ndarray,
                   tissue_labels_te: np.ndarray, ds_labels_te: list) -> dict:
    """CELLA 59 — STEP 7: Test e valutazione (AUC, BACC, classification report,
    ROC curve per dataset)."""
    P = cfg.paths

    proba_te = svm_final.predict_proba(X_te_fs)[:, 1]
    pred_te = svm_final.predict(X_te_fs)

    auc_overall = roc_auc_score(tissue_labels_te, proba_te)
    bacc_overall = balanced_accuracy_score(tissue_labels_te, pred_te)

    print("=" * 50)
    print(f"AUC  overall : {auc_overall:.3f}")
    print(f"BACC overall : {bacc_overall:.3f}")
    print("=" * 50)
    print(classification_report(tissue_labels_te, pred_te, target_names=["Normal", "Adjacent"]))

    print("\nAUC per dataset:")
    ds_labels_te_arr = np.array(ds_labels_te)
    auc_per_ds, bacc_per_ds = {}, {}
    for ds in ["GSE225845", "GSE287331"]:
        mask = ds_labels_te_arr == ds
        auc_ds = roc_auc_score(tissue_labels_te[mask], proba_te[mask])
        bacc_ds = balanced_accuracy_score(tissue_labels_te[mask], pred_te[mask])
        auc_per_ds[ds] = auc_ds
        bacc_per_ds[ds] = bacc_ds
        print(f"  {ds}: AUC={auc_ds:.3f}  BACC={bacc_ds:.3f}")

    # ── PLOT — ROC curve per dataset (risultato principale) ──
    if "apply_thesis_style" in globals():
        apply_thesis_style(use_tex=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for ds in ["GSE225845", "GSE287331"]:
        mask = ds_labels_te_arr == ds
        auc_ds = auc_per_ds[ds]
        RocCurveDisplay.from_predictions(
            tissue_labels_te[mask], proba_te[mask],
            name=f"{ds} (AUC={auc_ds:.2f})", ax=ax, color=DS_PALETTE[ds],
        )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC Curve per dataset — Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(P.OUTPUT_DIR, "plot_roc_per_dataset.pdf"), bbox_inches="tight")
    plt.show()

    return {
        "proba_te": proba_te, "pred_te": pred_te,
        "auc_overall": auc_overall, "bacc_overall": bacc_overall,
        "auc_per_ds": auc_per_ds, "bacc_per_ds": bacc_per_ds,
    }


def step_svm_bootstrap_ci(cfg: PipelineConfig, tissue_labels_te: np.ndarray,
                           proba_te: np.ndarray, pred_te: np.ndarray,
                           auc_overall: float, bacc_overall: float,
                           ds_labels_te: list) -> dict:
    """CELLA 60 — Intervalli di confidenza Bootstrap (AUC e BACC), overall e
    per dataset."""
    csvm = cfg.svm
    G = cfg.glob

    def bootstrap_ci(y_true, y_score, y_pred, n_boot=csvm.N_BOOT, alpha=csvm.ALPHA,
                      random_state=G.RANDOM_STATE):
        rng = np.random.RandomState(random_state)
        auc_scores, bacc_scores = [], []
        for _ in range(n_boot):
            idx = resample(np.arange(len(y_true)), random_state=rng)
            if len(np.unique(y_true[idx])) < 2:
                continue
            auc_scores.append(roc_auc_score(y_true[idx], y_score[idx]))
            bacc_scores.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
        auc_lo, auc_hi = np.percentile(auc_scores, [100*alpha/2, 100*(1-alpha/2)])
        bacc_lo, bacc_hi = np.percentile(bacc_scores, [100*alpha/2, 100*(1-alpha/2)])
        return (auc_lo, auc_hi), (bacc_lo, bacc_hi)

    print("\n" + "=" * 60)
    print(f"Bootstrap CI (n={csvm.N_BOOT}, alpha={csvm.ALPHA}) — 95% percentile interval")
    print("=" * 60)

    auc_ci, bacc_ci = bootstrap_ci(tissue_labels_te, proba_te, pred_te)
    print(f"\nOverall:")
    print(f"  AUC  = {auc_overall:.3f}  [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    print(f"  BACC = {bacc_overall:.3f}  [{bacc_ci[0]:.3f}, {bacc_ci[1]:.3f}]")

    ds_labels_te_arr = np.array(ds_labels_te)
    ci_per_ds = {}
    print(f"\nPer dataset:")
    for ds in ["GSE225845", "GSE287331"]:
        mask = ds_labels_te_arr == ds
        auc_ds = roc_auc_score(tissue_labels_te[mask], proba_te[mask])
        bacc_ds = balanced_accuracy_score(tissue_labels_te[mask], pred_te[mask])
        auc_ci_ds, bacc_ci_ds = bootstrap_ci(tissue_labels_te[mask], proba_te[mask], pred_te[mask])
        ci_per_ds[ds] = {"auc": auc_ci_ds, "bacc": bacc_ci_ds}
        print(f"  {ds}:")
        print(f"    AUC  = {auc_ds:.3f}  [{auc_ci_ds[0]:.3f}, {auc_ci_ds[1]:.3f}]")
        print(f"    BACC = {bacc_ds:.3f}  [{bacc_ci_ds[0]:.3f}, {bacc_ci_ds[1]:.3f}]")

    return {"auc_ci": auc_ci, "bacc_ci": bacc_ci, "ci_per_ds": ci_per_ds}


# ==============================================================================
# Manifest Illumina completo (usato da knapsack + FPI)
# ==============================================================================

def step_load_manifest(cfg: PipelineConfig) -> dict:
    """CELLA 61 — Carica il manifest Illumina EPIC completo (IlmnID,
    UCSC_RefGene_Name, UCSC_RefGene_Group, Relation_to_UCSC_CpG_Island, CHR,
    MAPINFO, ...) usato sia dal knapsack che dal calcolo finale dell'FPI."""
    P = cfg.paths
    man = read_illumina_manifest_csv(P.MANIFEST_PATH)
    print(f"Manifest caricato: {man.shape}")
    return {"man": man}


# ==============================================================================
# STEP knapsack — CpG KNAPSACK MILP bi-obiettivo con sweep su mu [v4 paper]
# ==============================================================================

def step_knapsack(cfg: PipelineConfig, final_cpgs: np.ndarray, svm_final,
                   X_tr_fs: np.ndarray, X_te_fs: np.ndarray,
                   tissue_labels_tr: np.ndarray, tissue_labels_te: np.ndarray,
                   man: pd.DataFrame, cpg_to_chr: dict,
                   col_to_j_common: dict) -> dict:
    """CELLA 63 — CpG KNAPSACK MILP bi-obiettivo con sweep su mu [v4 paper].

    Produce: selected_cpgs_final (sweep mu+W_COSMIC), consensus_panel
    (k-fold), pareto_results (curva K, contiene anche K=30 usato per FPI).

    Genera i 3 plot del knapsack richiesti:
      - Pareto frontier su mu
      - Pareto su K (AUC + COSMIC enrichment)
      - Bar chart di stabilita' k-fold

    NOTA: qui `col_to_j` (usato per indicizzare X_tr_fs/X_te_fs, che sono
    GIA' filtrate sulle sole `final_cpgs`) e' costruito su `final_cpgs`
    stesso (coerente con `cols_all = FINAL_CPGS` del notebook originale).
    Questo e' indipendente dal bug corretto nello step SVM: qui lo spazio
    delle colonne di X_tr_fs/X_te_fs E' final_cpgs, quindi va bene.
    """
    ck = cfg.knapsack
    G = cfg.glob
    P = cfg.paths

    OUTDIR = Path(P.OUTPUT_DIR) / "knapsack_outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    def rank_norm_asc(v):
        v = np.asarray(v, float)
        v2 = v.copy()
        if np.isnan(v2).any():
            v2 = np.where(np.isnan(v2), np.nanmin(v2) - 1.0, v2)
        r = np.argsort(np.argsort(v2))
        return r / max(1, len(v2) - 1)

    def rank_norm_desc(v):
        return 1.0 - rank_norm_asc(v)

    def parse_first_gene(g):
        if g is None:
            return None
        s = str(g).strip()
        if s == "" or s.lower() == "nan":
            return None
        for sep in [";", ","]:
            if sep in s:
                s = s.split(sep)[0].strip()
                break
        return s if s != "" else None

    def in_promoter(feat):
        if feat is None:
            return False
        s = str(feat)
        return any(k in s for k in ["TSS200", "TSS1500", "1stExon", "5'UTR", "5UTR"])

    def in_island(ctx):
        if ctx is None:
            return False
        s = str(ctx)
        return "Island" in s and "Shore" not in s and "Shelf" not in s

    def jaccard(s1, s2):
        s1, s2 = set(s1), set(s2)
        if len(s1 | s2) == 0:
            return 1.0
        return len(s1 & s2) / len(s1 | s2)

    def knee_index(f1_vals, f2_vals):
        f1 = np.array(f1_vals, float)
        f2 = np.array(f2_vals, float)
        valid = np.ones(len(f2), dtype=bool)
        for i in range(1, len(f2)):
            if f2[i] == f2[i - 1]:
                valid[i] = False
        idx_v = np.where(valid)[0]
        if len(idx_v) < 2:
            return 0
        f1_n = (f1[idx_v] - f1[idx_v].min()) / max(f1[idx_v].max() - f1[idx_v].min(), 1e-12)
        f2_n = (f2[idx_v] - f2[idx_v].min()) / max(f2[idx_v].max() - f2[idx_v].min(), 1e-12)
        p0 = np.array([f2_n[0], f1_n[0]])
        p1 = np.array([f2_n[-1], f1_n[-1]])
        d = p1 - p0
        d_norm = d / max(np.linalg.norm(d), 1e-12)
        dists = [
            np.linalg.norm(
                (np.array([f2_n[i], f1_n[i]]) - p0)
                - np.dot(np.array([f2_n[i], f1_n[i]]) - p0, d_norm) * d_norm
            )
            for i in range(len(idx_v))
        ]
        return int(idx_v[np.argmax(dists)])

    def eval_pair(name, X, ybin):
        out = {}
        n0, n1 = int((ybin == 0).sum()), int((ybin == 1).sum())
        print(f"\n  [{name}]  n={X.shape[0]} (0={n0}, 1={n1})")
        if X.shape[0] < 10 or n0 == 0 or n1 == 0:
            print("  Campioni insufficienti.")
            return out
        try:
            out["silhouette"] = float(silhouette_score(X, ybin, metric="euclidean"))
            print(f"  Silhouette    : {out['silhouette']:.4f}")
        except Exception as e:
            out["silhouette"] = None
            print(f"  Silhouette    : N/A ({e})")
        try:
            Xsc = StandardScaler().fit_transform(X)
            lr = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=2000, random_state=G.RANDOM_STATE)
            n_cv = min(5, n0, n1)
            aucs = cross_val_score(lr, Xsc, ybin, cv=n_cv, scoring="roc_auc")
            out["auc_mean"] = float(aucs.mean())
            out["auc_std"] = float(aucs.std())
            print(f"  AUC CV{n_cv}       : {out['auc_mean']:.4f} \u00b1 {out['auc_std']:.4f}")
        except Exception as e:
            out["auc_mean"] = out["auc_std"] = None
            print(f"  AUC CV        : N/A ({e})")
        m0, m1 = X[ybin == 0].mean(0), X[ybin == 1].mean(0)
        out["mean_abs_dM"] = float(np.abs(m1 - m0).mean())
        print(f"  mean|DeltaM|  : {out['mean_abs_dM']:.4f}")
        return out

    # ── STEP 1 — Estrai coef SVM ──
    print("=" * 70)
    print("STEP 1 — Estrazione coef SVM")
    print("=" * 70)

    cols_all = np.array(final_cpgs, dtype=object)
    col_to_j = {c: j for j, c in enumerate(cols_all.tolist())}
    final_arr = np.array(final_cpgs, dtype=object)

    raw_coef = np.mean(
        [est.estimator.coef_[0]
         for est in svm_final.named_steps["clf"].calibrated_classifiers_],
        axis=0
    )
    assert raw_coef.size == final_arr.size
    coef_final = raw_coef

    print(f"FINAL_CPGS        : {final_arr.size}")
    print(f"coef range        : [{coef_final.min():.4f}, {coef_final.max():.4f}]")

    # ── STEP 3 — Score e |DeltaM| ──
    dm_tr_abs = (
        X_tr_fs[tissue_labels_tr == 1].mean(axis=0) -
        X_tr_fs[tissue_labels_tr == 0].mean(axis=0)
    )
    absdM_final = np.abs(dm_tr_abs)
    score_proxy = X_tr_fs.std(axis=0)
    score_final = score_proxy

    # ── STEP 4 — Pool top-N ──
    order_coef = np.argsort(-np.abs(coef_final))
    pool_idx = order_coef[:ck.N_POOL]
    pool_cpgs = final_arr[pool_idx]
    pool_coef = coef_final[pool_idx]
    pool_absdM = absdM_final[pool_idx]

    valid_mask = np.array([c in col_to_j for c in pool_cpgs], dtype=bool)
    pool_cpgs = pool_cpgs[valid_mask]
    pool_coef = pool_coef[valid_mask]
    pool_absdM = pool_absdM[valid_mask]
    pool_score = score_final[pool_idx][valid_mask]
    n_p = len(pool_cpgs)

    # ── STEP 5 — Conflict edges ──
    jj_pool = np.array([col_to_j[c] for c in pool_cpgs], dtype=int)
    Z = X_tr_fs[:, jj_pool].astype(np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0, ddof=1) + 1e-12)
    C_corr = (Z.T @ Z) / (Z.shape[0] - 1)
    i_u, j_u = np.where(np.triu(np.abs(C_corr) >= ck.CORR_THR, k=1))
    conflict_pairs = list(zip(i_u.tolist(), j_u.tolist()))
    print(f"Conflict edges    : {len(conflict_pairs):,}")

    # ── STEP 6 — Attributi genomici + COSMIC ──
    man_sub = man.loc[
        man["IlmnID"].isin(pool_cpgs.tolist()),
        ["IlmnID", "UCSC_RefGene_Name", "UCSC_RefGene_Group", "Relation_to_UCSC_CpG_Island"]
    ].copy()
    man_sub["IlmnID"] = man_sub["IlmnID"].astype(str)

    cpg_gene_dict = dict(zip(man_sub["IlmnID"], man_sub["UCSC_RefGene_Name"].fillna("")))
    cpg_feat_dict = dict(zip(man_sub["IlmnID"], man_sub["UCSC_RefGene_Group"].fillna("")))
    cpg_island_dict = dict(zip(man_sub["IlmnID"], man_sub["Relation_to_UCSC_CpG_Island"].fillna("")))

    chr_arr = np.array([cpg_to_chr.get(c, "") for c in pool_cpgs], dtype=object)
    gene_arr = np.array([parse_first_gene(cpg_gene_dict.get(c, None)) for c in pool_cpgs], dtype=object)
    is_prom = np.array([in_promoter(cpg_feat_dict.get(c, "")) for c in pool_cpgs], dtype=int)
    is_island_arr = np.array([in_island(cpg_island_dict.get(c, "")) for c in pool_cpgs], dtype=int)
    genes_pool = sorted({g for g in gene_arr.tolist() if g is not None})

    cgc = pd.read_csv(P.CGC_PATH, sep="\t")
    mask_breast = cgc["TUMOUR_TYPES_SOMATIC"].astype(str).str.contains("breast", case=False, na=False)
    COSMIC_BREAST_GENES = set(cgc.loc[mask_breast, "GENE_SYMBOL"].astype(str))

    is_cosmic = np.array(
        [1.0 if (gene_arr[i] is not None and gene_arr[i] in COSMIC_BREAST_GENES) else 0.0
         for i in range(n_p)], dtype=float
    )

    print(f"Geni distinti pool : {len(genes_pool)}")
    print(f"CpG COSMIC breast  : {int(is_cosmic.sum())} ({100*is_cosmic.mean():.1f}%)")

    # ── STEP 7 — Funzione valore ──
    def compute_value(w_cosmic):
        r_coef = rank_norm_asc(np.abs(pool_coef))
        r_dM = rank_norm_asc(pool_absdM)
        r_score = rank_norm_desc(pool_score)
        return ck.W_COEF * r_coef + ck.W_DM * r_dM + ck.W_SCORE * r_score + w_cosmic * is_cosmic

    # ── STEP 8 — Funzione MILP ──
    def solve_milp(mu: float, value: np.ndarray, k: int = ck.K, label: str = ""):
        F2_NORM = float(k * (2 + ck.ETA))
        prob = pulp.LpProblem(f"CpG_Knapsack_{label}_mu{mu:.3f}_k{k}", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_p)]
        y_g = {g: pulp.LpVariable(f"yg_{i}", cat="Binary") for i, g in enumerate(genes_pool)}

        f1_expr = pulp.lpSum(float(value[i]) * x[i] for i in range(n_p))
        f2_expr = (
            pulp.lpSum(x[i] for i in range(n_p) if is_prom[i])
            + pulp.lpSum(x[i] for i in range(n_p) if is_island_arr[i])
            + ck.ETA * pulp.lpSum(y_g[g] for g in genes_pool)
        )
        f2_hat = f2_expr / F2_NORM
        prob += f1_expr + mu * f2_hat

        prob += pulp.lpSum(x) == k, "C1"
        for idx, (i, j) in enumerate(conflict_pairs):
            prob += x[i] + x[j] <= 1, f"C2_{idx}"
        for ch in sorted(set(chr_arr.tolist()) - {""}):
            prob += (
                pulp.lpSum(x[i] for i in range(n_p) if chr_arr[i] == ch) <= ck.CHR_MAX, f"C3_{ch}"
            )
        for i in range(n_p):
            g = gene_arr[i]
            if g is not None and g in y_g:
                prob += x[i] <= y_g[g], f"C5_{i}"
        for g in genes_pool:
            idx_g = [i for i in range(n_p) if gene_arr[i] == g]
            if idx_g:
                prob += pulp.lpSum(x[i] for i in idx_g) <= ck.MAX_PER_GENE, f"C6_{g}"

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=ck.CBC_TIME_LIMIT)
        status = prob.solve(solver)
        status_str = pulp.LpStatus.get(status, str(status))
        obj_val = pulp.value(prob.objective)

        if obj_val is None or status_str not in ["Optimal", "Feasible"]:
            return None

        sel_idx = np.array([i for i in range(n_p) if (pulp.value(x[i]) or 0) > 0.5], dtype=int)
        if sel_idx.size != k:
            return None

        selected = pool_cpgs[sel_idx].tolist()
        f1_val = float(sum(value[i] for i in sel_idx))
        f2_val = float(
            is_prom[sel_idx].sum() + is_island_arr[sel_idx].sum()
            + ck.ETA * len({gene_arr[i] for i in sel_idx if gene_arr[i] is not None})
        )
        n_cosmic = int(is_cosmic[sel_idx].sum())

        return {
            "status": status_str, "f1": f1_val, "f2": f2_val, "f2_hat": f2_val / F2_NORM,
            "n_promoter": int(is_prom[sel_idx].sum()), "n_island": int(is_island_arr[sel_idx].sum()),
            "n_genes": len({gene_arr[i] for i in sel_idx if gene_arr[i] is not None}),
            "n_cosmic": n_cosmic, "selected": selected, "sel_idx": sel_idx,
        }

    # ── STEP 9 — Sweep principale mu ──
    print("\n" + "=" * 70)
    print(f"STEP 9 — Sweep mu  (K={ck.K}, W_COSMIC={ck.W_COSMIC})")
    print("=" * 70)

    value_main = compute_value(ck.W_COSMIC)
    sweep_results = []
    sel_mu0 = None
    all_optimal = True

    for mu in ck.MU_GRID:
        print(f"\n[mu={mu:.3f}] Solving...")
        res = solve_milp(mu, value_main, k=ck.K, label="main")
        if res is None:
            sweep_results.append({"mu": mu, "f1": np.nan, "f2": np.nan,
                                   "status": "FAILED", "selected": []})
            print(f"  status=FAILED")
            all_optimal = False
            continue
        if mu == 0.0:
            sel_mu0 = res["selected"]
        jacc = jaccard(res["selected"], sel_mu0) if sel_mu0 else 1.0
        row = {"mu": mu, **res, "jaccard_vs_0": jacc}
        sweep_results.append(row)
        if res["status"] != "Optimal":
            all_optimal = False
        print(f"  status={res['status']}  f1={res['f1']:.4f}  f2={res['f2']:.4f}  "
              f"Jaccard(0)={jacc:.4f}  COSMIC={res['n_cosmic']}")

    print(f"\n{'OK — tutti i mu hanno raggiunto convergenza (Optimal)' if all_optimal else 'ATTENZIONE — alcuni mu non hanno raggiunto convergenza'}")

    valid_res = [r for r in sweep_results if not np.isnan(r.get("f1", np.nan))]
    k_idx = knee_index([r["f1"] for r in valid_res], [r["f2"] for r in valid_res])
    knee_res = valid_res[k_idx]
    knee_mu = knee_res["mu"]
    print(f"Knee: mu={knee_mu:.3f}  COSMIC={knee_res['n_cosmic']}")

    # ── PLOT 1/5 — Pareto frontier mu ──
    if len(valid_res) >= 2:
        f1_arr = np.array([r["f1"] for r in valid_res])
        f2_arr = np.array([r["f2"] for r in valid_res])
        mu_arr = [r["mu"] for r in valid_res]

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sc = ax.scatter(f2_arr, f1_arr, c=np.log1p(mu_arr), cmap="viridis", s=60, zorder=3)
        ax.plot(f2_arr, f1_arr, color="gray", lw=1.0, ls="--", zorder=2)
        ax.scatter(f2_arr[k_idx], f1_arr[k_idx], marker="*", s=220,
                   color=plt.cm.viridis(0.15), zorder=4, label=f"Knee mu={knee_mu:.3f}")
        for i, mu in enumerate(mu_arr):
            ax.annotate(f"mu={mu:.2f}", (f2_arr[i], f1_arr[i]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        plt.colorbar(sc, ax=ax, label="log(1+mu)")
        ax.set_xlabel("f2 (diversification)")
        ax.set_ylabel("f1 (composite score)")
        ax.set_title("Pareto frontier sweep")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(OUTDIR / "knapsack_pareto_mu.pdf", dpi=300, bbox_inches="tight")
        plt.show()
        print("Salvato: knapsack_pareto_mu.pdf")

    # ── STEP 10 — Knee + sweep W_COSMIC ──
    print("\n" + "=" * 70)
    print(f"STEP 10 — Sweep W_COSMIC (mu=knee={knee_mu})")
    print("=" * 70)

    cosmic_sweep = []
    for wc in ck.W_COSMIC_GRID:
        val_wc = compute_value(wc)
        res = solve_milp(knee_mu, val_wc, k=ck.K, label=f"wc{wc:.2f}")
        if res is None:
            cosmic_sweep.append({"w_cosmic": wc, "f1": np.nan, "n_cosmic": np.nan, "selected": []})
            print(f"  W_COSMIC={wc:.2f}  status=FAILED")
            continue
        cosmic_sweep.append({"w_cosmic": wc, **res})
        print(f"  W_COSMIC={wc:.2f}  status={res['status']}  f1={res['f1']:.4f}  COSMIC={res['n_cosmic']}")

    wc_valid = [r for r in cosmic_sweep if not np.isnan(r.get("f1", np.nan))]
    if wc_valid:
        f1_wc0 = next((r["f1"] for r in wc_valid if r["w_cosmic"] == 0.0), wc_valid[0]["f1"])
        candidates = [r for r in wc_valid if r["f1"] >= f1_wc0 * 0.999]
        best_wc_res = max(candidates, key=lambda r: r["n_cosmic"]) if candidates else wc_valid[-1]
        best_w_cosmic = best_wc_res["w_cosmic"]
        selected_cpgs_final = best_wc_res["selected"]
        print(f"\nW_COSMIC ottimale: {best_w_cosmic:.2f}  COSMIC={best_wc_res['n_cosmic']}")
    else:
        best_w_cosmic = ck.W_COSMIC
        selected_cpgs_final = knee_res["selected"]

    # ── STEP 11 — Curva Pareto K ──
    print("\n" + "=" * 70)
    print(f"STEP 11 — Curva Pareto K in {ck.K_PARETO_GRID}")
    print("=" * 70)

    pareto_results = []
    val_pareto = compute_value(best_w_cosmic)

    for k_val in ck.K_PARETO_GRID:
        print(f"\n[K={k_val}] Solving (mu={knee_mu}, W_COSMIC={best_w_cosmic})...")
        res = solve_milp(knee_mu, val_pareto, k=k_val, label=f"pareto_k{k_val}")
        if res is None:
            print(f"  [K={k_val}] status=FAILED")
            pareto_results.append({"k": k_val, "auc": np.nan, "n_cosmic": np.nan,
                                    "n_genes": np.nan, "selected": []})
            continue

        jj_k = np.array([col_to_j[c] for c in res["selected"] if c in col_to_j], dtype=int)
        X_te_k = X_te_fs[:, jj_k]
        pipe_k = Pipeline([("sc", StandardScaler()),
                           ("svm", LinearSVC(C=0.01, max_iter=10000, random_state=G.RANDOM_STATE))])
        pipe_k.fit(X_tr_fs[:, jj_k], tissue_labels_tr)
        auc_k = roc_auc_score(tissue_labels_te, pipe_k.decision_function(X_te_k))

        pareto_results.append({
            "k": k_val, "auc": auc_k, "n_cosmic": res["n_cosmic"], "n_genes": res["n_genes"],
            "f1": res["f1"], "f2": res["f2"], "selected": res["selected"],
        })
        print(f"  [K={k_val}] status={res['status']}  AUC={auc_k:.4f}  "
              f"COSMIC={res['n_cosmic']}  Geni={res['n_genes']}")

    # ── PLOT 2/5 — Pareto K (AUC + COSMIC) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    k_vals = [r["k"] for r in pareto_results if not np.isnan(r["auc"])]
    auc_vals = [r["auc"] for r in pareto_results if not np.isnan(r["auc"])]
    cos_vals = [r["n_cosmic"] for r in pareto_results if not np.isnan(r["auc"])]

    axes[0].plot(k_vals, auc_vals, marker="o", color=plt.cm.viridis(0.6), lw=1.8)
    axes[0].axvline(ck.K, ls="--", color=plt.cm.viridis(0.15), lw=1.4, label=f"K={ck.K} (default)")
    axes[0].set_xlabel("K (panel size)")
    axes[0].set_ylabel("AUC (test set)")
    axes[0].set_title("AUC vs K")
    axes[0].legend()

    axes[1].plot(k_vals, cos_vals, marker="o", color=plt.cm.viridis(0.85), lw=1.8)
    axes[1].axvline(ck.K, ls="--", color=plt.cm.viridis(0.15), lw=1.4, label=f"K={ck.K} (default)")
    axes[1].set_xlabel("K (panel size)")
    axes[1].set_ylabel("COSMIC breast hits")
    axes[1].set_title("COSMIC enrichment vs K")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTDIR / "knapsack_pareto_K.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    print("Salvato: knapsack_pareto_K.pdf")

    # ── STEP 12 — K-fold stabilita' ──
    print("\n" + "=" * 70)
    print(f"STEP 12 — K-fold stabilita' (N_FOLDS={ck.N_FOLDS_STAB}, K={ck.K})")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=ck.N_FOLDS_STAB, shuffle=True, random_state=G.RANDOM_STATE)
    cpg_selection_count = {c: 0 for c in pool_cpgs.tolist()}
    fold_convergence = []

    X_tr_pool_stab = X_tr_fs
    y_tr_pool_stab = tissue_labels_tr

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_tr_pool_stab, y_tr_pool_stab)):
        print(f"\n[Fold {fold_i+1}/{ck.N_FOLDS_STAB}]")

        pipe_fold = Pipeline([
            ("sc", StandardScaler()),
            ("svm", LinearSVC(C=0.1, max_iter=10000, random_state=G.RANDOM_STATE)),
        ])
        pipe_fold.fit(X_tr_pool_stab[tr_idx][:, jj_pool], y_tr_pool_stab[tr_idx])
        coef_fold = pipe_fold.named_steps["svm"].coef_[0]

        r_coef_f = rank_norm_asc(np.abs(coef_fold))
        r_dM_f = rank_norm_asc(pool_absdM)
        r_score_f = rank_norm_desc(pool_score)
        value_fold = (ck.W_COEF * r_coef_f + ck.W_DM * r_dM_f +
                      ck.W_SCORE * r_score_f + best_w_cosmic * is_cosmic)

        res_fold = solve_milp(knee_mu, value_fold, k=ck.K, label=f"fold{fold_i}")
        if res_fold is None:
            print(f"  Fold {fold_i+1}: status=FAILED")
            fold_convergence.append(False)
            continue

        fold_convergence.append(res_fold["status"] == "Optimal")
        for c in res_fold["selected"]:
            if c in cpg_selection_count:
                cpg_selection_count[c] += 1
        print(f"  Fold {fold_i+1}: status={res_fold['status']}  "
              f"COSMIC={res_fold['n_cosmic']}  Geni={res_fold['n_genes']}")

    n_optimal = sum(fold_convergence)
    print(f"\nConvergenza k-fold: {n_optimal}/{ck.N_FOLDS_STAB} fold hanno raggiunto l'ottimo")
    if n_optimal < ck.N_FOLDS_STAB:
        print("\u26a0 Considera di aumentare ulteriormente CBC_TIME_LIMIT")

    sel_freq = np.array([cpg_selection_count[c] / ck.N_FOLDS_STAB
                         for c in pool_cpgs.tolist()], dtype=float)
    stable_mask = sel_freq >= 0.6
    stable_cpgs = pool_cpgs[stable_mask].tolist()
    print(f"\nCpG stabili (freq>=0.6): {len(stable_cpgs)}")

    order_freq = np.argsort(-sel_freq)
    consensus_panel = pool_cpgs[order_freq[:ck.K]].tolist()
    print(f"Consensus panel (top-{ck.K} per frequenza): {len(consensus_panel)} CpG")

    freq_consensus = sel_freq[order_freq[:ck.K]]
    print(f"  freq range: [{freq_consensus.min():.2f}, {freq_consensus.max():.2f}]  "
          f"media: {freq_consensus.mean():.2f}")

    # ── PLOT 3/5 — Stabilita' k-fold ──
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(ck.K), np.sort(freq_consensus)[::-1],
           color=plt.cm.viridis(0.6), edgecolor="none")
    ax.axhline(0.6, ls="--", color=plt.cm.viridis(0.15), lw=1.2, label="soglia 0.6")
    ax.set_xlabel("CpG rank (per frequenza)")
    ax.set_ylabel("Frequenza selezione")
    ax.set_title(f"Stabilita' K-fold ({ck.N_FOLDS_STAB} fold) — top-{ck.K} CpG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "knapsack_kfold_stability.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    print("Salvato: knapsack_kfold_stability.pdf")

    # ── STEP 13 — Evaluation finale ──
    print("\n" + "=" * 70)
    print("STEP 13 — Evaluation finale")
    print("=" * 70)

    eval_results = {}
    for panel_name, panel in [("sweep_final", selected_cpgs_final),
                               ("kfold_consensus", consensus_panel)]:
        jj = np.array([col_to_j[c] for c in panel if c in col_to_j], dtype=int)
        X_te_p = X_te_fs[:, jj]
        mask_na = np.isin(tissue_labels_te, [0, 1])
        eval_results[panel_name] = eval_pair(
            f"{panel_name} — N vs A (test)",
            X_te_p[mask_na], (tissue_labels_te[mask_na] == 1).astype(int))

    print("\n" + "=" * 70)
    print("OUTPUT FINALE")
    print("=" * 70)
    print(f"  selected_cpgs_final  : {len(selected_cpgs_final)} CpG  (sweep)")
    print(f"  consensus_panel      : {len(consensus_panel)} CpG  (k-fold)")
    print(f"  knee_mu              : {knee_mu}")
    print(f"  best_w_cosmic        : {best_w_cosmic}")
    print(f"  convergenza sweep    : {'OK Optimal' if all_optimal else 'ATTENZIONE non tutti Optimal'}")
    print(f"  convergenza k-fold   : {n_optimal}/{ck.N_FOLDS_STAB} Optimal")
    print(f"  OUTDIR               : {OUTDIR}")
    print("=" * 70)

    selected_cpgs_with_cosmic = list(selected_cpgs_final)

    return {
        "selected_cpgs_final": selected_cpgs_final,
        "selected_cpgs_with_cosmic": selected_cpgs_with_cosmic,
        "consensus_panel": consensus_panel,
        "pareto_results": pareto_results,
        "sweep_results": sweep_results,
        "cosmic_sweep": cosmic_sweep,
        "knee_mu": knee_mu,
        "best_w_cosmic": best_w_cosmic,
        "eval_results": eval_results,
        "col_to_j_final_cpgs": col_to_j,
    }


def extract_selected_30(pareto_results: list) -> dict:
    """CELLA 64 — Estrae le 30 CpG finali dal Pareto K del knapsack (K=30)."""
    res_k30 = next((r for r in pareto_results if r["k"] == 30), None)
    if res_k30 is None:
        raise ValueError("K=30 non trovato in pareto_results — assicurati di "
                          "aver eseguito step_knapsack con 30 in K_PARETO_GRID.")
    selected_30 = [str(x) for x in res_k30["selected"]]
    print(f"selected_30 definita: {len(selected_30)} CpG")
    print("Esempio:", selected_30[:5])
    return {"selected_30": selected_30}


# ==============================================================================
# STEP finale — Field Progression Index (FPI, Eq. 5) + score biologico s_bio (Eq. 4)
# ==============================================================================

def step_fpi_final(cfg: PipelineConfig, selected_30: list, common_cpgs: list,
                    cpg_cols_current: np.ndarray, X_tr_combat: np.ndarray,
                    X_te_combat: np.ndarray, tissue_labels_tr: np.ndarray,
                    tissue_labels_te: np.ndarray, X_tum_combat_mspace: np.ndarray,
                    mean_normal_train: dict, man: pd.DataFrame) -> dict:
    """Versione CORRETTA (cella 65 del notebook originale) del calcolo finale:
      FPI_j   = (M_j^A - M_j^N) / (M_j^T - M_j^N + eps)   (Eq. 5), clip [0,1]
      s_bio,j = 0.5*(1 - FPI_j) + 0.5*(1 - w_bio,j)        (Eq. 4)

    A differenza della cella "ultima" del notebook (che e' errata), questa
    gestisce correttamente il fatto che X_tum_combat (qui passato come
    X_tum_combat_mspace, gia' in M-value dopo lo step di allineamento) vive
    in una scala diversa da X_tr_combat/X_te_combat (deviazione assoluta beta):
    lo riconverte in beta e poi in deviazione assoluta, prima di calcolare
    M_j^T, cosi' che sia comparabile con M_j^N e M_j^A.

    `cols` qui e' `cpg_cols_current` (le ~270k CpG post-Edgar/variance-filter,
    spazio nativo di X_tum_combat_mspace).
    """
    cfpi = cfg.fpi
    P = cfg.paths

    cols = cpg_cols_current
    X_tum_combat = X_tum_combat_mspace  # alias per leggibilita', come nel notebook

    OUTDIR = Path(P.OUTPUT_DIR) / "fpi_outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cpgs_30 = [str(c) for c in selected_30]
    use_corrected = getattr(cfg.glob, "FPI_USE_CORRECTED_VERSION", True)

    if use_corrected:
        # ── STEP 1a — Indici delle 30 CpG in common_cpgs (spazio Normal/Adjacent) ──
        col_to_j_common = {c: j for j, c in enumerate(common_cpgs)}
        missing_na = [c for c in cpgs_30 if c not in col_to_j_common]
        if missing_na:
            print(f"⚠ {len(missing_na)} CpG non trovate in common_cpgs (Normal/Adjacent): {missing_na[:5]}")

        # ── STEP 1b — Indici delle 30 CpG in cols/cpg_cols_current (spazio Tumour) ──
        col_to_j_270k = {c: j for j, c in enumerate(np.asarray(cols, dtype=object).tolist())}
        missing_tum = [c for c in cpgs_30 if c not in col_to_j_270k]
        if missing_tum:
            print(f"⚠ {len(missing_tum)} CpG non trovate in cpg_cols_current (Tumour): {missing_tum[:5]}")

        cpgs_30 = [c for c in cpgs_30 if c in col_to_j_common and c in col_to_j_270k]
        idx_30_common = np.array([col_to_j_common[c] for c in cpgs_30], dtype=int)
        idx_30_270k = np.array([col_to_j_270k[c] for c in cpgs_30], dtype=int)

        print(f"CpG finali utilizzabili per FPI: {len(cpgs_30)} / {len(selected_30)}")

        # ── STEP 2a — Medie Normal/Adjacent (deviazione assoluta beta, pool train+test) ──
        X_NA_pool = np.vstack([X_tr_combat, X_te_combat])
        labels_pool = np.concatenate([tissue_labels_tr, tissue_labels_te])

        mean_N = X_NA_pool[labels_pool == 0][:, idx_30_common].mean(axis=0)
        mean_A = X_NA_pool[labels_pool == 1][:, idx_30_common].mean(axis=0)

        # ── STEP 2b — Tumour: inverti M-value -> beta, poi -> deviazione assoluta ──
        idx_common_for_270k = np.array(
            [col_to_j_common[c] for c in np.asarray(cols, dtype=object).tolist()], dtype=int
        )
        mean_nor_225_270k = mean_normal_train["GSE225845"][idx_common_for_270k]
        mean_nor_287_270k = mean_normal_train["GSE287331"][idx_common_for_270k]
        mean_nor_pool_270k = (mean_nor_225_270k + mean_nor_287_270k) / 2.0

        X_tum_beta = m_to_beta(X_tum_combat, eps=cfpi.EPS_M)
        X_tum_abs = np.abs(X_tum_beta - mean_nor_pool_270k)

        mean_T = X_tum_abs[:, idx_30_270k].mean(axis=0)

    else:
        # Versione originale del notebook (ultima cella): NON inverte
        # M-value -> beta per il Tumour, tratta X_tum_combat come se fosse
        # gia' nello stesso spazio "deviazione assoluta beta" di
        # X_tr_combat/X_te_combat.
        col_to_j_common = {c: j for j, c in enumerate(common_cpgs)}
        missing = [c for c in cpgs_30 if c not in col_to_j_common]
        if missing:
            print(f"⚠ {len(missing)} CpG non trovate in common_cpgs (escluse): {missing[:5]}")
        cpgs_30 = [c for c in cpgs_30 if c in col_to_j_common]
        idx_30_common = np.array([col_to_j_common[c] for c in cpgs_30], dtype=int)

        print(f"CpG finali utilizzabili per FPI: {len(cpgs_30)} / {len(selected_30)}")

        X_NA_pool = np.vstack([X_tr_combat, X_te_combat])
        labels_pool = np.concatenate([tissue_labels_tr, tissue_labels_te])

        mean_N = X_NA_pool[labels_pool == 0][:, idx_30_common].mean(axis=0)
        mean_A = X_NA_pool[labels_pool == 1][:, idx_30_common].mean(axis=0)
        mean_T = X_tum_combat[:, idx_30_common].mean(axis=0)

    # ── STEP 3 — FPI (Eq. 5) ──
    num_fpi = mean_A - mean_N
    denom_fpi = mean_T - mean_N
    denom_safe = np.where(
        np.abs(denom_fpi) < cfpi.EPS_FPI,
        np.sign(denom_fpi + 1e-12) * cfpi.EPS_FPI,
        denom_fpi
    )
    FPI_raw = num_fpi / denom_safe
    FPI = np.clip(FPI_raw, 0.0, 1.0)

    # ── STEP 4 — peso genomico w_bio (CpG island / promoter-proximal) ──
    man_sub = man.loc[
        man["IlmnID"].astype(str).isin(cpgs_30),
        ["IlmnID", "Relation_to_UCSC_CpG_Island", "UCSC_RefGene_Group"]
    ].copy()
    man_sub["IlmnID"] = man_sub["IlmnID"].astype(str)
    island_map = dict(zip(man_sub["IlmnID"], man_sub["Relation_to_UCSC_CpG_Island"].fillna("")))
    feature_map = dict(zip(man_sub["IlmnID"], man_sub["UCSC_RefGene_Group"].fillna("")))

    def _is_island(c):
        return island_map.get(c, "") == "Island"

    def _is_promoter(c):
        return "TSS" in str(feature_map.get(c, ""))

    w_bio = np.array([
        1.00 if (_is_island(c) and _is_promoter(c)) else
        0.70 if _is_island(c) else
        0.50 if _is_promoter(c) else
        0.20
        for c in cpgs_30
    ], dtype=np.float64)

    # ── STEP 5 — score biologico (Eq. 4) ──
    s_bio = 0.5 * (1.0 - FPI) + 0.5 * (1.0 - w_bio)

    # ── STEP 6 — Tabella riassuntiva ──
    fpi_table = pd.DataFrame({
        "CpG": cpgs_30, "M_Normal": mean_N, "M_Adjacent": mean_A, "M_Tumour": mean_T,
        "FPI_raw": FPI_raw, "FPI": FPI, "w_bio": w_bio, "s_bio": s_bio,
    }).sort_values("FPI").reset_index(drop=True)

    print("\n" + "=" * 70)
    print("FPI — tabella riassuntiva (CpG finali, ordinate per FPI crescente)")
    print("=" * 70)
    print(fpi_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = OUTDIR / "fpi_30_cpgs.csv"
    fpi_table.to_csv(csv_path, index=False)
    print(f"\nSalvato: {csv_path}")

    # ── PLOT 4/5 (incluso nel set "principale") — asse di progressione Normal -> Tumour ──
    if "apply_thesis_style" in globals():
        apply_thesis_style(use_tex=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = fpi_table.sort_values("FPI").reset_index(drop=True)
    y_pos = np.arange(len(plot_df))

    cmap = plt.cm.viridis

    ax.axvline(0.0, color="gray", lw=1.2, ls="--", zorder=1)
    ax.axvline(1.0, color="gray", lw=1.2, ls="--", zorder=1)
    ax.text(0.0, len(plot_df), "Normal", ha="center", va="bottom", fontsize=9)
    ax.text(1.0, len(plot_df), "Tumour", ha="center", va="bottom", fontsize=9)

    ax.hlines(y_pos, xmin=0, xmax=plot_df["FPI"], color="#D0D0D0", lw=1.0, zorder=2)
    sc = ax.scatter(
        plot_df["FPI"], y_pos, c=plot_df["FPI"], cmap=cmap, vmin=0, vmax=1,
        s=70, edgecolor="white", linewidth=0.6, zorder=3
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["CpG"], fontsize=7.5)
    ax.set_xlabel(r"Field Progression Index ($\mathrm{FPI}_j$)")
    ax.set_ylabel("CpG")
    ax.set_title("Posizione delle CpG finali lungo l'asse Normal\u2013Tumour")
    ax.set_xlim(-0.05, 1.05)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"FPI$_j$ (0 = Normal, 1 = Tumour)")

    fig.tight_layout()
    fig_path = OUTDIR / "fpi_30_cpgs_axis.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Salvato: {fig_path}")

    return {"fpi_table": fpi_table}


# ==============================================================================
# ORCHESTRAZIONE — Parte 1: tutto cio' che precede lo STEP 3 (costoso,
# deterministico, NON dipende dai parametri che vuoi variare). Va eseguito
# una sola volta; il risultato viene salvato su disco (cache) e ricaricato
# dalle run successive.
# ==============================================================================

_STEP3_INPUT_KEYS = [
    "M_train", "y_train", "cpg_cols_current", "bio_weight_norm",
    "X_tum_combat_mspace",
    # in piu', tutto cio' che serve agli step DOPO lo step 3 (svm, knapsack, fpi):
    "common_cpgs", "X_tr_combat", "X_te_combat", "tissue_labels_tr",
    "tissue_labels_te", "ds_labels_te", "mean_normal_train", "man",
    "M_test", "y_test",
]


def run_until_step3_inputs(cfg: PipelineConfig, force_recompute: bool = False) -> dict:
    """Esegue STEP 0 -> STEP 2.5 (tutto cio' che precede lo STEP 3) e salva il
    risultato in cache su disco (cfg.paths.ARTIFACTS_DIR). Se la cache esiste
    gia' e `force_recompute=False`, la ricarica invece di rieseguire tutto
    (utile per non rifare ComBat/caricamento beta ad ogni esperimento)."""
    cache_path = Path(cfg.paths.ARTIFACTS_DIR) / "step3_inputs.pkl"

    if cache_path.exists() and not force_recompute:
        print(f"[cache] Trovata cache esistente: {cache_path}. Ricarico...")
        return load_pickle(cache_path)

    print("[cache] Nessuna cache trovata (o force_recompute=True). "
          "Eseguo la pipeline da STEP 0...")

    out = {}

    r = step0_build_meta(cfg)
    out.update(r)

    r = step0_train_test_split(cfg, out["meta_225"], out["meta_287"])
    out.update(r)

    r = step1_common_cpgs(cfg)
    out.update(r)

    r = step1b_load_beta(cfg, out["common_cpgs"], out["meta_train"], out["meta_test"],
                          out["meta_tumor"])
    out.update(r)

    r = step2_absolute_deviation(cfg, out["meta_train"], out["X_train"], out["X_test"],
                                  out["X_tumor"])
    out.update(r)

    r = step_combat(cfg, out["common_cpgs"], out["meta_train"], out["X_train_abs"],
                     out["X_test_abs"], out["X_tumor_abs"])
    out.update(r)

    r = step_tissue_labels_te(cfg, out["meta_test"])
    out.update(r)

    r = step_edgar_feature_selection(cfg, out["common_cpgs"], out["X_tr_combat"],
                                      out["X_te_combat"], out["tissue_labels_tr"],
                                      out["tissue_labels_te"])
    out.update(r)

    r = step2_m_transform_variance_filter(cfg, out["B_train"], out["B_test"],
                                           out["selected_cpgs"])
    out.update(r)

    r = step2_5_bio_weight(cfg, out["cpg_cols_current"])
    out.update(r)

    r = step_align_tumor_to_current(cfg, out["X_tum_combat"], out["common_cpgs"],
                                     out["cpg_cols_current"])
    out.update(r)

    r = step_load_manifest(cfg)
    out.update(r)

    # rinomina y_train/y_test coerenti con tissue_labels_tr/te (post-Edgar)
    out["y_train"] = out["tissue_labels_tr"]
    out["y_test"] = out["tissue_labels_te"]

    # salva in cache solo le chiavi necessarie ai run successivi
    cache_payload = {k: out[k] for k in _STEP3_INPUT_KEYS if k in out}
    save_pickle(cache_payload, cache_path)
    print(f"[cache] Salvato in: {cache_path}")

    return cache_payload


# ==============================================================================
# ORCHESTRAZIONE — Parte 2: STEP 3 in poi (quello che vuoi variare).
# Prende in input la cache prodotta da run_until_step3_inputs() e i parametri
# correnti (cfg.step3, ed eventualmente cfg.step3b/step4/step5/step6/knapsack),
# ed esegue tutta la pipeline dallo STEP 3 fino all'FPI finale.
# ==============================================================================

def run_step3_and_downstream(cfg: PipelineConfig, cached_inputs: dict,
                              run_knapsack: bool = True, run_fpi: bool = True) -> dict:
    """Riesegue STEP 3 -> STEP 3b -> STEP 4 -> STEP 5 -> STEP 6 -> SVM ->
    (opzionale) knapsack -> (opzionale) FPI finale, usando i parametri
    correnti in `cfg`. Pensata per essere chiamata dentro un ciclo `for`
    variando `cfg.step3` (e affini) tra le iterazioni.
    """
    out = dict(cached_inputs)  # copia leggera, non mutiamo la cache originale

    r = step3_rskf_score(cfg, out["M_train"], out["y_train"], out["cpg_cols_current"],
                          out["bio_weight_norm"], out["X_tum_combat_mspace"])
    out.update(r)

    r = step3b_region_anchoring(cfg, out["cpg_candidates"], out["step3_stats"])
    out.update(r)

    r = step4_correlation_clustering(cfg, out["M_train"], out["cpg_cols_current"],
                                      out["poolA_cpgs"], out["poolB_cpgs"], out["score_map"])
    out.update(r)

    r = step5_diversification(cfg, out["reps_union"], out["score_map"])
    out.update(r)

    r = step6_final_k(cfg, out["selected_diverse"])
    out.update(r)

    r = step_svm_train(cfg, out["common_cpgs"], out["cpg_cols_current"], out["final_cpgs"],
                        out["X_tr_combat"], out["X_te_combat"], out["tissue_labels_tr"])
    out.update(r)

    r = step_svm_eval(cfg, out["svm_final"], out["X_te_fs"], out["tissue_labels_te"],
                       out["ds_labels_te"])
    out.update(r)

    r = step_svm_bootstrap_ci(cfg, out["tissue_labels_te"], out["proba_te"], out["pred_te"],
                               out["auc_overall"], out["bacc_overall"], out["ds_labels_te"])
    out.update(r)

    if run_knapsack:
        r = step_knapsack(cfg, out["final_cpgs"], out["svm_final"], out["X_tr_fs"],
                           out["X_te_fs"], out["tissue_labels_tr"], out["tissue_labels_te"],
                           out["man"], out["cpg_to_chr"], out["col_to_j_common"])
        out.update(r)

        r = extract_selected_30(out["pareto_results"])
        out.update(r)

        if run_fpi:
            r = step_fpi_final(cfg, out["selected_30"], out["common_cpgs"],
                                out["cpg_cols_current"], out["X_tr_combat"],
                                out["X_te_combat"], out["tissue_labels_tr"],
                                out["tissue_labels_te"], out["X_tum_combat_mspace"],
                                out["mean_normal_train"], out["man"])
            out.update(r)

    return out


# ==============================================================================
# MAIN — esecuzione singola (parametri di default)
# ==============================================================================

def main():
    """Esegue l'intera pipeline una volta, con i parametri di default
    (identici al notebook originale)."""
    cfg = PipelineConfig()

    cached_inputs = run_until_step3_inputs(cfg, force_recompute=False)
    results = run_step3_and_downstream(cfg, cached_inputs, run_knapsack=True, run_fpi=True)

    print("\n\n" + "#" * 90)
    print("RIEPILOGO FINALE")
    print("#" * 90)
    print(f"CpG candidate (STEP 3)      : {len(results['cpg_candidates']):,}")
    print(f"final_cpgs (STEP 6, K={cfg.step6.K_FINAL}) : {len(results['final_cpgs']):,}")
    print(f"AUC test (overall)          : {results['auc_overall']:.3f}  "
          f"[{results['auc_ci'][0]:.3f}, {results['auc_ci'][1]:.3f}]")
    print(f"BACC test (overall)         : {results['bacc_overall']:.3f}  "
          f"[{results['bacc_ci'][0]:.3f}, {results['bacc_ci'][1]:.3f}]")
    if "selected_30" in results:
        print(f"selected_30 (FPI panel)     : {len(results['selected_30'])} CpG")
    print("#" * 90)

    return results


# ==============================================================================
# ESEMPIO — Ciclo for sui parametri dello score stat/bio (STEP 3)
# ==============================================================================
#
# Questo e' l'uso che hai in mente: variare W_BIO / W_IP / EPS_IP (o altri
# parametri di Step3Config) e vedere come cambiano i risultati finali, SENZA
# ripetere ogni volta i passaggi costosi (caricamento beta, ComBat, ...).
#
# Strategia:
#   1) `cached_inputs` viene calcolato una sola volta FUORI dal ciclo.
#   2) per ogni combinazione di parametri, si crea una NUOVA PipelineConfig
#      con cfg.step3 modificato, e si chiama run_step3_and_downstream().
#   3) i risultati salienti di ogni run vengono accumulati in una tabella
#      (sweep_summary) che puoi ispezionare/plottare a sweep finito.
#
# NOTA: ogni iterazione rilancia comunque lo STEP 3 (RSKF su GPU, costoso) e
# tutto cio' che segue, incluso il knapsack MILP (anche questo costoso, ha un
# CBC_TIME_LIMIT di default di 600s per ogni mu/K testato). Se vuoi sweep
# rapidi, valuta `run_knapsack=False` per isolare l'effetto dei parametri
# dello STEP 3 sul pannello di CpG candidate, prima di lanciare l'intera
# pipeline a valle solo sulle combinazioni che ti interessano davvero.
# ==============================================================================

def example_parameter_sweep():
    base_cfg = PipelineConfig()

    # STEP costoso, eseguito una sola volta (usa la cache se gia' presente).
    cached_inputs = run_until_step3_inputs(base_cfg, force_recompute=False)

    # Combinazioni di parametri da testare: cambia questa lista a piacere.
    # Esempio: variare il peso del blocco biologico nello score finale (W_BIO)
    # e il peso interno IP-vs-bio_weight (W_IP).
    param_grid = [
        {"W_BIO": 0.00, "W_IP": 1.00},
        {"W_BIO": 0.10, "W_IP": 0.90},
        {"W_BIO": 0.20, "W_IP": 0.80},
        {"W_BIO": 0.30, "W_IP": 0.70},
        {"W_BIO": 0.40, "W_IP": 0.60},
        {"W_BIO": 0.50, "W_IP": 0.50},   # = default
        {"W_BIO": 0.60, "W_IP": 0.40},
        {"W_BIO": 0.70, "W_IP": 0.30},
        {"W_BIO": 0.80, "W_IP": 0.20},
        {"W_BIO": 0.90, "W_IP": 0.10},
        {"W_BIO": 1.00, "W_IP": 0.00},
    ]

    sweep_summary = []

    for i, params in enumerate(param_grid):
        print("\n\n" + "=" * 100)
        print(f"[SWEEP {i+1}/{len(param_grid)}] params = {params}")
        print("=" * 100)

        # nuova config con i SOLI parametri di step3 modificati
        cfg = PipelineConfig()
        cfg.step3 = Step3Config(**{**asdict(base_cfg.step3), **params})

        results = run_step3_and_downstream(cfg, cached_inputs, run_knapsack=True, run_fpi=True)

        sweep_summary.append({
            **params,
            "n_candidates_step3": len(results["cpg_candidates"]),
            "n_final_cpgs": len(results["final_cpgs"]),
            "auc_overall": results["auc_overall"],
            "bacc_overall": results["bacc_overall"],
            "auc_ci_lo": results["auc_ci"][0],
            "auc_ci_hi": results["auc_ci"][1],
            "knee_mu": results.get("knee_mu"),
            "best_w_cosmic": results.get("best_w_cosmic"),
            "n_selected_30": len(results.get("selected_30", [])),
        })

    sweep_df = pd.DataFrame(sweep_summary)
    print("\n\n" + "#" * 90)
    print("SWEEP SUMMARY")
    print("#" * 90)
    print(sweep_df.to_string(index=False))

    out_path = Path(base_cfg.paths.OUTPUT_DIR) / "parameter_sweep_summary.csv"
    sweep_df.to_csv(out_path, index=False)
    print(f"\nSalvato: {out_path}")

    return sweep_df


if __name__ == "__main__":
    main()

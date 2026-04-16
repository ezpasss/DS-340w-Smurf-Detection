"""
Combined Smurf Detection Pipeline
===================================
Runs two independent smurf detection methods on the same dataset:
  1. LSTM Model  (Conv1D → LSTM ensemble: raw / cut / pool variants)
  2. KMeans+PCA  (Feature engineering → StandardScaler → PCA → Gap Stats → KMeans → IQR)

Each method produces a per-player {puuid: "Smurf" | "Honest"} verdict.
The results are then cross-compared:
  - Both flagged  → "Confirmed Smurf" (true smurf)
  - LSTM only     → "LSTM Only"
  - KMeans only   → "KMeans Only"
  - Neither       → "Honest"

Outputs:
  smurf_results_lstm.csv       — per-player LSTM verdict
  smurf_results_kmeans.csv     — per-player KMeans verdict
  smurf_results_combined.csv   — merged verdict for every player
  smurf_comparison_summary.txt — confusion-matrix-style breakdown
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from math import gcd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, BatchNormalization, MaxPooling1D,
                                     LSTM, Dropout, TimeDistributed, Dense)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve

import os
os.environ["OMP_NUM_THREADS"] = "9"

# ─────────────────────────────────────────────────────────────────────────────
# RUN MODE — change this single variable to control what gets executed
# ─────────────────────────────────────────────────────────────────────────────
#   "BOTH"   — run LSTM + KMeans fresh, then compare
#   "LSTM"   — run LSTM only (saves smurf_results_lstm.csv)
#   "KMEANS" — run KMeans only (saves smurf_results_kmeans.csv)
#   "COMPARE"— skip all training; load existing CSV results and compare
RUN_MODE = "BOTH"

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
X_PATH     = "X_data_no_padding.npy"
Y_PATH     = "y_data_no_padding.npy"
PUUID_PATH = "puuid_data_no_padding.npy"

MIN_GAME_MINUTES = 15   # drop remakes (KMeans pipeline filter)
IQR_C            = 1.5  # IQR multiplier for KMeans smurf flagging
LSTM_THRESHOLD   = 0.5  # probability threshold for LSTM smurf verdict

RANK_NAMES = {
    0: "Iron", 1: "Bronze", 2: "Silver",  3: "Gold",
    4: "Platinum", 5: "Emerald", 6: "Diamond",
    7: "Master", 8: "GrandMaster", 9: "Challenger",
}

FEATURE_LIST = [
    'championStats_abilityHaste',                #  0
    'championStats_abilityPower',                #  1
    'championStats_armor',                       #  2
    'championStats_armorPen',                    #  3
    'championStats_armorPenPercent',             #  4
    'championStats_attackDamage',                #  5
    'championStats_attackSpeed',                 #  6
    'championStats_bonusArmorPenPercent',        #  7
    'championStats_bonusMagicPenPercent',        #  8
    'championStats_ccReduction',                 #  9
    'championStats_cooldownReduction',           # 10
    'championStats_health',                      # 11
    'championStats_healthMax',                   # 12
    'championStats_healthRegen',                 # 13
    'championStats_lifesteal',                   # 14
    'championStats_magicPen',                    # 15
    'championStats_magicPenPercent',             # 16
    'championStats_magicResist',                 # 17
    'championStats_movementSpeed',               # 18
    'championStats_omnivamp',                    # 19
    'championStats_physicalVamp',                # 20
    'championStats_power',                       # 21
    'championStats_powerMax',                    # 22
    'championStats_powerRegen',                  # 23
    'championStats_spellVamp',                   # 24
    'currentGold',                               # 25
    'damageStats_magicDamageDone',               # 26
    'damageStats_magicDamageDoneToChampions',    # 27
    'damageStats_magicDamageTaken',              # 28
    'damageStats_physicalDamageDone',            # 29
    'damageStats_physicalDamageDoneToChampions', # 30
    'damageStats_physicalDamageTaken',           # 31
    'damageStats_totalDamageDone',               # 32
    'damageStats_totalDamageDoneToChampions',    # 33
    'damageStats_totalDamageTaken',              # 34
    'damageStats_trueDamageDone',                # 35
    'damageStats_trueDamageDoneToChampions',     # 36
    'damageStats_trueDamageTaken',               # 37
    'goldPerSecond',                             # 38
    'jungleMinionsKilled',                       # 39
    'level',                                     # 40
    'minionsKilled',                             # 41
    'position_x',                                # 42
    'position_y',                                # 43
    'timeEnemySpentControlled',                  # 44
    'totalGold',                                 # 45
    'xp',                                        # 46
]

COL = {name: idx for idx, name in enumerate(FEATURE_LIST)}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    """Load raw arrays shared by both pipelines."""
    x_data    = np.load(X_PATH,     allow_pickle=True)
    y_data    = np.load(Y_PATH,     allow_pickle=True)
    puuids    = np.load(PUUID_PATH, allow_pickle=True)
    print(f"[Data] Loaded {len(x_data)} games for {len(np.unique(puuids))} unique players")
    return x_data, y_data, puuids

# ─────────────────────────────────────────────────────────────────────────────
# DTW + FEATURE SCORING  (ported from algs.py)
# ─────────────────────────────────────────────────────────────────────────────
def dtw(x, y):
    n, m = len(x), len(y)
    dtw_matrix = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dtw_matrix[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw_matrix[i][j] = cost + min(
                dtw_matrix[i - 1][j],
                dtw_matrix[i][j - 1],
                dtw_matrix[i - 1][j - 1]
            )
    return dtw_matrix[n][m]


def calculate_feature_score(X_padded, y_data, dtw_fn, sparsity_threshold=0.30):
    N, T, F = X_padded.shape
    present_tiers = np.unique(y_data)
    num_tiers     = len(present_tiers)
    scores        = np.zeros((F, num_tiers), dtype=float)
    tier_idx      = [np.where(y_data == i)[0] for i in present_tiers]

    sparsity = (X_padded != 0.0).mean(axis=(0, 1))   # (F,)

    for f in range(F):
        if sparsity[f] < sparsity_threshold:
            continue

        # Step 1: mean per tier ignoring padding zeros
        mean_per_tier = np.zeros((num_tiers, T), dtype=float)
        for i, idx in enumerate(tier_idx):
            if len(idx) == 0:
                continue
            tier_data      = X_padded[idx, :, f]
            counts         = (tier_data != 0.0).sum(axis=0)
            sums           = tier_data.sum(axis=0)
            mean_per_tier[i] = np.where(counts > 0, sums / counts, 0.0)

        # Step 2: normalise and cumsum
        cumsum = np.zeros((num_tiers, T), dtype=float)
        for i in range(num_tiers):
            s = mean_per_tier[i].sum()
            if s != 0:
                cumsum[i] = np.cumsum(mean_per_tier[i] / s)

        # Step 3: mean DTW distance from each tier to all tiers
        for i in range(num_tiers):
            sum3 = 0.0
            for j in range(num_tiers):
                sum3 += dtw_fn(cumsum[i], cumsum[j])
            scores[f, i] = sum3 / num_tiers

    return scores   # (F, num_present_tiers)


def print_top_features(scores, label=""):
    feature_score = scores.mean(axis=1)
    ranked        = np.argsort(feature_score)[::-1]
    print(f"\n=== Top 10 Features [{label}] ===")
    for rank, idx in enumerate(ranked[:10], 1):
        print(f"  {rank:>2}. {FEATURE_LIST[idx]:<45}  score = {feature_score[idx]:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
#  METHOD 1 — LSTM PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def lcm(a, b):
    return a * b // gcd(a, b)

# takes a time series of any length and converts it into a fixed-length time series
# stretches the match and then samples it down to a fixed number of timesteps
def pool_method_1d(input_array, target_size):
    x = np.asarray(input_array, dtype=np.float32)
    N = x.shape[0]
    K = target_size
    LCM = lcm(N, K)
    L1  = LCM // N
    L2  = LCM // K
    tmp = np.zeros(LCM, dtype=np.float32)
    for i in range(N):
        tmp[i * L1:(i + 1) * L1] = x[i]
    out = np.zeros(K, dtype=np.float32)
    for i in range(K):
        chunk  = tmp[i * L2:(i + 1) * L2]
        out[i] = 0.0 if (i == 0 and x[0] == 0) else float(chunk.mean())
    return out
# Apply Algorithm 2 along TIME axis for (N, T, F)
def pool_method(X, target_size):
    X   = np.asarray(X, dtype=np.float32)
    N, T, F = X.shape
    out = np.zeros((N, target_size, F), dtype=np.float32)
    for n in range(N):
        for f in range(F):
            out[n, :, f] = pool_method_1d(X[n, :, f], target_size)
    return out

# USED FOR POOL SIZE
# determines how many timesteps after a Conv1D + Pooling layer
def output_timesteps(T, kernel_size=3, pool_size=2):
    return (T - (kernel_size - 1)) // pool_size

# puts a the y label on every time step
def make_timestep_labels(y_bin, T_out):
    return np.repeat(y_bin[:, None], T_out, axis=1)[..., None].astype(np.float32)

# kernel_size is how many time steps it looks at at once and pool size is how many it downsamples/combines
def build_play_pattern_model(F, kernel_size, pool_size):
    model = Sequential([
        Conv1D(512, kernel_size=kernel_size, activation='relu', padding='valid', input_shape=(None, F)),
        BatchNormalization(),
        MaxPooling1D(pool_size=pool_size),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(96, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(1, activation='sigmoid')),
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=0.0005),
        metrics=['accuracy']
    )
    return model

# calculates the AUC over time for plotting 
def auc_over_time(model, X, y_bin):
    y_pred = model.predict(X, verbose=0)[:, :, 0]
    T_out  = y_pred.shape[1]
    y_true = make_timestep_labels(y_bin, T_out)[:, :, 0]
    return np.array([
        roc_auc_score(y_true[:, :m].reshape(-1), y_pred[:, :m].reshape(-1))
        for m in range(1, T_out + 1)
    ])


def run_lstm_pipeline(x_data, y_data, puuids):
    """
    Trains three LSTM variants (raw / cut / pool), averages their smurf
    probabilities per player, and returns a DataFrame:
        puuid | lstm_prob | lstm_label ("Smurf" | "Honest")
    
    Smurf definition (LSTM): model predicts high-tier (≥1) probability
    exceeds LSTM_THRESHOLD, but the player's actual rank is low-tier (0).
    """
    print("\n" + "=" * 70)
    print("  METHOD 1 — LSTM PIPELINE")
    print("=" * 70)

    # Binary high/low tier label
    y_bin = (y_data > 4).astype(int)

    F = x_data[0].shape[1]
    T = max(len(seq) for seq in x_data)

    # ── Pad all sequences to uniform length ──────────────────────────────────
    X_padded = np.zeros((len(x_data), T, F), dtype=np.float32)
    for i, seq in enumerate(x_data):
        X_padded[i, :len(seq), :] = seq

    # ── Train / test split (shared indices so puuids stay aligned) ───────────
    idx_all = np.arange(len(x_data))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx_all, y_bin, test_size=0.3, random_state=42, stratify=y_bin)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp, test_size=1/3, random_state=42, stratify=y_tmp)
    
    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE SCORING — run once on the full padded set before any model trains
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[LSTM] Computing DTW feature scores (this may take a few minutes) ...")
    feat_scores = calculate_feature_score(X_padded, y_data, dtw)
    print_top_features(feat_scores, label="all models")

    # ── RAW MODEL ─────────────────────────────────────────────────────────────
    print("\n[LSTM] Training RAW model ...")
    T_out_raw = output_timesteps(T, kernel_size=3, pool_size=2)

    X_tr_raw  = X_padded[idx_tr]
    X_val_raw = X_padded[idx_val]
    X_te_raw  = X_padded[idx_test]

    raw_model = build_play_pattern_model(F, kernel_size=3, pool_size=2)
    raw_model.summary()
    raw_model.fit(
        X_tr_raw, make_timestep_labels(y_tr, T_out_raw),
        validation_data=(X_val_raw, make_timestep_labels(y_val, T_out_raw)),
        epochs=100, batch_size=32,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=1
    )

    raw_pred_test  = raw_model.predict(X_te_raw, verbose=0)           # (N_test, T_out, 1)
    y_test_seq_raw = make_timestep_labels(y_test, T_out_raw)
    auc_raw        = roc_auc_score(y_test_seq_raw.reshape(-1), raw_pred_test.reshape(-1))
    print(f"[LSTM] Test Raw AUC: {auc_raw:.4f}")

    # ── CUT MODEL (only games with full 25-min data) ──────────────────────────
    print("\n[LSTM] Training CUT model ...")
    gold_idx = FEATURE_LIST.index('totalGold')
    
    x_26 = X_padded[:, :26, :]

    # Count timesteps where totalGold is nonzero — more reliable than any feature
    real_steps = (x_26[:, :, gold_idx] != 0.0).sum(axis=1)  # (N,)
    cut_mask = real_steps == 26  # game must have real totalGold for all 26 minutes



    x_25_cut = x_26[cut_mask]
    y_25_cut = y_bin[cut_mask]
    pu_25_cut = puuids[cut_mask]

    N_cut, T_cut, F_cut = x_25_cut.shape
    print(f"Shape after filtering for 25-minute sequences: N={N_cut}, T={T_cut}, F={F_cut}")

    idx_cut_all = np.arange(len(x_25_cut))
    idx_ctr, idx_ctmp, y_ctr, y_ctmp = train_test_split(
        idx_cut_all, y_25_cut, test_size=0.3, random_state=42, stratify=y_25_cut)
    idx_cval, idx_ctest, y_cval, y_ctest = train_test_split(
        idx_ctmp, y_ctmp, test_size=1/3, random_state=42, stratify=y_ctmp)

    X_ctr  = x_25_cut[idx_ctr]
    X_cval = x_25_cut[idx_cval]
    X_cte  = x_25_cut[idx_ctest]

    T_out_cut = output_timesteps(T_cut, kernel_size=3, pool_size=2)
    cut_model = build_play_pattern_model(F, kernel_size=3, pool_size=2)
    cut_model.summary()
    cut_model.fit(
        X_ctr, make_timestep_labels(y_ctr, T_out_cut),
        validation_data=(X_cval, make_timestep_labels(y_cval, T_out_cut)),
        epochs=100, batch_size=32,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=1
    )

    cut_pred_test  = cut_model.predict(X_cte, verbose=0)
    y_test_seq_cut = make_timestep_labels(y_ctest, T_out_cut)
    auc_cut        = roc_auc_score(y_test_seq_cut.reshape(-1), cut_pred_test.reshape(-1))
    print(f"[LSTM] Test Cut AUC: {auc_cut:.4f}")

    # ── POOL MODEL ────────────────────────────────────────────────────────────
    print("\n[LSTM] Training POOL model ...")
    X_pooled = pool_method(X_padded, target_size=26)
    T_out_pool = output_timesteps(26, kernel_size=3, pool_size=2)

    X_ptr  = X_pooled[idx_tr]
    X_pval = X_pooled[idx_val]
    X_pte  = X_pooled[idx_test]

    pool_model = build_play_pattern_model(F, kernel_size=3, pool_size=2)
    pool_model.summary()
    pool_model.fit(
        X_ptr,  make_timestep_labels(y_tr,   T_out_pool),
        validation_data=(X_pval, make_timestep_labels(y_val, T_out_pool)),
        epochs=100, batch_size=32, callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)], verbose=1
    )

    pool_pred_test  = pool_model.predict(X_pte, verbose=0)
    y_test_seq_pool = make_timestep_labels(y_test, T_out_pool)
    auc_pool        = roc_auc_score(y_test_seq_pool.reshape(-1), pool_pred_test.reshape(-1))
    print(f"[LSTM] Test Pool AUC: {auc_pool:.4f}")
     
    # ── AUC OVER TIME CHART ───────────────────────────────────────────────────
    print("\n[LSTM] Generating AUC-over-time chart ...")
 
    def auc_curve_for(model, X_test, y_test, T_out):
        preds  = model.predict(X_test, verbose=0)[:, :, 0]
        y_true = make_timestep_labels(y_test, T_out)[:, :, 0]
        return np.array([
            roc_auc_score(y_true[:, :m].reshape(-1), preds[:, :m].reshape(-1))
            for m in range(1, T_out + 1)
        ])
 
    import matplotlib.pyplot as plt

    MAX_MIN = 26
    kernel_size, pool_size = 3, 2

    raw_auc  = auc_curve_for(raw_model,  X_te_raw, y_test,  T_out_raw)
    cut_auc  = auc_curve_for(cut_model,  X_cte,    y_ctest, T_out_cut)
    pool_auc = auc_curve_for(pool_model, X_pte,    y_test,  T_out_pool)

    def to_minutes(auc_curve, T_model):
        return np.minimum(
            (np.arange(1, len(auc_curve) + 1) * pool_size + (kernel_size - 1)),
            T_model
        )

    raw_min  = to_minutes(raw_auc,  T)
    cut_min  = to_minutes(cut_auc,  T_cut)
    pool_min = to_minutes(pool_auc, 26)

    plt.figure(figsize=(7, 5))
    for auc_curve, minutes, label, color in [
        (raw_auc,  raw_min,  "raw",  "tab:blue"),
        (cut_auc,  cut_min,  "cut",  "tab:orange"),
        (pool_auc, pool_min, "pool", "tab:green"),
    ]:
        mask = minutes <= MAX_MIN
        mins_masked = minutes[mask]
        auc_masked  = auc_curve[mask]

        plt.plot(mins_masked, auc_masked, label=label, color=color)
        plt.scatter(mins_masked[-1], auc_masked[-1], s=35, color=color, zorder=5)
        plt.axhline(auc_masked[-1], linestyle="--", linewidth=1, color=color, alpha=0.75)
        plt.text(mins_masked[0], auc_masked[-1], f"{auc_masked[-1]:.4f}",
                 ha="left", va="bottom", color=color, fontsize=9)

    plt.axvline(MAX_MIN, linestyle="--", linewidth=1, color="k", alpha=0.75)
    plt.title("AUC")
    plt.xlabel("Elapsed Time (minute)")
    plt.ylabel("Probability")
    plt.ylim(0.5, 1.0)
    plt.xlim(0, MAX_MIN)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("auc_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[LSTM] AUC chart saved -> auc_over_time.png")

    # ── INFERENCE — full dataset for smurf scoring ────────────────────────────
    print("\n[LSTM] Running inference on full dataset ...")

    # Raw: predict on all padded games
    raw_probs_all  = raw_model.predict(X_padded, verbose=0)[:, -1, 0]

    # Pool: predict on all pooled games
    pool_probs_all = pool_model.predict(pool_method(X_padded, target_size=26), verbose=0)[:, -1, 0]

    # Cut: predict only where cut_mask is True; fill rest with NaN
    cut_scores_cut = cut_model.predict(x_25_cut, verbose=0)[:, -1, 0]
    cut_probs_all  = np.full(len(x_data), np.nan)
    cut_probs_all[cut_mask] = cut_scores_cut


    # ─────────────────────────────────────────────────────────────────────────
    # SMURF DETECTION — optimal ROC thresholds (from algs.py smurf_detection)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[LSTM] Detecting smurfs via optimal ROC thresholds ...")

    # calculates the optimal roc threshold
    def get_threshold(labels, probs):
        fpr, tpr, thresholds = roc_curve(labels, probs)
        return thresholds[np.argmax(tpr - fpr)]

    # Work on test-set game rows so thresholds are found on held-out data
    df_test_raw  = pd.DataFrame({"prob": raw_probs_all[idx_test],  "label": y_test})
    df_test_pool = pd.DataFrame({"prob": pool_probs_all[idx_test], "label": y_test})

    # Cut test set maps to cut-subset indices
    cut_test_probs  = cut_model.predict(X_cte, verbose=0)[:, -1, 0]
    df_test_cut     = pd.DataFrame({"prob": cut_test_probs, "label": y_ctest})

    raw_thresh  = get_threshold(df_test_raw["label"],  df_test_raw["prob"])
    pool_thresh = get_threshold(df_test_pool["label"], df_test_pool["prob"])
    cut_thresh  = get_threshold(df_test_cut["label"],  df_test_cut["prob"])
    print(f"[LSTM] Thresholds — raw: {raw_thresh:.4f}  cut: {cut_thresh:.4f}  pool: {pool_thresh:.4f}")

    # Build per-game DataFrame for all games
    df_games = pd.DataFrame({
        "puuid":     puuids,
        "raw_prob":  raw_probs_all,
        "cut_prob":  cut_probs_all,    # NaN for games shorter than 26 min
        "pool_prob": pool_probs_all,
        "rank_id":   y_data,
    })

    # Per-game smurf predictions using per-model thresholds
    df_games["raw_pred"]  = (df_games["raw_prob"]  >= raw_thresh).astype(int)
    df_games["pool_pred"] = (df_games["pool_prob"] >= pool_thresh).astype(int)
    df_games["cut_pred"]  = np.where(
        df_games["cut_prob"].notna(),
        (df_games["cut_prob"] >= cut_thresh).astype(int),
        np.nan
    )

    # Smurf = predicted high-tier but actually low-tier
    label_bin = (df_games["rank_id"] > 4).astype(int)
    df_games["raw_smurf"]  = ((df_games["raw_pred"]  == 1) & (label_bin == 0)).astype(int)
    df_games["pool_smurf"] = ((df_games["pool_pred"] == 1) & (label_bin == 0)).astype(int)
    df_games["cut_smurf"]  = np.where(
        df_games["cut_pred"].notna(),
        ((df_games["cut_pred"] == 1) & (label_bin == 0)).astype(int),
        np.nan
    )

    print(f"[LSTM] Smurfs found (game level) — "
          f"raw: {df_games['raw_smurf'].sum():.0f}  "
          f"cut: {df_games['cut_smurf'].sum():.0f}  "
          f"pool: {df_games['pool_smurf'].sum():.0f}")

    # ── Aggregate to player level ─────────────────────────────────────────────
    def safe_mean(s):
        return s.dropna().mean() if s.notna().any() else np.nan

    df_player = df_games.groupby("puuid").agg(
        raw_prob  =("raw_prob",  "mean"),
        cut_prob  =("cut_prob",  safe_mean),
        pool_prob =("pool_prob", "mean"),
        rank_id   =("rank_id",   "median"),
    ).reset_index()

    # Ensemble probability: mean of available model probs per player
    df_player["lstm_prob"] = df_player[["raw_prob", "cut_prob", "pool_prob"]].mean(axis=1)

    # Player-level smurf flag: ensemble prob >= mean of the three thresholds, actually low-tier
    ensemble_thresh = np.mean([raw_thresh, cut_thresh, pool_thresh])
    df_player["lstm_label"] = np.where(
        (df_player["lstm_prob"] >= ensemble_thresh) & (df_player["rank_id"] <= 4),
        "Smurf", "Honest"
    )

    n_smurfs = (df_player["lstm_label"] == "Smurf").sum()
    print(f"[LSTM] {n_smurfs} smurfs detected out of {len(df_player)} players "
          f"({100 * n_smurfs / len(df_player):.1f}%)  "
          f"[ensemble threshold: {ensemble_thresh:.4f}]")

    result = df_player[["puuid", "lstm_prob", "lstm_label"]].copy()
    result.to_csv("smurf_results_lstm.csv", index=False)
    print("[LSTM] Results saved → smurf_results_lstm.csv")
    return result, df_games, {"raw": raw_thresh, "cut": cut_thresh, "pool": pool_thresh}


# ═════════════════════════════════════════════════════════════════════════════
#  METHOD 2 — KMEANS + PCA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

KMEANS_FEATURE_NAMES = ['GPM', 'XPM', 'LHPM', 'HDPM', 'CCPM', 'DTPM']

def gap_statistics(X, k_max=10, n_refs=10, random_state=42):
    rng  = np.random.default_rng(random_state)
    gaps = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        Wk = km.inertia_
        ref_inertias = []
        for _ in range(n_refs):
            ref    = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            km_ref = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            km_ref.fit(ref)
            ref_inertias.append(np.log(km_ref.inertia_))
        gaps.append(np.mean(ref_inertias) - np.log(Wk))
    optimal_k = int(np.argmax(gaps)) + 1
    print(f"[Gap Stats] Optimal k = {optimal_k}  "
          f"(gaps: {[f'{g:.3f}' for g in gaps]})")
    return optimal_k


def run_kmeans_pipeline(x_data, y_data, puuids):
    """
    Runs the KMeans+PCA pipeline and returns a DataFrame:
        puuid | kmeans_dist | kmeans_label ("Smurf" | "Honest")
    """
    print("\n" + "=" * 70)
    print("  METHOD 2 — KMEANS + PCA PIPELINE")
    print("=" * 70)

    # ── Filter short games ────────────────────────────────────────────────────
    lengths   = np.array([len(seq) for seq in x_data])
    keep_mask = lengths >= MIN_GAME_MINUTES
    n_dropped = int((~keep_mask).sum())
    x_filt    = x_data[keep_mask]
    y_filt    = y_data[keep_mask]
    pu_filt   = puuids[keep_mask]
    print(f"[KMeans] Dropped {n_dropped} games < {MIN_GAME_MINUTES} min → "
          f"{len(x_filt)} remain")

    # ── Feature engineering ───────────────────────────────────────────────────
    records = []
    for seq in x_filt:
        T     = len(seq)
        final = seq[-1]
        gpm   = final[COL['goldPerSecond']] * 60
        xpm   = final[COL['xp']] / T
        lhpm  = (final[COL['minionsKilled']] +
                 final[COL['jungleMinionsKilled']]) / T
        hdpm  = final[COL['damageStats_totalDamageDoneToChampions']] / T
        ccpm  = final[COL['timeEnemySpentControlled']] / T
        dtpm  = final[COL['damageStats_totalDamageTaken']] / T
        records.append([gpm, xpm, lhpm, hdpm, ccpm, dtpm])

    X_feat = np.nan_to_num(np.array(records, dtype=np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)

    # ── Aggregate per player ──────────────────────────────────────────────────
    df_temp = pd.DataFrame(X_feat, columns=KMEANS_FEATURE_NAMES)
    df_temp['puuid']   = pu_filt
    df_temp['rank_id'] = y_filt

    player_profiles = df_temp.groupby('puuid').agg(
        **{feat: (feat, 'mean') for feat in KMEANS_FEATURE_NAMES},
        rank_id=('rank_id', 'median')
    ).reset_index()

    print(f"[KMeans] {len(player_profiles)} player fingerprints built "
          f"(avg {len(df_temp) / len(player_profiles):.1f} games/player)")

    X_player = player_profiles[KMEANS_FEATURE_NAMES].values

    # ── Standardise + PCA ─────────────────────────────────────────────────────
    X_scaled = StandardScaler().fit_transform(X_player)
    pca      = PCA(n_components=0.90, random_state=42)
    X_pca    = pca.fit_transform(X_scaled)
    print(f"[PCA] {pca.n_components_} components explain 90% variance")

    # ── Gap statistics + KMeans ───────────────────────────────────────────────
    optimal_k      = gap_statistics(X_pca)
    km             = KMeans(n_clusters=optimal_k, n_init=30, random_state=42)
    cluster_labels = km.fit_predict(X_pca)
    player_profiles['cluster'] = cluster_labels

    # ── Label clusters (low / avg / high) ─────────────────────────────────────
    # Use mean of the standardised feature values so non-feature columns
    # (puuid, rank_id, cluster) don't corrupt the rank.
    cluster_means   = {c: X_scaled[cluster_labels == c].mean()
                       for c in range(optimal_k)}
    sorted_clusters = sorted(cluster_means, key=cluster_means.get)
    high_cluster    = sorted_clusters[-1]
    low_cluster     = sorted_clusters[0]

    label_map = {high_cluster: 'high-performing', low_cluster: 'low-performing'}
    for c in range(optimal_k):
        if c not in label_map:
            label_map[c] = 'average-performing'
    player_profiles['profile'] = player_profiles['cluster'].map(label_map)

    # Print per-cluster stats so you can sanity-check the labelling
    for c in sorted_clusters:
        mask_c = cluster_labels == c
        lbl    = label_map[c]
        means  = X_player[mask_c].mean(axis=0)
        stats  = "  ".join(f"{n}={v:.2f}" for n, v in zip(KMEANS_FEATURE_NAMES, means))
        print(f"[KMeans] cluster {c} → {lbl:20s}  n={mask_c.sum():5d}  {stats}")
    print(f"[KMeans] high-performing = cluster {high_cluster}")

    # ── IQR smurf detection within high-performing cluster ────────────────────
    # Paper method (Algorithm 3 + Section IV-B):
    #   Apply IQR_WH — IQR on the WHOLE high-performing cluster using the
    #   original (pre-PCA, pre-scale) feature values.  A player is a smurf/
    #   booster if they exceed  Q3 + c*IQR  on ANY of the performance features.
    #   This matches the paper's Table VI (high values in ALL features for the
    #   smurf profile) and recovers ~1.6% smurf rate rather than near-zero.
    high_mask        = cluster_labels == high_cluster
    high_feat_vals   = X_player[high_mask]          # shape (n_high, n_features)

    smurf_in_high    = np.zeros(high_mask.sum(), dtype=bool)
    for fi in range(high_feat_vals.shape[1]):
        col_vals  = high_feat_vals[:, fi]
        Q1        = np.percentile(col_vals, 25)
        Q3        = np.percentile(col_vals, 75)
        upper     = Q3 + IQR_C * (Q3 - Q1)
        smurf_in_high |= (col_vals > upper)

    high_indices               = np.where(high_mask)[0]
    smurf_mask                 = np.zeros(len(player_profiles), dtype=bool)
    smurf_mask[high_indices[smurf_in_high]] = True

    player_profiles['kmeans_label'] = np.where(smurf_mask, 'Smurf', 'Honest')

    # Also store centroid distance for diagnostics (optional, not used for detection)
    centroid     = km.cluster_centers_[high_cluster]
    high_pca_pts = X_pca[high_mask]
    distances    = np.linalg.norm(high_pca_pts - centroid, axis=1)
    all_distances            = np.full(len(player_profiles), np.nan)
    all_distances[high_mask] = distances
    player_profiles['kmeans_dist'] = all_distances

    n_smurfs = smurf_mask.sum()
    print(f"[KMeans] {n_smurfs} smurfs detected out of {len(player_profiles)} players "
          f"({100 * n_smurfs / len(player_profiles):.1f}%)")

    result = player_profiles[["puuid", "kmeans_dist", "kmeans_label"]].copy()
    result.to_csv("smurf_results_kmeans.csv", index=False)
    print("[KMeans] Results saved → smurf_results_kmeans.csv")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  COMPARISON & REPORTING
# ═════════════════════════════════════════════════════════════════════════════

def compare_and_report(lstm_df, lstm_games_df, lstm_thresholds, kmeans_df):
    """
    Produces a confusion-matrix-style cross-model summary for:
      - RAW model vs KMeans
      - CUT model vs KMeans
      - POOL model vs KMeans
      - LSTM Ensemble vs KMeans
    """
    from sklearn.metrics import roc_curve

    print("\n" + "=" * 70)
    print("  CROSS-MODEL COMPARISON")
    print("=" * 70)

    # ── Build per-player verdicts for each LSTM variant ───────────────────────
    def safe_mean(s):
        return s.dropna().mean() if s.notna().any() else np.nan

    def player_label_from_prob(df_games, prob_col, threshold, pred_col):
        """Aggregate game-level probs to player level, apply threshold, flag smurfs."""
        df_p = df_games.groupby("puuid").agg(
            prob    =(prob_col, safe_mean),
            rank_id =("rank_id", "median"),
        ).reset_index()
        df_p[pred_col] = (df_p["prob"] >= threshold).astype(int)
        label_bin      = (df_p["rank_id"] > 4).astype(int)
        df_p["label"]  = np.where(
            (df_p[pred_col] == 1) & (label_bin == 0), "Smurf", "Honest"
        )
        return df_p[["puuid", "label"]].rename(columns={"label": pred_col + "_label"})

    raw_player  = player_label_from_prob(lstm_games_df, "raw_prob",  lstm_thresholds["raw"],  "raw")
    pool_player = player_label_from_prob(lstm_games_df, "pool_prob", lstm_thresholds["pool"], "pool")

    # Cut: only players who have at least one cut-eligible game get a cut score
    cut_games   = lstm_games_df[lstm_games_df["cut_prob"].notna()]
    if len(cut_games) > 0:
        cut_player = player_label_from_prob(cut_games, "cut_prob", lstm_thresholds["cut"], "cut")
    else:
        cut_player = pd.DataFrame(columns=["puuid", "cut_label"])

    # ── Helper: single confusion matrix block ─────────────────────────────────
    def print_confusion(lstm_label_col, model_name, merged):
        l = merged[lstm_label_col].fillna("Honest") == "Smurf"
        k = merged["kmeans_label"].fillna("Honest") == "Smurf"

        confirmed   = int(( l &  k).sum())
        lstm_only   = int(( l & ~k).sum())
        kmeans_only = int((~l &  k).sum())
        honest      = int((~l & ~k).sum())
        total       = len(merged)

        sep  = "─" * 54
        lines = [
            "",
            f"╔══════════════════════════════════════════════════════╗",
            f"║   {model_name:<51}║",
            f"╚══════════════════════════════════════════════════════╝",
            "",
            f"{'Category':<28} {'Count':>8}  {'% of Players':>12}",
            sep,
            f"{'Confirmed Smurf (both)':<28} {confirmed:>8}  {100*confirmed/total:>11.1f}%",
            f"{'Flagged by LSTM only':<28} {lstm_only:>8}  {100*lstm_only/total:>11.1f}%",
            f"{'Flagged by KMeans only':<28} {kmeans_only:>8}  {100*kmeans_only/total:>11.1f}%",
            f"{'Honest (neither)':<28} {honest:>8}  {100*honest/total:>11.1f}%",
            sep,
            f"{'Total players':<28} {total:>8}",
            "",
            f"  Confusion-Matrix View ({model_name} rows × KMeans columns):",
            "",
            f"  {'':16}  {'KMeans: Smurf':>14}  {'KMeans: Honest':>15}",
            f"  {sep}",
            f"  {'LSTM: Smurf':<16}  {confirmed:>14}  {lstm_only:>15}",
            f"  {'LSTM: Honest':<16}  {kmeans_only:>14}  {honest:>15}",
            "",
        ]
        text = "\n".join(lines)
        print(text)
        return text

    # ── Merge everything onto kmeans base ─────────────────────────────────────
    base = kmeans_df[["puuid", "kmeans_label"]].copy()

    merged_raw  = base.merge(raw_player,  on="puuid", how="outer")
    merged_cut  = base.merge(cut_player,  on="puuid", how="outer")
    merged_pool = base.merge(pool_player, on="puuid", how="outer")
    merged_ens  = base.merge(lstm_df[["puuid", "lstm_label"]], on="puuid", how="outer")

    all_text  = print_confusion("raw_label",   "RAW Model vs KMeans",      merged_raw)
    all_text += print_confusion("cut_label",   "CUT Model vs KMeans",      merged_cut)
    all_text += print_confusion("pool_label",  "POOL Model vs KMeans",     merged_pool)
    all_text += print_confusion("lstm_label",  "Ensemble vs KMeans",       merged_ens)

    # ── Save outputs ──────────────────────────────────────────────────────────
    merged_ens.to_csv("smurf_results_combined.csv", index=False)
    print("[Compare] Full merged results saved → smurf_results_combined.csv")

    with open("smurf_comparison_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(all_text)
    print("[Compare] Summary saved → smurf_comparison_summary.txt")

    return merged_ens


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── GPU check ────────────────────────────────────────────────────────────
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))
    print(tf.sysconfig.get_build_info())
    print(tf.sysconfig.get_build_info()['is_cuda_build'])

    print(f"\n{'='*70}")
    print(f"  RUN MODE: {RUN_MODE}")
    print(f"{'='*70}\n")

    # ── COMPARE mode — load saved CSVs, skip all training ────────────────────
    if RUN_MODE == "COMPARE":
        print("[Mode] Loading existing results from CSV files ...")
        lstm_results   = pd.read_csv("smurf_results_lstm.csv")
        kmeans_results = pd.read_csv("smurf_results_kmeans.csv")

        # Reconstruct a minimal lstm_games_df and thresholds so compare_and_report
        # can still run.  Since we have no raw game-level data here, we build a
        # stub that maps every player to their saved ensemble label/prob.
        lstm_games = lstm_results.rename(columns={"lstm_prob": "raw_prob"})
        lstm_games["cut_prob"]  = lstm_games["raw_prob"]
        lstm_games["pool_prob"] = lstm_games["raw_prob"]
        lstm_games["rank_id"]   = 0   # unknown without re-loading — only affects confusion counts
        lstm_thresholds         = {"raw": 0.5, "cut": 0.5, "pool": 0.5}
        compare_and_report(lstm_results, lstm_games, lstm_thresholds, kmeans_results)

    # ── KMEANS only ───────────────────────────────────────────────────────────
    elif RUN_MODE == "KMEANS":
        x_data, y_data, puuids = load_data()
        kmeans_results = run_kmeans_pipeline(x_data, y_data, puuids)

    # ── LSTM only ─────────────────────────────────────────────────────────────
    elif RUN_MODE == "LSTM":
        x_data, y_data, puuids = load_data()
        lstm_results, lstm_games, lstm_thresholds = run_lstm_pipeline(x_data, y_data, puuids)

    # ── BOTH — run everything fresh then compare ──────────────────────────────
    elif RUN_MODE == "BOTH":
        x_data, y_data, puuids = load_data()
        lstm_results, lstm_games, lstm_thresholds = run_lstm_pipeline(x_data, y_data, puuids)
        kmeans_results = run_kmeans_pipeline(x_data, y_data, puuids)
        compare_and_report(lstm_results, lstm_games, lstm_thresholds, kmeans_results)

    else:
        raise ValueError(
            f"Unknown RUN_MODE '{RUN_MODE}'. "
            "Choose one of: 'BOTH', 'LSTM', 'KMEANS', 'COMPARE'"
        )
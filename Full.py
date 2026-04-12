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


# ═════════════════════════════════════════════════════════════════════════════
#  METHOD 1 — LSTM PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def lcm(a, b):
    return a * b // gcd(a, b)

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

def pool_method(X, target_size):
    X   = np.asarray(X, dtype=np.float32)
    N, T, F = X.shape
    out = np.zeros((N, target_size, F), dtype=np.float32)
    for n in range(N):
        for f in range(F):
            out[n, :, f] = pool_method_1d(X[n, :, f], target_size)
    return out

def output_timesteps(T, kernel_size=3, pool_size=2):
    return (T - (kernel_size - 1)) // pool_size

def make_timestep_labels(y_bin, T_out):
    return np.repeat(y_bin[:, None], T_out, axis=1)[..., None].astype(np.float32)

def build_play_pattern_model(F, kernel_size, pool_size):
    model = Sequential([
        Conv1D(512, kernel_size=kernel_size, activation='relu',
               padding='valid', input_shape=(None, F)),
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
    y_bin = (y_data > 3).astype(int)

    F = x_data[0].shape[1]
    T = max(len(seq) for seq in x_data)

    # ── Pad all sequences to uniform length ──────────────────────────────────
    X_padded = np.zeros((len(x_data), T, F), dtype=np.float32)
    for i, seq in enumerate(x_data):
        X_padded[i, :len(seq), :] = seq

    early_stop = EarlyStopping(monitor="val_loss", patience=5,
                                restore_best_weights=True)

    # ── Train / test split (shared indices so puuids stay aligned) ───────────
    idx_all = np.arange(len(x_data))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx_all, y_bin, test_size=0.3, random_state=42, stratify=y_bin)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp, test_size=1/3, random_state=42, stratify=y_tmp)

    # ── RAW MODEL ─────────────────────────────────────────────────────────────
    print("\n[LSTM] Training RAW model ...")
    T_out_raw = output_timesteps(T, kernel_size=3, pool_size=2)

    X_tr_raw  = X_padded[idx_tr]
    X_val_raw = X_padded[idx_val]
    X_te_raw  = X_padded[idx_test]

    raw_model = build_play_pattern_model(F, kernel_size=3, pool_size=2)
    raw_model.summary()
    raw_model.fit(
        X_tr_raw,  make_timestep_labels(y_tr,  T_out_raw),
        validation_data=(X_val_raw, make_timestep_labels(y_val, T_out_raw)),
        epochs=100, batch_size=32, callbacks=[early_stop], verbose=1
    )

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

    N, T, F = x_26.shape
    print(f"Shape after filtering for 25-minute sequences: N={N}, T={T}, F={F}")

    idx_cut_all = np.arange(len(x_25_cut))
    idx_ctr, idx_ctmp, y_ctr, y_ctmp = train_test_split(
        idx_cut_all, y_25_cut, test_size=0.3, random_state=42, stratify=y_25_cut)
    idx_cval, idx_ctest, y_cval, y_ctest = train_test_split(
        idx_ctmp, y_ctmp, test_size=1/3, random_state=42, stratify=y_ctmp)

    mean_c = x_25_cut[idx_ctr].mean(axis=(0, 1), keepdims=True)
    std_c  = x_25_cut[idx_ctr].std(axis=(0, 1),  keepdims=True) + 1e-8
    X_ctr  = (x_25_cut[idx_ctr]   - mean_c) / std_c
    X_cval = (x_25_cut[idx_cval]  - mean_c) / std_c
    X_cte  = (x_25_cut[idx_ctest] - mean_c) / std_c

    T_out_cut = output_timesteps(25, kernel_size=3, pool_size=2)
    cut_model = build_play_pattern_model(F, kernel_size=3, pool_size=2)
    cut_model.summary()
    cut_model.fit(
        X_ctr,  make_timestep_labels(y_ctr,  T_out_cut),
        validation_data=(X_cval, make_timestep_labels(y_cval, T_out_cut)),
        epochs=100, batch_size=32, callbacks=[early_stop], verbose=1
    )

    # ── POOL MODEL ────────────────────────────────────────────────────────────
    print("\n[LSTM] Training POOL model ...")
    X_pooled = pool_method(X_padded, target_size=26)
    T_out_pool = output_timesteps(26, kernel_size=3, pool_size=2)

    X_ptr  = X_pooled[idx_tr]
    X_pval = X_pooled[idx_val]
    X_pte  = X_pooled[idx_test]

    mean_p = X_ptr.mean(axis=(0, 1), keepdims=True)
    std_p  = X_ptr.std(axis=(0, 1),  keepdims=True) + 1e-8
    X_ptr  = (X_ptr  - mean_p) / std_p
    X_pval = (X_pval - mean_p) / std_p
    X_pte  = (X_pte  - mean_p) / std_p

    pool_model = build_play_pattern_model(F, kernel_size=3, pool_size=2)
    pool_model.summary()
    pool_model.fit(
        X_ptr,  make_timestep_labels(y_tr,   T_out_pool),
        validation_data=(X_pval, make_timestep_labels(y_val, T_out_pool)),
        epochs=100, batch_size=32, callbacks=[early_stop], verbose=1
    )
     
    # ── AUC OVER TIME CHART ───────────────────────────────────────────────────
    print("\n[LSTM] Generating AUC-over-time chart ...")
 
    def auc_curve_for(model, X_test, y_test, T_out):
        preds  = model.predict(X_test, verbose=0)[:, :, 0]
        y_true = make_timestep_labels(y_test, T_out)[:, :, 0]
        return np.array([
            roc_auc_score(y_true[:, :m].reshape(-1), preds[:, :m].reshape(-1))
            for m in range(1, T_out + 1)
        ])
 
    kernel_size, pool_size = 3, 2
 
    raw_auc  = auc_curve_for(raw_model,  X_padded[idx_test],  y_test,  T_out_raw)
    cut_auc  = auc_curve_for(cut_model,  X_cte,               y_ctest, T_out_cut)
    pool_auc = auc_curve_for(pool_model, X_pte,               y_test,  T_out_pool)
 
    def to_minutes(auc_curve, T_model):
        return np.minimum(
            (np.arange(1, len(auc_curve) + 1) * pool_size + (kernel_size - 1)),
            T_model
        )
 
    raw_min  = to_minutes(raw_auc,  T)
    cut_min  = to_minutes(cut_auc,  25)
    pool_min = to_minutes(pool_auc, 26)
 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    for auc_curve, minutes, label, color in [
        (raw_auc,  raw_min,  "raw",  "tab:blue"),
        (cut_auc,  cut_min,  "cut",  "tab:orange"),
        (pool_auc, pool_min, "pool", "tab:green"),
    ]:
        plt.plot(minutes, auc_curve, label=label, color=color)
        plt.axhline(auc_curve[-1], linestyle="--", linewidth=1, color=color, alpha=0.5)
        plt.text(minutes[0], auc_curve[-1] + 0.003,
                 f"{auc_curve[-1]:.4f}", ha="left", va="bottom",
                 color=color, fontsize=9)
 
    plt.title("AUC")
    plt.xlabel("Elapsed Time (minute)")
    plt.ylabel("Probability")
    plt.ylim(0.5, 1.0)
    plt.xlim(1, max(raw_min[-1], cut_min[-1], pool_min[-1]))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("auc_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[LSTM] AUC chart saved -> auc_over_time.png")

    # ── INFERENCE — full dataset for smurf scoring ────────────────────────────
    print("\n[LSTM] Running inference on full dataset ...")

    raw_probs  = raw_model.predict(X_padded, verbose=0)[:, -1, 0]
    pool_probs = pool_model.predict(
        pool_method(X_padded, target_size=26), verbose=0)[:, -1, 0]

    # Cut model: score only the games in the cut subset; fill others with NaN
    cut_full_probs = np.full(len(x_data), np.nan)
    cut_scores     = cut_model.predict(
        (X_padded[cut_mask, :25, :] - mean_c) / std_c, verbose=0)[:, -1, 0]
    cut_full_probs[cut_mask] = cut_scores

    # Build per-game DataFrame
    df_games = pd.DataFrame({
        "puuid":     puuids,
        "raw_prob":  raw_probs,
        "cut_prob":  cut_full_probs,   # NaN for games shorter than 25 min
        "pool_prob": pool_probs,
        "rank_id":   y_data,
    })

    # Aggregate per player — average available probabilities
    def safe_mean(s):
        return s.dropna().mean() if s.notna().any() else np.nan

    df_player = df_games.groupby("puuid").agg(
        raw_prob  =("raw_prob",  "mean"),
        cut_prob  =("cut_prob",  safe_mean),
        pool_prob =("pool_prob", "mean"),
        rank_id   =("rank_id",   "median"),
    ).reset_index()

    # Ensemble: mean of available probabilities per player
    prob_cols = ["raw_prob", "cut_prob", "pool_prob"]
    df_player["lstm_prob"] = df_player[prob_cols].mean(axis=1)

    # Smurf = predicted high-tier but actually low-tier
    df_player["lstm_label"] = np.where(
        (df_player["lstm_prob"] >= LSTM_THRESHOLD) & (df_player["rank_id"] <= 3),
        "Smurf", "Honest"
    )

    n_smurfs = (df_player["lstm_label"] == "Smurf").sum()
    print(f"[LSTM] {n_smurfs} smurfs detected out of {len(df_player)} players "
          f"({100 * n_smurfs / len(df_player):.1f}%)")

    result = df_player[["puuid", "lstm_prob", "lstm_label"]].copy()
    result.to_csv("smurf_results_lstm.csv", index=False)
    print("[LSTM] Results saved → smurf_results_lstm.csv")
    return result


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
    cluster_means   = {c: X_player[cluster_labels == c].mean()
                       for c in range(optimal_k)}
    sorted_clusters = sorted(cluster_means, key=cluster_means.get)
    high_cluster    = sorted_clusters[-1]
    low_cluster     = sorted_clusters[0]

    label_map = {high_cluster: 'high-performing', low_cluster: 'low-performing'}
    for c in range(optimal_k):
        if c not in label_map:
            label_map[c] = 'average-performing'
    player_profiles['profile'] = player_profiles['cluster'].map(label_map)
    print(f"[KMeans] high-performing = cluster {high_cluster}")

    # ── IQR smurf detection within high-performing cluster ────────────────────
    high_mask       = cluster_labels == high_cluster
    high_pca_pts    = X_pca[high_mask]
    centroid        = km.cluster_centers_[high_cluster]
    distances       = np.linalg.norm(high_pca_pts - centroid, axis=1)

    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    upper_bound     = Q3 + IQR_C * (Q3 - Q1)

    is_smurf_in_high           = distances > upper_bound
    high_indices               = np.where(high_mask)[0]
    smurf_mask                 = np.zeros(len(player_profiles), dtype=bool)
    smurf_mask[high_indices[is_smurf_in_high]] = True

    player_profiles['kmeans_label'] = np.where(smurf_mask, 'Smurf', 'Honest')

    # Store centroid distance for all players (NaN for non-high clusters)
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

def compare_and_report(lstm_df, kmeans_df):
    """
    Merge the two verdict DataFrames on puuid and produce a confusion-matrix-
    style summary showing overlap between the two methods.
    """
    print("\n" + "=" * 70)
    print("  CROSS-MODEL COMPARISON")
    print("=" * 70)

    merged = pd.merge(lstm_df, kmeans_df, on="puuid", how="outer")

    # Fill gaps — players only present in one pipeline get "Honest" in the other
    merged["lstm_label"]   = merged["lstm_label"].fillna("Honest")
    merged["kmeans_label"] = merged["kmeans_label"].fillna("Honest")

    # Derived verdict
    def combined_verdict(row):
        l = row["lstm_label"]   == "Smurf"
        k = row["kmeans_label"] == "Smurf"
        if l and k:
            return "Confirmed Smurf"
        elif l:
            return "LSTM Only"
        elif k:
            return "KMeans Only"
        else:
            return "Honest"

    merged["verdict"] = merged.apply(combined_verdict, axis=1)

    # ── Counts ────────────────────────────────────────────────────────────────
    counts = merged["verdict"].value_counts()
    total  = len(merged)

    confirmed  = counts.get("Confirmed Smurf", 0)
    lstm_only  = counts.get("LSTM Only",       0)
    kmeans_only= counts.get("KMeans Only",     0)
    honest     = counts.get("Honest",           0)

    # ── Print summary ─────────────────────────────────────────────────────────
    sep  = "─" * 54
    line = f"{'Category':<28} {'Count':>8}  {'% of Players':>12}"

    summary_lines = [
        "",
        "╔══════════════════════════════════════════════════════╗",
        "║          SMURF DETECTION — CROSS-MODEL SUMMARY       ║",
        "╚══════════════════════════════════════════════════════╝",
        "",
        line,
        sep,
        f"{'Confirmed Smurf (both)':<28} {confirmed:>8}  {100*confirmed/total:>11.1f}%",
        f"{'Flagged by LSTM only':<28} {lstm_only:>8}  {100*lstm_only/total:>11.1f}%",
        f"{'Flagged by KMeans only':<28} {kmeans_only:>8}  {100*kmeans_only/total:>11.1f}%",
        f"{'Honest (neither)':<28} {honest:>8}  {100*honest/total:>11.1f}%",
        sep,
        f"{'Total players':<28} {total:>8}",
        "",
        "  Confusion-Matrix View (LSTM rows × KMeans columns):",
        "",
    ]

    # 2×2 confusion matrix layout
    cm_header = f"  {'':16}  {'KMeans: Smurf':>14}  {'KMeans: Honest':>15}"
    cm_row1   = (f"  {'LSTM: Smurf':<16}  {confirmed:>14}  {lstm_only:>15}")
    cm_row2   = (f"  {'LSTM: Honest':<16}  {kmeans_only:>14}  {honest:>15}")
    summary_lines += [cm_header, "  " + sep, cm_row1, cm_row2, ""]

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # ── Save outputs ──────────────────────────────────────────────────────────
    merged.to_csv("smurf_results_combined.csv", index=False)
    print("[Compare] Full merged results saved → smurf_results_combined.csv")

    with open("smurf_comparison_summary.txt", "w") as fh:
        fh.write(summary_text)
    print("[Compare] Summary saved → smurf_comparison_summary.txt")

    return merged


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Load shared data
    x_data, y_data, puuids = load_data()

    # 2. Run LSTM pipeline
    lstm_results   = run_lstm_pipeline(x_data, y_data, puuids)

    # 3. Run KMeans pipeline
    kmeans_results = run_kmeans_pipeline(x_data, y_data, puuids)

    # 4. Compare and report
    final_df = compare_and_report(lstm_results, kmeans_results)
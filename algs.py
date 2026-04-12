import pandas as pd
import numpy as np
from math import gcd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D
from tensorflow.keras.layers import LSTM, Dropout, TimeDistributed, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_DIR = "data"
OUTPUT_FILE = "justice_league_dataset_final.csv"

# performs dynamic time warping, 
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

SPARSITY_THRESHOLD = 0.30  # feature must be nonzero at least 30% of timesteps

def calculate_feature_score_TEST(x_data, y_data, dtw_fn):
    N, T, F = x_data.shape
    num_tiers = 9
    scores = np.zeros((F, num_tiers), dtype=float)
    tier_idx = [np.where(y_data == i)[0] for i in range(num_tiers)]

    # Compute sparsity mask once upfront
    sparsity = (x_data != 0.0).mean(axis=(0, 1))  # shape: (F,)
    sparse_features = np.where(sparsity < SPARSITY_THRESHOLD)[0]
    dense_features = np.where(sparsity >= SPARSITY_THRESHOLD)[0]
    
    print(f"Skipping {len(sparse_features)} sparse features (<{SPARSITY_THRESHOLD*100:.0f}% nonzero):")
    for f in sparse_features:
        print(f"  {FEATURE_LIST[f]:<45} {sparsity[f]*100:.1f}%")
    print(f"Scoring {len(dense_features)} dense features...")

    for f in dense_features:
        # Step 1: mean per tier ignoring padding zeros
        mean_per_tier = np.zeros((num_tiers, T), dtype=float)
        for i in range(num_tiers):
            idx = tier_idx[i]
            if len(idx) == 0:
                continue
            tier_data = x_data[idx, :, f]
            counts = (tier_data != 0.0).sum(axis=0)
            sums = tier_data.sum(axis=0)
            mean_per_tier[i] = np.where(counts > 0, sums / counts, 0.0)

        # Step 2: normalize and cumsum
        cumsum = np.zeros((num_tiers, T), dtype=float)
        for i in range(num_tiers):
            s = mean_per_tier[i].sum()
            if s != 0:
                cumsum[i] = np.cumsum(mean_per_tier[i] / s)

        # Step 3: mean DTW distance
        for i in range(num_tiers):
            sum3 = 0.0
            for j in range(num_tiers):
                sum3 += dtw_fn(cumsum[i], cumsum[j])
            scores[f, i] = sum3 / num_tiers

    return scores

def calculate_feature_score(X_padded, y_data, dtw_fn, sparsity_threshold=0.30):
    N, T, F = X_padded.shape
    present_tiers = np.unique(y_data)
    num_tiers = len(present_tiers)

    scores = np.zeros((F, num_tiers), dtype=float)
    tier_idx = [np.where(y_data == i)[0] for i in present_tiers]

    # Compute sparsity per feature and skip sparse ones
    sparsity = (X_padded != 0.0).mean(axis=(0, 1))  # shape: (F,)

    for f in range(F):
        if sparsity[f] < sparsity_threshold:
            continue  # leave scores[f] as zeros

        # Step 1: mean per tier ignoring padding zeros
        mean_per_tier = np.zeros((num_tiers, T), dtype=float)
        for i, idx in enumerate(tier_idx):
            if len(idx) == 0:
                continue
            tier_data = X_padded[idx, :, f]
            counts = (tier_data != 0.0).sum(axis=0)
            sums   = tier_data.sum(axis=0)
            mean_per_tier[i] = np.where(counts > 0, sums / counts, 0.0)

        # Step 2: normalize and cumsum
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

    return scores  # shape: (F, num_present_tiers)

#######################################################
def lcm(a, b):
    return a * b // gcd(a, b)

# takes a time series of any length and converts it into a fixed-length time series
def pool_method_1d(input_array, target_size):
    x = np.asarray(input_array, dtype=np.float32)
    N = x.shape[0]
    K = target_size

    LCM = lcm(N, K)
    L1 = LCM // N
    L2 = LCM // K

    tmp = np.zeros(LCM, dtype=np.float32)
    for i in range(N):
        tmp[i*L1:(i+1)*L1] = x[i]

    out = np.zeros(K, dtype=np.float32)
    for i in range(K):
        chunk = tmp[i*L2:(i+1)*L2]
        out[i] = 0.0 if (i == 0 and x[0] == 0) else float(chunk.mean())
    return out

# Apply Algorithm 2 along TIME axis for (N, T, F)
def pool_method(X, target_size):
    X = np.asarray(X, dtype=np.float32)   # (N, T, F)
    N, T, F = X.shape
    out = np.zeros((N, target_size, F), dtype=np.float32)
    for n in range(N):
        for f in range(F):
            out[n, :, f] = pool_method_1d(X[n, :, f], target_size)
    return out

############################################

# kernel_size is how many time steps it looks at at once and pool size is how many it downsamples/combines
def build_play_pattern_model(F, kernel_size, pool_size):
    model = Sequential()
    model.add(Conv1D(512, kernel_size=kernel_size, activation='relu', padding='valid', input_shape=(None, F)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0005), metrics=['accuracy'])
    return model

# USED FOR POOL SIZE
# determines how many timesteps after a Conv1D + Pooling layer
def output_timesteps(T, kernel_size=3, pool_size=2):
    T1 = T - (kernel_size - 1)
    return T1 // pool_size

# puts a the y label on every time step
def make_timestep_labels(y_bin, T_out):
    return np.repeat(y_bin[:, None], T_out, axis=1)[..., None].astype(np.float32)

# def input_perturbation_feature_importance(model, X, noise_std=0.02, seed=42):
#     rng = np.random.default_rng(seed)
#     N, T, F = X.shape
#     base = model.predict(X, verbose=0)
#     scores = np.zeros(F, dtype=np.float64)
#     for f in range(F):
#         Xn = X.copy()
#         noise = rng.normal(0.0, noise_std, size=(N, T)).astype(X.dtype)
#         Xn[:, :, f] = Xn[:, :, f] + noise
#         pred = model.predict(Xn, verbose=0)
#         scores[f] = float(np.mean(np.abs(pred - base)))
#     return scores

# calculates the AUC over time for plotting 
def auc_over_time(model, X, y_bin):
    N, T, F = X.shape
    y_pred = model.predict(X, verbose=0)[:, :, 0]  # (N, T_out)
    T_out = y_pred.shape[1]
    y_true = make_timestep_labels(y_bin, T_out)[:, :, 0]  # (N, T_out)

    auc_by_min = np.zeros(T_out, dtype=float)
    for m in range(1, T_out + 1):
        auc_by_min[m - 1] = roc_auc_score(y_true[:, :m].reshape(-1), y_pred[:, :m].reshape(-1))
    return auc_by_min
#############################################################
if __name__ == "__main__":
    FEATURE_LIST = [
    'championStats_abilityHaste', 'championStats_abilityPower', 'championStats_armor', 
    'championStats_armorPen', 'championStats_armorPenPercent', 'championStats_attackDamage', 
    'championStats_attackSpeed', 'championStats_bonusArmorPenPercent', 'championStats_bonusMagicPenPercent', 
    'championStats_ccReduction', 'championStats_cooldownReduction', 'championStats_health', 
    'championStats_healthMax', 'championStats_healthRegen', 'championStats_lifesteal', 
    'championStats_magicPen', 'championStats_magicPenPercent', 'championStats_magicResist', 
    'championStats_movementSpeed', 'championStats_omnivamp', 'championStats_physicalVamp', 
    'championStats_power', 'championStats_powerMax', 'championStats_powerRegen', 
    'championStats_spellVamp', 'currentGold', 'damageStats_magicDamageDone', 
    'damageStats_magicDamageDoneToChampions', 'damageStats_magicDamageTaken', 'damageStats_physicalDamageDone', 
    'damageStats_physicalDamageDoneToChampions', 'damageStats_physicalDamageTaken', 'damageStats_totalDamageDone', 
    'damageStats_totalDamageDoneToChampions', 'damageStats_totalDamageTaken', 'damageStats_trueDamageDone', 
    'damageStats_trueDamageDoneToChampions', 'damageStats_trueDamageTaken', 'goldPerSecond', 
    'jungleMinionsKilled', 'level', 'minionsKilled', 'position_x', 
    'position_y', 'timeEnemySpentControlled', 'totalGold', 'xp']

    # load data and split to only take the first 26 min.
    x_data = np.load("X_data_no_padding.npy", allow_pickle=True)
    y_data = np.load("y_data_no_padding.npy")
    puid_data = np.load("puuid_data_no_padding.npy", allow_pickle=True)
    # np.save("X_train_big.npy", X_train)
    # np.save("y_train_big.npy", y_train)
    # np.save("X_test_big.npy", X_test)
    # np.save("y_test_big.npy", y_test)
    # np.save("X_val_big.npy", X_val)
    # np.save("y_val_big.npy", y_val)

    # print(x_data[1])
    print(x_data.shape)
    print(y_data.shape)

    y = (y_data > 3).astype(int)
    print("High tier count:", y.sum(), "/", len(y), "rate:", y.mean())
    print(y[:20])

    # print("X min/max:", x_data.min(), x_data.max())
    # print("X mean/std:", x_data.mean(), x_data.std())
    
    # code for the raw model that uses data from all matches up to 26 minutes, including those that ended before 26 min.
    def raw_model():
        print('entering raw model function')

        F = x_data[0].shape[1]
        T = max(len(seq) for seq in x_data)

        # Pad to uniform array
        X_padded_raw = np.zeros((len(x_data), T, F), dtype=np.float32)
        for i, seq in enumerate(x_data):
            X_padded_raw[i, :len(seq), :] = seq

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_padded_raw, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

        T_out = output_timesteps(T, kernel_size=3, pool_size=2)
        y_train_seq = make_timestep_labels(y_train, T_out)
        y_val_seq   = make_timestep_labels(y_val,   T_out)
        y_test_seq  = make_timestep_labels(y_test,  T_out)

        model = build_play_pattern_model(F, kernel_size=3, pool_size=2)
        model.summary()
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        model.fit(
            X_train, y_train_seq,
            validation_data=(X_val, y_val_seq),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop]
        )

        y_pred = model.predict(X_test, verbose=0)
        # print("y_test_seq shape:", y_test_seq.shape)
        # print(y_pred[:10])
        # y_test_seq shape: (321, 28, 1)



        auc = roc_auc_score(y_test_seq.reshape(-1), y_pred.reshape(-1))
        print("Test AUC:", auc)

        auc_curve = auc_over_time(model, X_test, y_test)
        pool_size = 2
        kernel_size = 3
        effective_minutes = np.minimum((np.arange(1, len(auc_curve) + 1) * pool_size + (kernel_size - 1)), T)
        minutes = effective_minutes

        plt.figure(figsize=(7, 5))
        plt.plot(minutes, auc_curve, label="raw")
        plt.axhline(auc_curve[-1], linestyle="--", linewidth=1)
        plt.text(minutes[0], auc_curve[-1], f"{auc_curve[-1]:.4f}", ha="left", va="bottom", color="k", alpha=0.9)
        plt.axvline(minutes[-1], linestyle="--", linewidth=1)
        plt.title("AUC")
        plt.xlabel("Elapsed Time (minute)")
        plt.ylabel("Probability")
        plt.ylim(0.5, 1.0)
        plt.xlim(1, minutes[-1])
        plt.legend()
        plt.grid(True)
        plt.savefig("alg30_raw_auc_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        return model, auc_curve
    
    # MODEL USING ONLY CUT DATA (26 minutes OR LONGER)
    def cut_model():
        print('entering cut model function')
        # Use totalGold (index 45) as the reliable "is this a real timestep" indicator
        # Every player always has totalGold > 0 from minute 0
        gold_idx = FEATURE_LIST.index('totalGold')
        
        x_25 = X_padded[:, :25, :]

        # Count timesteps where totalGold is nonzero — more reliable than any feature
        real_steps = (x_25[:, :, gold_idx] != 0.0).sum(axis=1)  # (N,)
        mask = real_steps == 25  # game must have real totalGold for all 25 minutes

        x_25 = x_25[mask]
        y_25 = y[mask]

        print("Filtered X:", x_25.shape)
        print("Filtered y:", y_25.shape)


        X_train, X_temp, y_train, y_temp = train_test_split(
            x_25, y_25, test_size=0.3, random_state=42, stratify=y_25)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

        mean = X_train.mean(axis=(0,1), keepdims=True)
        std  = X_train.std(axis=(0,1), keepdims=True) + 1e-8

        X_train = (X_train - mean) / std
        X_val   = (X_val   - mean) / std
        X_test  = (X_test  - mean) / std

        N, T, F = x_25.shape
        print(f"Shape after filtering for 25-minute sequences: N={N}, T={T}, F={F}")

        T_out = output_timesteps(T, kernel_size=3, pool_size=2)
        print("T_out:", T_out)
        y_train_seq = make_timestep_labels(y_train, T_out)
        y_val_seq   = make_timestep_labels(y_val,   T_out)
        y_test_seq  = make_timestep_labels(y_test,  T_out)

        model = build_play_pattern_model(F, kernel_size=3, pool_size=2)
        
        model.summary()
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        model.fit(
            X_train, y_train_seq,
            validation_data=(X_val, y_val_seq),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop]
        )

        y_pred = model.predict(X_test, verbose=0)

        auc = roc_auc_score(y_test_seq.reshape(-1), y_pred.reshape(-1))
        print("Test AUC:", auc)

        auc_curve = auc_over_time(model, X_test, y_test)
        pool_size=2
        kernel_size = 3
        effective_minutes = np.minimum((np.arange(1, len(auc_curve) + 1) * pool_size + (kernel_size - 1)), T)
        minutes = effective_minutes


        plt.figure(figsize=(7, 5))
        plt.plot(minutes, auc_curve, label="cut")
        plt.axhline(auc_curve[-1], linestyle="--", linewidth=1)
        plt.text(minutes[0], auc_curve[-1], f"{auc_curve[-1]:.4f}", ha="left", va="bottom", color="k", alpha=0.9)
        plt.axvline(minutes[-1], linestyle="--", linewidth=1)
        plt.title("AUC")
        plt.xlabel("Elapsed Time (minute)")
        plt.ylabel("Probability")
        plt.ylim(0.5, 1.0)
        plt.xlim(1, minutes[-1])
        plt.legend()
        plt.grid(True)
        plt.savefig("alg30_cut_auc_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        return model, auc_curve
    
    # code for the pool model that uses pooling to lengthen shorter sequences to 26 minutes.
    def pool_model():
        print('entering pool model function')
        x_data_pooled = pool_method(X_padded, target_size=26)
        print(x_data_pooled.shape, 'after pool model slicing')

        X_train, X_temp, y_train, y_temp = train_test_split(
            x_data_pooled, y, test_size=0.3, random_state=42, stratify=y)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

        mean = X_train.mean(axis=(0,1), keepdims=True)
        std  = X_train.std(axis=(0,1), keepdims=True) + 1e-8

        X_train = (X_train - mean) / std
        X_val   = (X_val   - mean) / std
        X_test  = (X_test  - mean) / std

        N, T, F = x_data_pooled.shape

        T_out = output_timesteps(T, kernel_size=3, pool_size=2)
        print("T_out:", T_out)
        y_train_seq = make_timestep_labels(y_train, T_out)
        y_val_seq   = make_timestep_labels(y_val,   T_out)
        y_test_seq  = make_timestep_labels(y_test,  T_out)

        model = build_play_pattern_model(F, kernel_size=3, pool_size=2)
        
        model.summary()
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        model.fit(
            X_train, y_train_seq,
            validation_data=(X_val, y_val_seq),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop]
        )

        y_pred = model.predict(X_test, verbose=0)

        auc = roc_auc_score(y_test_seq.reshape(-1), y_pred.reshape(-1))
        print("Test AUC:", auc)

        auc_curve = auc_over_time(model, X_test, y_test)
        pool_size=2
        kernel_size = 3
        effective_minutes = np.minimum((np.arange(1, len(auc_curve) + 1) * pool_size + (kernel_size - 1)), T)
        minutes = effective_minutes


        plt.figure(figsize=(7, 5))
        plt.plot(minutes, auc_curve, label="pool")
        plt.axhline(auc_curve[-1], linestyle="--", linewidth=1)
        plt.text(minutes[0], auc_curve[-1], f"{auc_curve[-1]:.4f}", ha="left", va="bottom", color="k", alpha=0.9)
        plt.axvline(minutes[-1], linestyle="--", linewidth=1)
        plt.title("AUC")
        plt.xlabel("Elapsed Time (minute)")
        plt.ylabel("Probability")
        plt.ylim(0.5, 1.0)
        plt.xlim(1, minutes[-1])
        plt.legend()
        plt.grid(True)
        plt.savefig("alg30_pool_auc_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        return model, auc_curve
    


    #### ACTUAL RUNNING CODE ####
    F = x_data[0].shape[1]
    T = max(len(seq) for seq in x_data)
    X_padded = np.zeros((len(x_data), T, F), dtype=np.float32)

    def padding_bias_check():
        for i, seq in enumerate(x_data):
            X_padded[i, :len(seq), :] = seq

        nonzero_per_timestep = (X_padded != 0.0).mean(axis=(0, 2))  # shape: (T,)

        plt.plot(nonzero_per_timestep)
        plt.xlabel("Timestep (minute)")
        plt.ylabel("% samples with nonzero data")
        plt.title("Data density over time — shows where padding kicks in")
        plt.axhline(0.5, color='red', linestyle='--', label='50% threshold')
        plt.legend()
        plt.show()
    padding_bias_check()


    def minions_killed_test():
        f_idx = FEATURE_LIST.index("minionsKilled")

        tiers = np.unique(y_data)
        T = X_padded.shape[1]
        plt.figure()

        for tier in tiers:
            idx = np.where(y_data == tier)[0]
            mean_curve = np.zeros(T, dtype=float)

            for t in range(T):
                vals = X_padded[idx, t, f_idx]
                valid = vals != 0
                if valid.any():
                    mean_curve[t] = vals[valid].mean()

            plt.plot(mean_curve, label=f"Tier {tier}")

        plt.xlabel("Time")
        plt.ylabel("Mean minionsKilled (ignoring padded zeros)")
        plt.title("Mean minionsKilled per Tier (no padding bias)")
        plt.legend()
        plt.show()
    minions_killed_test()


    scores = calculate_feature_score(X_padded, y_data, dtw)
    # print(scores)
    print(scores.shape)
    feature_score = scores.mean(axis=1)               # (F,) one score per feature
    ranked = np.argsort(feature_score)[::-1]          # high = more distinctive
    top_k = 10
    print(f"=== Top {top_k} Features ===")
    for idx in ranked[:top_k]:
        print(f"{FEATURE_LIST[idx]:45s}  score = {feature_score[idx]:.6f}")

    print("Minion stats")
    idx = FEATURE_LIST.index("minionsKilled")
    print("minionsKilled score per tier:", scores[idx])
    print("minionsKilled overall score:", scores[idx].mean())

    print(']]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]')
    # print(f"{'Feature':<45} {'% Nonzero':>10} {'Mean (nonzero)':>15}")
    # print("-" * 72)
    # for f_idx, name in enumerate(FEATURE_LIST):
    #     col = X_padded[:, :, f_idx].flatten()
    #     nonzero_mask = col != 0.0
    #     pct_nonzero = nonzero_mask.mean() * 100
    #     mean_nonzero = col[nonzero_mask].mean() if nonzero_mask.any() else 0.0
    #     print(f"{name:<45} {pct_nonzero:>9.1f}% {mean_nonzero:>15.2f}")

    print("################################################")
    def model_testing():

        raw_model_trained, raw_auc_curve = raw_model()
        cut_model_trained, cut_auc_curve = cut_model()
        pool_model_trained, pool_auc_curve = pool_model()

        MAX_MIN = 26
        pool_size = 2
        kernel_size = 3

        plt.figure(figsize=(7, 5))

        curves = [("raw", raw_auc_curve), ("cut", cut_auc_curve), ("pool", pool_auc_curve)]

        for name, auc in curves:
            auc = np.asarray(auc)

            mins = (np.arange(1, len(auc) + 1) * pool_size + (kernel_size - 1))
            mask = mins <= MAX_MIN

            plt.plot(mins[mask], auc[mask], label=name)
            plt.scatter(mins[mask][-1], auc[mask][-1], s=35, zorder=5)
            plt.axhline(auc[mask][-1], ls="--", lw=1, c="k", alpha=0.75)
            plt.text(mins[mask][0], auc[mask][-1], f"{auc[mask][-1]:.4f}",
                    ha="left", va="bottom", c="k", alpha=0.9)


        plt.axvline(MAX_MIN, ls="--", lw=1, c="k", alpha=0.75)
        plt.title("AUC"); plt.xlabel("Elapsed Time (minute)"); plt.ylabel("Probability")
        plt.ylim(0.5, 1.0); plt.xlim(0, MAX_MIN)
        plt.legend(); plt.grid(True)
        plt.savefig("alg30_combined_auc_plot.png", dpi=300, bbox_inches="tight")
        plt.show()
        return raw_model_trained, cut_model_trained, pool_model_trained
    


    print('################################################################################')
    # determining if they a smurf
    def smurf_detection(x_data_test, y_data_test, puid_data_test):
        raw_model_trained, cut_model_trained, pool_model_trained = model_testing()
        print('entering smurf detection function')

        X_padded = np.zeros((len(x_data_test), T, F), dtype=np.float32)
        for i, seq in enumerate(x_data_test):
            X_padded[i, :len(seq), :] = seq

        raw_pred = raw_model_trained.predict(X_padded, verbose=0)

        cut_pred = cut_model_trained.predict(X_padded, verbose=0)

        x_data_pooled = pool_method(X_padded, target_size=26)
        pool_pred = pool_model_trained.predict(x_data_pooled, verbose=0)

        raw_pred_prob  = raw_pred[:, -1, 0]
        cut_pred_prob  = cut_pred[:, -1, 0]
        pool_pred_prob = pool_pred[:, -1, 0]

        df = pd.DataFrame({
            "puuid": puid_data_test,
            "raw_prob": raw_pred_prob,
            "cut_prob": cut_pred_prob,
            "pool_prob": pool_pred_prob,
            "label": y_data_test
        })

        df_player = df.groupby("puuid", as_index=False).agg({
        "raw_prob": "mean",
        "cut_prob": "mean",
        "pool_prob": "mean",
        "label": "first"})
        # smurf = predicted high (1) but actually low (0)
        df_player["raw_smurf"]  = ((df_player["raw_pred"] == 1) & (df_player["label"] == 0)).astype(int)
        df_player["cut_smurf"]  = ((df_player["cut_pred"] == 1) & (df_player["label"] == 0)).astype(int)
        df_player["pool_smurf"] = ((df_player["pool_pred"] == 1) & (df_player["label"] == 0)).astype(int)

        print(df_player.head(3))

        auc_raw  = roc_auc_score(df_player["label"], df_player["raw_prob"])
        auc_cut  = roc_auc_score(df_player["label"], df_player["cut_prob"])
        auc_pool = roc_auc_score(df_player["label"], df_player["pool_prob"])

        print("Player-level AUC:")
        print(auc_raw, auc_cut, auc_pool)

        


    smurf_detection(x_data, y, puid_data)

###############################################################################
    # teirs = len(np.unique(y_data))

    # N, T, F = x_data.shape
    # timeline_match = np.zeros((teirs, N, T, F), dtype=x_data.dtype)

    # for j in range(N):
    #     tier = y_data[j]
    #     timeline_match[tier, j, :, :] = x_data[j]
    # print(timeline_match.shape)

    # feature_scores = calculate_feature_score(timeline_match, dtw)
    # # print(feature_scores)
    # print(feature_scores.shape)






    
  







    



r""" 
 (base) PS C:\Users\rowan\Desktop\Classes\DS 340w\DS-340w-Smurf-Detection> & C:/ProgramData/anaconda3/python.exe "c:/Users/rowan/Desktop/Classes/DS 340w/DS-340w-Smurf-Detection/algs.py"
2026-02-01 12:48:17.698858: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-01 12:48:19.480770: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
(175410, 26, 47)
(175410,)
(10, 175410, 26, 47)
(47, 10)
High tier count: 32720 / 175410 rate: 0.1865344051080326
X min/max: -31.0 4231932.0
X mean/std: 5486.0967 27752.87
C:\Users\rowan\AppData\Roaming\Python\Python312\site-packages\keras\src\layers\convolutional\base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
2026-02-01 12:59:02.107034: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv1d (Conv1D)                      │ (None, None, 512)           │          72,704 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, None, 512)           │           2,048 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling1d (MaxPooling1D)         │ (None, None, 512)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm (LSTM)                          │ (None, None, 256)           │         787,456 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, None, 256)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, None, 128)           │         197,120 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, None, 128)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_2 (LSTM)                        │ (None, None, 96)            │          86,400 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, None, 96)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ time_distributed (TimeDistributed)   │ (None, None, 1)             │              97 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,145,825 (4.37 MB)
 Trainable params: 1,144,801 (4.37 MB)
 Non-trainable params: 1,024 (4.00 KB)
Epoch 1/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 112s 29ms/step - accuracy: 0.8137 - loss: 0.4465 - val_accuracy: 0.8143 - val_loss: 0.4347
Epoch 2/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 100s 26ms/step - accuracy: 0.8159 - loss: 0.4245 - val_accuracy: 0.8090 - val_loss: 0.4329
Epoch 3/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 102s 27ms/step - accuracy: 0.8172 - loss: 0.4174 - val_accuracy: 0.8189 - val_loss: 0.4114
Epoch 4/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 103s 27ms/step - accuracy: 0.8189 - loss: 0.4122 - val_accuracy: 0.8145 - val_loss: 0.4160
Epoch 5/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 101s 26ms/step - accuracy: 0.8210 - loss: 0.4072 - val_accuracy: 0.8204 - val_loss: 0.4094
Epoch 6/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 101s 26ms/step - accuracy: 0.8218 - loss: 0.4030 - val_accuracy: 0.8200 - val_loss: 0.4065
Epoch 7/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 102s 27ms/step - accuracy: 0.8232 - loss: 0.3997 - val_accuracy: 0.8214 - val_loss: 0.4074
Epoch 8/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 102s 27ms/step - accuracy: 0.8243 - loss: 0.3959 - val_accuracy: 0.8193 - val_loss: 0.4140
Epoch 9/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 102s 27ms/step - accuracy: 0.8264 - loss: 0.3923 - val_accuracy: 0.8219 - val_loss: 0.4047
Epoch 10/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 109s 28ms/step - accuracy: 0.8273 - loss: 0.3888 - val_accuracy: 0.8217 - val_loss: 0.4032
Epoch 11/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 107s 28ms/step - accuracy: 0.8287 - loss: 0.3865 - val_accuracy: 0.8146 - val_loss: 0.4114
Epoch 12/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 106s 28ms/step - accuracy: 0.8300 - loss: 0.3827 - val_accuracy: 0.8199 - val_loss: 0.4213
Epoch 13/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 106s 28ms/step - accuracy: 0.8322 - loss: 0.3787 - val_accuracy: 0.8218 - val_loss: 0.4128
Epoch 14/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 107s 28ms/step - accuracy: 0.8339 - loss: 0.3753 - val_accuracy: 0.8222 - val_loss: 0.3985
Epoch 15/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 106s 28ms/step - accuracy: 0.8355 - loss: 0.3713 - val_accuracy: 0.8221 - val_loss: 0.4173
Epoch 16/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 106s 28ms/step - accuracy: 0.8373 - loss: 0.3673 - val_accuracy: 0.8212 - val_loss: 0.4228
Epoch 17/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 104s 27ms/step - accuracy: 0.8396 - loss: 0.3633 - val_accuracy: 0.8222 - val_loss: 0.4181
Epoch 18/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 105s 27ms/step - accuracy: 0.8421 - loss: 0.3585 - val_accuracy: 0.8199 - val_loss: 0.4069
Epoch 19/100
3838/3838 ━━━━━━━━━━━━━━━━━━━━ 108s 28ms/step - accuracy: 0.8441 - loss: 0.3537 - val_accuracy: 0.8208 - val_loss: 0.4320
Test AUC: 0.7854177739987331
Top 10 features (index -> score):
41 -> 0.004216632805764675
12 -> 0.002903253771364689
29 -> 0.0023104557767510414
46 -> 0.001957567408680916
45 -> 0.0018193019786849618
39 -> 0.0017028233269229531
32 -> 0.0016835454152897
42 -> 0.0016491854330524802
21 -> 0.0015587561065331101
5 -> 0.0015079252189025283
(base) PS C:\Users\rowan\Desktop\Classes\DS 340w\DS-340w-Smurf-Detection> """
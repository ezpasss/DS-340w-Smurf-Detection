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


def calculate_feature_score(timeline_match, dtw):
    num_tiers, N, T, F = timeline_match.shape
    feature_scores = [[0.0 for _ in range(num_tiers)] for _ in range(F)]

    for f in range(F):
        mean_per_tier = [[0.0 for _ in range(T)] for _ in range(num_tiers)]

        for i in range(num_tiers):
            for t in range(T):
                sum1 = 0.0
                for j in range(N):
                    sum1 += timeline_match[i][j][t][f]
                mean_per_tier[i][t] = sum1 / N

        cumsum = [[0.0 for _ in range(T)] for _ in range(num_tiers)]

        for i in range(num_tiers):
            sum_feature = sum(mean_per_tier[i])
            if sum_feature == 0:
                continue
            else:
                sum2 = 0.0
                for t in range(T):
                    sum2 += mean_per_tier[i][t] / sum_feature
                    cumsum[i][t] = sum2

        for i in range(num_tiers):
            sum3 = 0.0
            for j in range(num_tiers):
                sum3 += dtw(cumsum[i], cumsum[j])
            feature_scores[f][i] = sum3 / num_tiers

    return np.array(feature_scores)

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

def input_perturbation_feature_importance(model, X, noise_std=0.02, seed=42):
    rng = np.random.default_rng(seed)
    N, T, F = X.shape
    base = model.predict(X, verbose=0)
    scores = np.zeros(F, dtype=np.float64)
    for f in range(F):
        Xn = X.copy()
        noise = rng.normal(0.0, noise_std, size=(N, T)).astype(X.dtype)
        Xn[:, :, f] = Xn[:, :, f] + noise
        pred = model.predict(Xn, verbose=0)
        scores[f] = float(np.mean(np.abs(pred - base)))
    return scores

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
    x_data = np.load("X_47_t60.npy")[:20000, :, :]
    y_data = np.load("y_47_t60.npy")[:20000]

    print(x_data.shape)
    print(y_data.shape)

    y = (y_data > 5).astype(int)
    print("High tier count:", y.sum(), "/", len(y), "rate:", y.mean())

    print("X min/max:", x_data.min(), x_data.max())
    print("X mean/std:", x_data.mean(), x_data.std())
    
    # code for the raw model that uses data from all matches up to 26 minutes, including those that ended before 26 min.
    def raw_model():
        print('entering raw model function')

        X_train, X_temp, y_train, y_temp = train_test_split(
            x_data, y, test_size=0.3, random_state=42, stratify=y)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

        mean = X_train.mean(axis=(0,1), keepdims=True)
        std  = X_train.std(axis=(0,1), keepdims=True) + 1e-8

        X_train = (X_train - mean) / std
        X_val   = (X_val   - mean) / std
        X_test  = (X_test  - mean) / std

        N, T, F = x_data.shape

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

        # print('Calculating feature importance scores using input_perturbation_feature_importance...')
        # scores = input_perturbation_feature_importance(model, X_test, noise_std=0.02)
        # top10 = np.argsort(scores)[::-1][:10]
        # print("Top 10 features (index -> score):")
        # for idx in top10:
        #     print(idx, "->", scores[idx])

        auc_curve = auc_over_time(model, X_test, y_test)
        pool_size=2
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
        plt.savefig("test60_raw_auc_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        return auc_curve
    
    # MODEL USING ONLY CUT DATA (26 minutes OR LONGER)
    def cut_model():
        print('entering cut model function')
        x_26 = x_data[:,:26,:]
        valid_steps = np.any(x_26 != 0.0, axis=2)   # (N, 25) boolean
        lengths = valid_steps.sum(axis=1)           # (N,)

        mask = lengths == 26                        # must have all 26 minutes real

        x_26 = x_26[mask]
        y_26 = y[mask]

        print("Filtered X:", x_26.shape)
        print("Filtered y:", y_26.shape)


        X_train, X_temp, y_train, y_temp = train_test_split(
            x_26, y_26, test_size=0.3, random_state=42, stratify=y_26)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

        mean = X_train.mean(axis=(0,1), keepdims=True)
        std  = X_train.std(axis=(0,1), keepdims=True) + 1e-8

        X_train = (X_train - mean) / std
        X_val   = (X_val   - mean) / std
        X_test  = (X_test  - mean) / std

        N, T, F = x_26.shape

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
        plt.savefig("test60_cut_auc_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        return auc_curve
    
    # code for the pool model that uses pooling to lengthen shorter sequences to 26 minutes.
    def pool_model():
        print('entering pool model function')
        x_data_pooled = pool_method(x_data, target_size=26)
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

        # print('Calculating feature importance scores using input_perturbation_feature_importance...')
        # scores = input_perturbation_feature_importance(model, X_test, noise_std=0.02)
        # top10 = np.argsort(scores)[::-1][:10]
        # print("Top 10 features (index -> score):")
        # for idx in top10:
        #     print(idx, "->", scores[idx])

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
        plt.savefig("test60_pool_auc_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        return auc_curve
    


    #### ACTUAL RUNNING CODE ####


    raw_auc_curve = raw_model()
    cut_auc_curve = cut_model()
    pool_auc_curve = pool_model()

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
    plt.savefig("test60_combined_auc_plot.png", dpi=300, bbox_inches="tight")
    plt.show()



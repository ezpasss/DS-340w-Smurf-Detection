import pandas as pd
import numpy as np
from math import gcd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dropout, TimeDistributed, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_DIR = "data"
OUTPUT_FILE = "justice_league_dataset_final.csv"

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

def resample_array(input_array, target_size):
    input_array = np.array(input_array, dtype=float)
    N = len(input_array)
    K = target_size

    LCM = lcm(N, K)

    tmp_array = np.zeros(LCM, dtype=float)
    L1 = LCM // N
    for i in range(N):
        for j in range(L1):
            tmp_array[i * L1 + j] = input_array[i]

    output_array = np.zeros(K, dtype=float)
    L2 = LCM // K
    for i in range(K):
        chunk = tmp_array[i * L2 : (i + 1) * L2]
        s = chunk.sum()
        if i == 0 and input_array[0] == 0:
            output_array[i] = 0
        else:
            output_array[i] = s / L2

    return output_array

############################################

def build_play_pattern_model(F, kernel_size=3, pool_size=2):
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

def output_timesteps(T, kernel_size=3, pool_size=2):
    T1 = T - (kernel_size - 1)
    return T1 // pool_size

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

def auc_over_time(model, X, y_bin, kernel_size=3, pool_size=2):
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
    'position_y', 'timeEnemySpentControlled', 'totalGold', 'xp'
]

    all_data = []
    x_data = np.load("X_47.npy")
    print(x_data.shape)
    y_data = np.load("y_47.npy")
    print(y_data.shape)

    teirs = len(np.unique(y_data))

    N, T, F = x_data.shape
    timeline_match = np.zeros((teirs, N, T, F), dtype=x_data.dtype)

    for j in range(N):
        tier = y_data[j]
        timeline_match[tier, j, :, :] = x_data[j]
    print(timeline_match.shape)

    feature_scores = calculate_feature_score(timeline_match, dtw)
    # print(feature_scores)
    print(feature_scores.shape)


    y = (y_data > 6).astype(int)
    print("High tier count:", y.sum(), "/", len(y), "rate:", y.mean())

    print("X min/max:", x_data.min(), x_data.max())
    print("X mean/std:", x_data.mean(), x_data.std())



    N, T, F = x_data.shape
    T_out = output_timesteps(T, kernel_size=3, pool_size=2)

    X_train, X_temp, y_train, y_temp = train_test_split(
        x_data, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
    )

    mean = X_train.mean(axis=(0,1), keepdims=True)
    std  = X_train.std(axis=(0,1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)
    np.save("X_val.npy", X_val)
    np.save("y_val.npy", y_val)

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
    flattened_auc = roc_auc_score(
    y_test_seq.reshape(-1),
    y_pred.reshape(-1))

    auc = roc_auc_score(y_test_seq.reshape(-1), y_pred.reshape(-1))
    print("Test AUC:", auc)

    scores = input_perturbation_feature_importance(model, X_test, noise_std=0.02)
    top10 = np.argsort(scores)[::-1][:10]
    print("Top 10 features (index -> score):")
    for idx in top10:
        print(idx, "->", scores[idx])

    auc_curve = auc_over_time(model, X_test, y_test)
    pool_size = 2
    kernel_size = 3
    effective_minutes = np.minimum((np.arange(1, len(auc_curve) + 1) * pool_size + (kernel_size - 1)), T)
    minutes = effective_minutes


    plt.figure(figsize=(7, 5))
    plt.plot(minutes, auc_curve, label="raw")
    plt.axhline(auc_curve[-1], linestyle="--", linewidth=1)
    plt.axvline(minutes[-1], linestyle="--", linewidth=1)
    plt.title("AUC")
    plt.xlabel("Elapsed Time (minute)")
    plt.ylabel("Probability")
    plt.ylim(0.5, 1.0)
    plt.xlim(1, minutes[-1])
    plt.legend()
    plt.grid(True)
    plt.show()
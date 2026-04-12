import json
import numpy as np
import os

INPUT_FILE = "Full_Data.jsonl"
OUTPUT_X = "X_data_no_padding.npy"
OUTPUT_Y = "y_data_no_padding.npy"
OUTPUT_PUUID = "puuid_data_no_padding.npy"

RANK_MAP = {
    'IRON': 0, 'BRONZE': 1, 'SILVER': 2, 'GOLD': 3, 'PLATINUM': 4,
    'EMERALD': 5, 'DIAMOND': 6, 'MASTER': 7, 'GRANDMASTER': 8, 'CHALLENGER': 9
}

def process():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Store as list of arrays — each a different length, no padding
    X_list = []
    y_list = []
    puuid_list = []

    print(f"Processing {INPUT_FILE}...")

    with open(INPUT_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line)

                tier = data.get("tier")
                if tier not in RANK_MAP:
                    continue

                frames = data.get("frames")
                if not frames or len(frames) == 0:
                    continue

                # frames is already (T_actual, 47) — just convert to numpy
                seq = np.array(frames, dtype=np.float32)  # shape: (T_actual, 47)

                X_list.append(seq)
                y_list.append(RANK_MAP[tier])
                puuid_list.append(data["puuid"])

            except json.JSONDecodeError:
                continue

    # X is a numpy object array — each element is (T_i, 47), different T per game
    X = np.array(X_list, dtype=object)
    y = np.array(y_list, dtype=np.int32)
    puuids = np.array(puuid_list)

    print(f"\nTotal samples: {len(y)}")
    print(f"X: {len(X)} sequences, each shape (T_i, 47)")
    print(f"y shape: {y.shape}")
    print(f"Frame length range: {min(len(s) for s in X)} - {max(len(s) for s in X)} minutes")

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    np.save(OUTPUT_PUUID, puuids)
    print("[✓] Saved X_data.npy, y_data.npy, puuid_data.npy")

if __name__ == "__main__":
    process()

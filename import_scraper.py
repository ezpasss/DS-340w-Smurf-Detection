import time
import random
import json
import os
import signal
import sys
from riotwatcher import LolWatcher, ApiError

# --- CONFIGURATION ---
# Set your key in PowerShell before running:
#   $env:RIOT_API_KEY="RGAPI-...your key..."
API_KEY = "RGAPI-f58e41fa-43e7-4acf-81bc-f931567a20b1"
REGION_PLATFORM = "na1"      # league-v4, summoner-v4 live on platform routing
REGION_ROUTING = "AMERICAS"  # match-v5 + timeline-v5 use regional routing for NA
OUTPUT_FILE = "Full_Data.jsonl"

# How deep to go per player?
MATCHES_PER_PLAYER = 3

# Ranks to cycle through
TIERS = [
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM",
    "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"
]
DIVISIONS = ["I", "II", "III", "IV"]

if not API_KEY or not API_KEY.startswith("RGAPI-"):
    print("❌ API_KEY missing/invalid. Paste a valid RGAPI key.")
    sys.exit(1)

watcher = LolWatcher(API_KEY)
seen_match_ids = set()
running = True

# --- 1) GRACEFUL SHUTDOWN HANDLER ---
def signal_handler(sig, frame):
    global running
    print("\n\n[!] Ctrl+C received. Finishing current item and saving...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# --- 2) LOAD PREVIOUS PROGRESS ---
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                seen_match_ids.add(data["match_id"])
            except:
                pass
    print(f"Loaded {len(seen_match_ids)} existing matches from database.")

# --- 3) HELPER FUNCTIONS ---
def get_puuid_robust(summoner_id):
    try:
        summ = watcher.summoner.by_id(REGION_PLATFORM, summoner_id)
        return summ["puuid"]
    except ApiError:
        return None

# ANY LENGTH: no minimum frame requirement
def get_timeline_any_length(match_id):
    try:
        timeline = watcher.match.timeline_by_match(REGION_ROUTING, match_id)
        frames = timeline["info"]["frames"]
        if not frames:
            return None

        extracted = []
        for i, frame in enumerate(frames):
            extracted.append({
                "minute": i,
                "participantFrames": frame["participantFrames"]
            })
        return extracted
    except:
        return None

# --- 4) MAIN LOOP ---
print("--- STARTING OVERNIGHT HARVEST (Press Ctrl+C to Stop) ---")

while running:
    tier = random.choice(TIERS)
    division = random.choice(DIVISIONS) if tier not in ["MASTER", "GRANDMASTER", "CHALLENGER"] else "I"

    print(f"\n>>> Scavenging {tier} {division}...")

    try:
        # Get players for this rank
        if tier in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
            if tier == "MASTER":
                func = watcher.league.masters_by_queue
            elif tier == "GRANDMASTER":
                func = watcher.league.grandmaster_by_queue
            else:
                func = watcher.league.challenger_by_queue

            entries = func(REGION_PLATFORM, "RANKED_SOLO_5x5")["entries"]
        else:
            page = random.randint(1, 5)
            entries = watcher.league.entries(
                REGION_PLATFORM, "RANKED_SOLO_5x5", tier, division, page=page
            )

        if not entries:
            continue

        random.shuffle(entries)
        batch = entries[:5]

        for entry in batch:
            if not running:
                break

            puuid = entry.get("puuid")
            if not puuid:
                puuid = get_puuid_robust(entry["summonerId"])
            if not puuid:
                continue

            # Get recent matches for this player (match-v5 uses REGION_ROUTING)
            try:
                matches = watcher.match.matchlist_by_puuid(
                    REGION_ROUTING, puuid, queue=420, count=MATCHES_PER_PLAYER
                )
            except ApiError:
                continue

            print(f"  > Processing {entry.get('summonerName', 'Player')} ({len(matches)} matches)...")

            for m_id in matches:
                if not running:
                    break
                if m_id in seen_match_ids:
                    continue

                timeline = get_timeline_any_length(m_id)
                if timeline:
                    record = {
                        "match_id": m_id,
                        "tier": tier,
                        "division": division,
                        "timeline": timeline
                    }

                    with open(OUTPUT_FILE, "a") as f:
                        f.write(json.dumps(record) + "\n")

                    seen_match_ids.add(m_id)
                    print(f"    + Secured Match {m_id} (frames={len(timeline)})")

                time.sleep(1.2)

    except ApiError as e:
        # 401 = bad/expired key. 429 = rate limit. Riotwatcher wraps both as ApiError.
        print(f"  ! API Error: {e}")
        time.sleep(10)
    except Exception as e:
        print(f"  ! Error: {e}")

print("\n[✓] Script stopped safely. All data saved.")
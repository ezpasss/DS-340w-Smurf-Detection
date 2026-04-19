import time
import random
import json
import os
import signal
import sys
from riotwatcher import LolWatcher, ApiError

API_KEY = "RGAPI-da2073f2-56af-40fe-921e-6970dc19a532"  # <-- REPLACE with your Riot API key
REGION_PLATFORM = "na1"
REGION_ROUTING = "AMERICAS"
OUTPUT_FILE = "Full_Data.jsonl"
MATCHES_PER_PLAYER = 10

TIERS = [
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM",
    "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"
]
DIVISIONS = ["I", "II", "III", "IV"]
TIER_TO_IDX = {t: i for i, t in enumerate(TIERS)}

watcher = LolWatcher(API_KEY)
seen_player_matches = set()
running = True

# if ctrl+c is entered makes sure outputs are saved to json before closing. 
def signal_handler(sig, frame):
    global running
    print("\n[!] Ctrl+C received. Finishing current item...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# opens the json file and loads existing saved data
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                seen_player_matches.add((data["puuid"], data["match_id"]))
            except:
                pass
    print(f"Loaded {len(seen_player_matches)} existing player-game records.")

# api calling to get a summoners puuid from their summoner id. 
def get_puuid_robust(summoner_id):
    try:
        summ = watcher.summoner.by_id(REGION_PLATFORM, summoner_id)
        return summ["puuid"]
    except ApiError:
        return None

# collects a frame of time series data for a match
def extract_frame_vector(p_data):
    """Extract all 47 features from a single participant frame."""
    cs = p_data.get("championStats", {})
    ds = p_data.get("damageStats", {})
    pos = p_data.get("position", {"x": 0, "y": 0})

    return [
        cs.get("abilityHaste", 0),
        cs.get("abilityPower", 0),
        cs.get("armor", 0),
        cs.get("armorPen", 0),
        cs.get("armorPenPercent", 0),
        cs.get("attackDamage", 0),
        cs.get("attackSpeed", 0),
        cs.get("bonusArmorPenPercent", 0),
        cs.get("bonusMagicPenPercent", 0),
        cs.get("ccReduction", 0),
        cs.get("cooldownReduction", 0),
        cs.get("health", 0),
        cs.get("healthMax", 0),
        cs.get("healthRegen", 0),
        cs.get("lifesteal", 0),
        cs.get("magicPen", 0),
        cs.get("magicPenPercent", 0),
        cs.get("magicResist", 0),
        cs.get("movementSpeed", 0),
        cs.get("omnivamp", 0),
        cs.get("physicalVamp", 0),
        cs.get("power", 0),
        cs.get("powerMax", 0),
        cs.get("powerRegen", 0),
        cs.get("spellVamp", 0),
        p_data.get("currentGold", 0),
        ds.get("magicDamageDone", 0),
        ds.get("magicDamageDoneToChampions", 0),
        ds.get("magicDamageTaken", 0),
        ds.get("physicalDamageDone", 0),
        ds.get("physicalDamageDoneToChampions", 0),
        ds.get("physicalDamageTaken", 0),
        ds.get("totalDamageDone", 0),
        ds.get("totalDamageDoneToChampions", 0),
        ds.get("totalDamageTaken", 0),
        ds.get("trueDamageDone", 0),
        ds.get("trueDamageDoneToChampions", 0),
        ds.get("trueDamageTaken", 0),
        p_data.get("goldPerSecond", 0),
        p_data.get("jungleMinionsKilled", 0),
        p_data.get("level", 1),
        p_data.get("minionsKilled", 0),
        pos.get("x", 0),
        pos.get("y", 0),
        p_data.get("timeEnemySpentControlled", 0),
        p_data.get("totalGold", 0),
        p_data.get("xp", 0),
    ]

# api call to get the timeline of a match and extract the 47 features for each frame for a given player
def get_player_timeline(match_id, puuid):
    """
    Returns list of 47-feature vectors at natural game length — no padding.
    """
    try:
        timeline = watcher.match.timeline_by_match(REGION_ROUTING, match_id)
        info = timeline["info"]
        frames = info["frames"]
        if not frames:
            return None

        # Find this player's participant slot
        participant_id = None
        for p in info.get("participants", []):
            if p["puuid"] == puuid:
                participant_id = p["participantId"]
                break

        if participant_id is None:
            print(f"    ! PUUID not found in match {match_id}")
            return None

        player_frames = []
        for frame in frames:
            p_data = frame["participantFrames"].get(str(participant_id), {})
            player_frames.append(extract_frame_vector(p_data))

        return player_frames

    except Exception as e:
        print(f"    ! Timeline error for {match_id}: {e}")
        return None

# --- MAIN LOOP ---
print("--- STARTING HARVEST (Ctrl+C to stop) ---")

# runs until ctrl+c
while running:
    # random select a tier and devision to scrape from. 
    tier = random.choice(TIERS)
    division = random.choice(DIVISIONS) if tier not in ["MASTER", "GRANDMASTER", "CHALLENGER"] else "I"
    print(f"\n>>> Scavenging {tier} {division}...")

    try:
        # if player is in master, grandmaster, or challenger tiers their is no devision
        # randomly slects a page of players to scrape
        if tier in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
            func = {
                "MASTER": watcher.league.masters_by_queue,
                "GRANDMASTER": watcher.league.grandmaster_by_queue,
                "CHALLENGER": watcher.league.challenger_by_queue
            }[tier]
            entries = func(REGION_PLATFORM, "RANKED_SOLO_5x5")["entries"]
        else:
            page = random.randint(1, 5)
            entries = watcher.league.entries(
                REGION_PLATFORM, "RANKED_SOLO_5x5", tier, division, page=page
            )

        if not entries:
            continue
        
        # randomly shuffle the players
        random.shuffle(entries)
        # for the first 5 players in the list, get their puuid and then get a list of their most recent matches
        for entry in entries[:5]:
            if not running:
                break

            puuid = entry.get("puuid") or get_puuid_robust(entry["summonerId"])
            if not puuid:
                continue

            try:
                match_ids = watcher.match.matchlist_by_puuid(
                    REGION_ROUTING, puuid, queue=420, count=MATCHES_PER_PLAYER
                )
            except ApiError:
                continue

            print(f"  > Player {puuid[:12]}... ({len(match_ids)} matches)")

            # for each match id get the timeline and extract the 47 features for each frame and save to jsonl file.
            for m_id in match_ids:
                if not running:
                    break
                if (puuid, m_id) in seen_player_matches:
                    continue

                player_frames = get_player_timeline(m_id, puuid)
                if player_frames:
                    record = {
                        "puuid": puuid,
                        "match_id": m_id,
                        "tier": tier,
                        "tier_idx": TIER_TO_IDX[tier],
                        "division": division,
                        "num_frames": len(player_frames),
                        "frames": player_frames  # shape: (T_actual, 47)
                    }

                    with open(OUTPUT_FILE, "a") as f:
                        f.write(json.dumps(record) + "\n")

                    seen_player_matches.add((puuid, m_id))
                    print(f"    + {puuid[:12]}... | {m_id} | {len(player_frames)} frames")

                # added to prevent overcalling the api and getting rate limited.
                time.sleep(1.2)

    except ApiError as e:
        print(f"  ! API Error: {e}")
        time.sleep(10)
    except Exception as e:
        print(f"  ! Error: {e}")

print("\n[✓] Stopped safely.")
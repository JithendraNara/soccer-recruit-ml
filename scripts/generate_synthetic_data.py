"""Generate synthetic but realistic player data for testing.

Creates ~500 players across 5 leagues with position-correlated stats.
Stats follow real distributions observed in football:
- Age: 18-38 (peak ~27)
- Height/weight: position-correlated (CB taller, FW lighter)
- Goals: highly position-correlated (FW >> CB)
- Value: age * performance curve
- Wage: ~0.1% of value (industry standard)

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --count 1000
"""
import sys
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from faker import Faker

from src.data.database import SessionLocal, init_db
from src.data.repositories import PlayerRepository
from src.utils.logger import logger

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Real leagues + their "tier" (affects value/wage)
LEAGUES = {
    "Premier League": 1.0,
    "La Liga": 0.95,
    "Bundesliga": 0.85,
    "Serie A": 0.80,
    "Ligue 1": 0.65,
    "Eredivisie": 0.40,
    "Primeira Liga": 0.35,
    "MLS": 0.30,
}

LEAGUE_TEAMS = {
    "Premier League": ["Manchester City", "Arsenal", "Liverpool", "Manchester United", "Chelsea",
                       "Tottenham", "Newcastle", "Aston Villa", "Brighton", "West Ham",
                       "Crystal Palace", "Fulham", "Wolves", "Everton", "Brentford"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Real Sociedad",
                "Villarreal", "Real Betis", "Athletic Bilbao", "Valencia", "Osasuna"],
    "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "Bayer Leverkusen", "RB Leipzig",
                   "Eintracht Frankfurt", "Wolfsburg", "Union Berlin", "Stuttgart", "Hoffenheim"],
    "Serie A": ["Inter Milan", "AC Milan", "Juventus", "Napoli", "Roma", "Lazio",
                "Atalanta", "Fiorentina", "Bologna", "Torino"],
    "Ligue 1": ["PSG", "Marseille", "Monaco", "Lyon", "Lille", "Rennes", "Nice", "Lens"],
    "Eredivisie": ["Ajax", "PSV Eindhoven", "Feyenoord", "AZ Alkmaar", "FC Twente"],
    "Primeira Liga": ["Benfica", "Porto", "Sporting CP", "Braga", "Vitória"],
    "MLS": ["Inter Miami", "LA Galaxy", "LAFC", "Atlanta United", "NYC FC", "Seattle Sounders"],
}

POSITIONS = {
    "GK": {"count": 1, "tier": 0.6},
    "CB": {"count": 2, "tier": 0.7},
    "LB": {"count": 1, "tier": 0.65},
    "RB": {"count": 1, "tier": 0.65},
    "DM": {"count": 1, "tier": 0.75},
    "CM": {"count": 2, "tier": 0.85},
    "CAM": {"count": 1, "tier": 0.95},
    "LW": {"count": 1, "tier": 1.0},
    "RW": {"count": 1, "tier": 1.0},
    "ST": {"count": 1, "tier": 1.1},
    "CF": {"count": 1, "tier": 1.05},
}

# Position → physical/performance profile
POSITION_PROFILES = {
    "GK": {
        "height_mean": 189, "height_std": 4, "weight_mean": 84, "weight_std": 5,
        "goals_per_game": (0.0, 0.02), "assists_per_game": (0.0, 0.05),
        "saves_per_game": (2.5, 1.0), "clean_sheet_p": 0.4,
        "tackles_per_game": (0.3, 0.4), "interceptions_per_game": (0.2, 0.3),
        "pass_accuracy": (78, 8), "shots_per_game": (0.05, 0.05),
    },
    "CB": {
        "height_mean": 188, "height_std": 4, "weight_mean": 82, "weight_std": 5,
        "goals_per_game": (0.05, 0.05), "assists_per_game": (0.03, 0.04),
        "saves_per_game": (0, 0), "clean_sheet_p": 0.4,
        "tackles_per_game": (3.0, 1.0), "interceptions_per_game": (2.0, 0.8),
        "pass_accuracy": (84, 6), "shots_per_game": (0.4, 0.3),
    },
    "LB": {
        "height_mean": 180, "height_std": 4, "weight_mean": 73, "weight_std": 4,
        "goals_per_game": (0.05, 0.05), "assists_per_game": (0.2, 0.15),
        "saves_per_game": (0, 0), "clean_sheet_p": 0.35,
        "tackles_per_game": (2.8, 0.9), "interceptions_per_game": (1.5, 0.6),
        "pass_accuracy": (82, 6), "shots_per_game": (0.5, 0.3),
    },
    "RB": {
        "height_mean": 180, "height_std": 4, "weight_mean": 73, "weight_std": 4,
        "goals_per_game": (0.05, 0.05), "assists_per_game": (0.2, 0.15),
        "saves_per_game": (0, 0), "clean_sheet_p": 0.35,
        "tackles_per_game": (2.8, 0.9), "interceptions_per_game": (1.5, 0.6),
        "pass_accuracy": (82, 6), "shots_per_game": (0.5, 0.3),
    },
    "DM": {
        "height_mean": 183, "height_std": 4, "weight_mean": 76, "weight_std": 4,
        "goals_per_game": (0.1, 0.08), "assists_per_game": (0.15, 0.1),
        "saves_per_game": (0, 0), "clean_sheet_p": 0.35,
        "tackles_per_game": (3.5, 1.0), "interceptions_per_game": (2.2, 0.8),
        "pass_accuracy": (86, 5), "shots_per_game": (0.7, 0.4),
    },
    "CM": {
        "height_mean": 180, "height_std": 4, "weight_mean": 73, "weight_std": 4,
        "goals_per_game": (0.2, 0.15), "assists_per_game": (0.25, 0.18),
        "saves_per_game": (0, 0), "clean_sheet_p": 0.3,
        "tackles_per_game": (2.5, 0.9), "interceptions_per_game": (1.5, 0.6),
        "pass_accuracy": (87, 5), "shots_per_game": (1.2, 0.6),
    },
    "CAM": {
        "height_mean": 178, "height_std": 4, "weight_mean": 70, "weight_std": 4,
        "goals_per_game": (0.4, 0.2), "assists_per_game": (0.4, 0.2),
        "saves_per_game": (0, 0), "clean_sheet_p": 0.25,
        "tackles_per_game": (1.5, 0.7), "interceptions_per_game": (1.0, 0.5),
        "pass_accuracy": (85, 5), "shots_per_game": (2.0, 0.8),
    },
    "LW": {
        "height_mean": 177, "height_std": 4, "weight_mean": 69, "weight_std": 4,
        "goals_per_game": (0.5, 0.25), "assists_per_game": (0.35, 0.2),
        "saves_per_game": (0, 0), "clean_sheet_p": 0.2,
        "tackles_per_game": (1.2, 0.6), "interceptions_per_game": (0.8, 0.5),
        "pass_accuracy": (82, 6), "shots_per_game": (3.0, 1.0),
    },
    "RW": {
        "height_mean": 177, "height_std": 4, "weight_mean": 69, "weight_std": 4,
        "goals_per_game": (0.5, 0.25), "assists_per_game": (0.35, 0.2),
        "saves_per_game": (0, 0), "clean_sheet_p": 0.2,
        "tackles_per_game": (1.2, 0.6), "interceptions_per_game": (0.8, 0.5),
        "pass_accuracy": (82, 6), "shots_per_game": (3.0, 1.0),
    },
    "ST": {
        "height_mean": 184, "height_std": 5, "weight_mean": 78, "weight_std": 5,
        "goals_per_game": (0.7, 0.3), "assists_per_game": (0.2, 0.15),
        "saves_per_game": (0, 0), "clean_sheet_p": 0.15,
        "tackles_per_game": (0.6, 0.4), "interceptions_per_game": (0.4, 0.3),
        "pass_accuracy": (78, 7), "shots_per_game": (3.5, 1.2),
    },
    "CF": {
        "height_mean": 182, "height_std": 4, "weight_mean": 76, "weight_std": 4,
        "goals_per_game": (0.6, 0.3), "assists_per_game": (0.3, 0.18),
        "saves_per_game": (0, 0), "clean_sheet_p": 0.2,
        "tackles_per_game": (0.8, 0.5), "interceptions_per_game": (0.5, 0.4),
        "pass_accuracy": (80, 6), "shots_per_game": (3.2, 1.0),
    },
}

NATIONALITIES = ["Argentina", "Brazil", "France", "Germany", "Spain", "Italy", "England",
                "Netherlands", "Portugal", "Belgium", "Poland", "Nigeria", "Croatia",
                "Uruguay", "Colombia", "Mexico", "USA", "Japan", "Senegal", "Morocco",
                "Egypt", "South Korea", "Austria", "Switzerland", "Denmark", "Sweden",
                "Norway", "Turkey", "Serbia", "Greece"]


def generate_player_name(nationality: str) -> str:
    """Generate a realistic-sounding player name."""
    if nationality in ("Japan", "South Korea"):
        return fake.name()
    return fake.name()


def generate_value(age: int, perf_score: float, league_tier: float, pos_tier: float) -> float:
    """Generate a realistic market value in EUR.

    Real-world formula approximation:
    - Young + peak perf + top league + top position = mega-money
    - Age penalty after 28, big drop after 32
    """
    # Base value for a "good" player
    base = 5_000_000

    # Age curve: peaks at 26-28
    if age < 22:
        age_mult = 2.5  # wonderkids get hyped
    elif 22 <= age <= 28:
        age_mult = 1.0 + (age - 22) * 0.30
    elif 28 < age <= 31:
        age_mult = 2.5 - (age - 28) * 0.5
    else:
        age_mult = 1.0 - (age - 31) * 0.35
    age_mult = max(0.05, age_mult)

    value = base * age_mult * perf_score * league_tier * pos_tier
    # Add noise
    value *= np.random.uniform(0.6, 1.4)
    # Round nicely
    return round(value / 100_000) * 100_000


def generate_wage(value: float) -> float:
    """Weekly wage, typically 0.05-0.15% of value."""
    pct = np.random.uniform(0.05, 0.15) / 100
    return round(value * pct / 100) * 100


def generate_player(player_id: int, league: str) -> dict:
    """Generate one realistic player record."""
    # Choose position weighted by squad composition
    pos_choices = []
    for pos, info in POSITIONS.items():
        pos_choices.extend([pos] * info["count"])
    position = random.choice(pos_choices)
    profile = POSITION_PROFILES[position]

    # Age: weighted toward 22-30
    age = int(np.clip(np.random.normal(26, 4), 17, 38))

    # Physical
    height = int(np.clip(np.random.normal(profile["height_mean"], profile["height_std"]), 165, 205))
    weight = int(np.clip(np.random.normal(profile["weight_mean"], profile["weight_std"]), 60, 100))

    # Career stats (more appearances for older, less for young)
    base_apps = max(0, int((age - 17) * 12 + np.random.normal(0, 20)))
    appearances = min(base_apps, 500)

    # Performance: log-normal scaled to position
    perf_quality = np.random.lognormal(0, 0.6)  # multiplier

    minutes_played = int(appearances * np.random.uniform(60, 90))

    # Goals from per-game rate
    gpg = np.clip(np.random.normal(*profile["goals_per_game"]) * perf_quality, 0, 1.5)
    goals = int(minutes_played / 90 * gpg)

    # Assists
    apg = np.clip(np.random.normal(*profile["assists_per_game"]) * perf_quality, 0, 1.0)
    assists = int(minutes_played / 90 * apg)

    # Defensive stats
    tpg = np.clip(np.random.normal(*profile["tackles_per_game"]) * perf_quality, 0, None)
    tackles = int(minutes_played / 90 * tpg)

    ipg = np.clip(np.random.normal(*profile["interceptions_per_game"]) * perf_quality, 0, None)
    interceptions = int(minutes_played / 90 * ipg)

    # GK-only stats
    if position == "GK":
        saves = int(minutes_played / 90 * np.random.normal(*profile["saves_per_game"]))
    else:
        saves = 0

    clean_sheets = int(appearances * profile["clean_sheet_p"] * np.random.uniform(0.5, 1.5))
    clean_sheets = min(clean_sheets, appearances)

    # Advanced stats
    pass_accuracy = float(np.clip(np.random.normal(*profile["pass_accuracy"]), 50, 95))
    shots_per_game = float(np.clip(np.random.normal(*profile["shots_per_game"]), 0, 8))

    # Performance score (0-2, used for value calc)
    perf_score = min(2.0, gpg * 5 + apg * 3 + (pass_accuracy - 70) / 20 + np.log1p(appearances) / 5)
    perf_score = max(0.1, perf_score)

    # Value
    value = generate_value(age, perf_score, LEAGUES[league], POSITIONS[position]["tier"])
    wage = generate_wage(value)

    # Contract end
    from datetime import date
    years_left = max(1, 6 - max(0, age - 28))  # older players have shorter contracts
    contract_end = date(2024 + years_left, 6, 30)

    return {
        "id": player_id,
        "name": generate_player_name(random.choice(NATIONALITIES)),
        "age": age,
        "nationality": random.choice(NATIONALITIES),
        "position": position,
        "height": height,
        "weight": weight,
        "appearances": appearances,
        "minutes_played": minutes_played,
        "goals": goals,
        "assists": assists,
        "pass_accuracy": round(pass_accuracy, 1),
        "shots_per_game": round(shots_per_game, 1),
        "tackles": tackles,
        "interceptions": interceptions,
        "saves": saves,
        "clean_sheets": clean_sheets,
        "value": float(value),
        "wage": float(wage),
        "contract_end": contract_end,
        "league": league,
        "team": random.choice(LEAGUE_TEAMS[league]),
        "season": "2024",
    }


def generate_roster(league: str, base_id: int) -> list:
    """Generate a full squad (25 players) for one team."""
    return [generate_player(base_id + i, league) for i in range(25)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500, help="Number of players to generate")
    parser.add_argument("--replace", action="store_true", help="Replace existing data")
    args = parser.parse_args()

    logger.info(f"Initializing database...")
    init_db()

    db = SessionLocal()
    repo = PlayerRepository(db)

    if args.replace:
        logger.info("Wiping existing players...")
        # Note: can't easily call delete all with the repo interface
        from src.data.models import Player
        from src.data.database import Base
        # Use raw SQL via session for efficiency
        db.query(Player).delete()
        db.commit()
        logger.info("Existing players deleted")

    # Pick teams to fill until we hit the count
    n_per_league = args.count // len(LEAGUES)
    logger.info(f"Generating ~{n_per_league} players per league...")

    all_players = []
    player_id = 1

    for league in LEAGUES:
        n_teams = max(1, n_per_league // 25)
        for team_idx in range(n_teams):
            roster = generate_roster(league, player_id)
            all_players.extend(roster)
            player_id += 25
            if len(all_players) >= args.count:
                break
        if len(all_players) >= args.count:
            break

    # Truncate to exact count
    all_players = all_players[: args.count]
    logger.info(f"Generated {len(all_players)} players")

    # Save to DB
    repo.bulk_create(all_players)
    logger.info(f"Inserted {len(all_players)} players into DB")

    # Print summary
    df = pd.DataFrame(all_players)
    print("\n=== Generation Summary ===")
    print(f"Total players: {len(df)}")
    print(f"\nBy league:")
    print(df["league"].value_counts().to_string())
    print(f"\nBy position:")
    print(df["position"].value_counts().to_string())
    print(f"\nValue distribution (EUR):")
    print(f"  Min: {df['value'].min():,.0f}")
    print(f"  Max: {df['value'].max():,.0f}")
    print(f"  Mean: {df['value'].mean():,.0f}")
    print(f"  Median: {df['value'].median():,.0f}")
    print(f"\nAge distribution:")
    print(f"  Min: {df['age'].min()}")
    print(f"  Max: {df['age'].max()}")
    print(f"  Mean: {df['age'].mean():.1f}")
    print(f"\nTop 5 most valuable:")
    top5 = df.nlargest(5, "value")[["name", "position", "age", "league", "team", "value"]]
    for _, row in top5.iterrows():
        print(f"  {row['name']:30} {row['position']:4} age={row['age']:2} {row['league']:15} €{row['value']:,.0f}")


if __name__ == "__main__":
    main()

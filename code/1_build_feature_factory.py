import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'data/PSL.csv'
OUTPUT_PLAYER_STATS = 'data/player_stats.csv'
OUTPUT_VENUE_STATS = 'data/venue_stats.csv'

def build_feature_factory():
    print("--- Starting Phase 1: Feature Factory Construction ---")

    # 1. Load Data
    # ----------------------------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 2. Preprocessing & Standardization
    # ----------------------------------------------------------
    # Normalize column names based on the snippet provided (batter -> striker, etc.)
    # This ensures the code works even if column names vary slightly.
    column_mapping = {
        'batter': 'striker',
        'batsman_runs': 'runs_off_bat',
        'extra_runs': 'extras',
        'inning': 'innings',
        'match_id': 'match_id',
        'winner': 'winner',
        'venue': 'venue'
    }
    df.rename(columns=column_mapping, inplace=True)

    # --- NEW STEP: Standardize Venue Names ---
    # Fixes duplicates found in the data (e.g., "Gaddafi Stadium" vs "Gaddafi Stadium, Lahore")
    venue_mapping = {
        'Gaddafi Stadium, Lahore': 'Gaddafi Stadium',
        'National Stadium, Karachi': 'National Stadium',
        'Sheikh Zayed Stadium, Abu Dhabi': 'Sheikh Zayed Stadium',
        'Sharjah Cricket Stadium': 'Sharjah Cricket Stadium',
        'Dubai International Cricket Stadium': 'Dubai International Cricket Stadium'
    }
    df['venue'] = df['venue'].replace(venue_mapping)
    print(f"Standardized venues. Unique venues after cleanup: {df['venue'].nunique()}")

    # Ensure total_runs exists
    if 'total_runs' not in df.columns:
        df['total_runs'] = df['runs_off_bat'] + df['extras']

    # Filter: Remove "No Result" matches
    # Assuming 'winner' column contains 'No Result' or NaN for abandoned matches
    initial_match_count = df['match_id'].nunique()
    df = df.dropna(subset=['winner']) # Drop rows where winner is NaN
    df = df[df['winner'] != 'No Result']
    print(f"Filtered matches: {initial_match_count} -> {df['match_id'].nunique()} (Removed No Results)")

    # Create Helper Columns for Wides and NoBalls
    # The snippet shows 'extras_type'. We need boolean flags for calculation.
    df['extras_type'] = df['extras_type'].fillna('')
    df['is_wide'] = df['extras_type'].apply(lambda x: 1 if 'wide' in str(x).lower() else 0)
    df['is_noball'] = df['extras_type'].apply(lambda x: 1 if 'no ball' in str(x).lower() or 'nb' in str(x).lower() else 0)

    # 3. Batting Statistics
    # ----------------------------------------------------------
    print("Calculating Batting Stats...")

    # Batting grouping
    batter_grp = df.groupby('striker')

    # A. Total Runs
    bat_runs = batter_grp['runs_off_bat'].sum()

    # B. Balls Faced (Total balls - Wides)
    # Note: No-balls count as a ball faced by the batter in stats, but Wides do not.
    bat_balls = batter_grp['ball'].count() - batter_grp['is_wide'].sum()

    # C. Innings Played
    bat_innings = batter_grp['match_id'].nunique()

    # D. Dismissals
    # We count rows where player_dismissed == striker
    # We must filter the original DF to find where the striker was the one dismissed
    dismissals = df[df['player_dismissed'] == df['striker']].groupby('player_dismissed')['match_id'].count()

    # Combine into a DataFrame
    bat_stats = pd.DataFrame({
        'bat_runs': bat_runs,
        'bat_balls': bat_balls,
        'bat_innings': bat_innings
    })
    bat_stats['dismissals'] = dismissals
    bat_stats['dismissals'] = bat_stats['dismissals'].fillna(0) # Fill 0 if never dismissed

    # Filter: At least 50 balls faced
    bat_stats = bat_stats[bat_stats['bat_balls'] >= 50]

    # Calculate Metrics
    # Average: If dismissals is 0, we can use a proxy (like dividing by 1) or set to specific logic.
    # Standard practice: Average is undefined or equals runs. Here we use max(1, dismissals) to avoid DivByZero.
    bat_stats['psl_batting_avg'] = bat_stats['bat_runs'] / bat_stats['dismissals'].replace(0, 1)
    bat_stats['psl_strike_rate'] = (bat_stats['bat_runs'] / bat_stats['bat_balls']) * 100

    # 4. Bowling Statistics
    # ----------------------------------------------------------
    print("Calculating Bowling Stats...")

    bowler_grp = df.groupby('bowler')

    # A. Runs Conceded
    # Bowlers are charged for runs_off_bat + wides + noballs.
    # They are usually NOT charged for byes/legbyes (which are fielding extras).
    # Logic: If extras_type is 'byes' or 'legbyes', don't count towards bowler.
    def get_bowler_runs(x):
        is_bye_legbye = ('bye' in str(x['extras_type']).lower())
        return 0 if is_bye_legbye else x['total_runs']

    # Apply is slow, so we use vectorization
    # Mask for byes/legbyes
    mask_fielding_extras = df['extras_type'].astype(str).str.contains('bye', case=False)
    # Total runs excluding fielding extras
    df['bowler_run_cost'] = np.where(mask_fielding_extras, 0, df['total_runs'])

    bowl_runs = df.groupby('bowler')['bowler_run_cost'].sum()

    # B. Balls Bowled (Legal Deliveries)
    # Total balls - (Wides + NoBalls)
    bowl_balls = bowler_grp['ball'].count() - (bowler_grp['is_wide'].sum() + bowler_grp['is_noball'].sum())

    # C. Wickets
    # Filter for wickets that count for the bowler (exclude run outs, etc.)
    non_bowler_wickets = ['run out', 'retired hurt', 'retired out', 'obstructing the field', 'timed out']
    wicket_mask = (df['is_wicket'] == True) & (~df['dismissal_kind'].isin(non_bowler_wickets))
    wickets = df[wicket_mask].groupby('bowler')['match_id'].count()

    # Combine
    bowl_stats = pd.DataFrame({
        'bowl_runs': bowl_runs,
        'bowl_balls': bowl_balls
    })
    bowl_stats['wickets'] = wickets
    bowl_stats['wickets'] = bowl_stats['wickets'].fillna(0)

    # Filter: At least 50 balls bowled
    bowl_stats = bowl_stats[bowl_stats['bowl_balls'] >= 50]

    # Calculate Metrics
    bowl_stats['psl_bowling_avg'] = bowl_stats['bowl_runs'] / bowl_stats['wickets'].replace(0, np.nan) # Keep nan if 0 wickets to handle later
    bowl_stats['psl_economy'] = (bowl_stats['bowl_runs'] / bowl_stats['bowl_balls']) * 6

    # 5. Merge Player Stats
    # ----------------------------------------------------------
    print("Merging Batting and Bowling Stats...")
    player_stats = pd.merge(bat_stats, bowl_stats, left_index=True, right_index=True, how='outer')
    player_stats = player_stats.fillna(0)

    # Reset index to make 'player' a column
    player_stats.index.name = 'player'
    player_stats.reset_index(inplace=True)

    # 6. Venue Statistics
    # ----------------------------------------------------------
    print("Calculating Venue Stats (1st Innings Par Scores)...")

    # Filter for Innings 1
    # Note: 'innings' column might be int (1, 2) or string ('1st innings').
    # We assume standardized int based on typical usage, but let's be safe.

    # Helper to convert innings to simple int if needed, though usually it's 1 or 2 in int format
    match_scores = df.groupby(['match_id', 'venue', 'innings'])['total_runs'].sum().reset_index()

    # Filter for 1st innings (usually represented as 1)
    first_innings_scores = match_scores[match_scores['innings'] == 1]

    venue_stats = first_innings_scores.groupby('venue')['total_runs'].mean().reset_index()
    venue_stats.rename(columns={'total_runs': 'venue_avg_1st_innings_score'}, inplace=True)

    # 7. Save Files
    # ----------------------------------------------------------
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    player_stats.to_csv(OUTPUT_PLAYER_STATS, index=False)
    venue_stats.to_csv(OUTPUT_VENUE_STATS, index=False)

    print("--------------------------------------------------")
    print(f"SUCCESS! Files generated in 'data/' folder.")
    print(f"Player Stats Shape: {player_stats.shape}")
    print(f"Venue Stats Shape: {venue_stats.shape}")
    print(f"Sample Player Stats:\n{player_stats.head(3)}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    build_feature_factory()
import pandas as pd
from questdb.ingress import Sender, IngressError, TimestampNanos
from datetime import timedelta
import numpy as np
import requests

questdb_url = "http://localhost:9000/exec"  
questdb_conf = "http::addr=localhost:9000;auto_flush=on;auto_flush_rows=100000;auto_flush_bytes=1048576;"

minutes_before = 60  
minutes_after = 60 

table_name = "XAUUSD_1M_TRAINING_WITH_FEATURES"
data_path = "XAUUSD_M1_RAW.csv"

short_gap_threshold = 120
fill_gap_threshold = 2  

def fetch_market_news():
    query = "SELECT timestamp FROM market_news"
    response = requests.get(questdb_url, params={"query": query})
    response.raise_for_status()
    data = response.json()

    timestamps = []
    for row in data["dataset"]:
        ts = pd.to_datetime(row[0], errors='coerce', utc=True)
        if pd.notna(ts):
            ts = ts.tz_convert('Europe/Sofia')
            ts = ts.tz_localize(None)
        timestamps.append(ts)

    return timestamps

def clear_table(table_name):
    query = f"TRUNCATE TABLE {table_name}"
    response = requests.get(questdb_url, params={"query": query})
    if response.status_code == 200:
        print(f"Cleared table {table_name}.")
    else:
        raise RuntimeError(f"Error clearing table {table_name}: {response.text}")

def insert_updated_data_with_sender(df, table_name):
    try:
        with Sender.from_conf(questdb_conf) as sender:
            for _, row in df.iterrows():
                columns = {
                    col: row[col]
                    for col in df.columns
                    if col != "timestamp" and not pd.isna(row[col])
                }
                sender.row(
                    table_name,
                    columns=columns,
                    at=TimestampNanos(int(row["timestamp"].timestamp() * 1e9))
                )
        print(f"Updated {len(df)} rows in {table_name}.")
    except IngressError as e:
        raise RuntimeError(f"Error updating data in QuestDB: {e}")

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    df['timestamp'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    rename_map = {
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'tick_volume'
    }
    df.rename(columns=rename_map, inplace=True)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']]

def handle_gaps_and_tradable(df, short_gap_threshold=120, fill_gap_threshold=2):
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df['GAPFLAG'] = 0
    df['tradable'] = 1

    gaps = df['timestamp'].diff().dt.total_seconds() / 60
    gaps = gaps.fillna(0)

    very_short_gaps = (gaps > 1) & (gaps <= fill_gap_threshold)
    df.loc[very_short_gaps, 'GAPFLAG'] = 1

    short_gaps = (gaps > fill_gap_threshold) & (gaps <= short_gap_threshold)
    large_gaps = gaps > short_gap_threshold

    def apply_short_gap_logic(idx):
        if idx > 0:
            df.at[idx - 1, 'GAPFLAG'] = 1
            df.at[idx - 1, 'tradable'] = 0

        if idx < len(df) - 1:
            df.at[idx, 'GAPFLAG'] = 1
            df.at[idx, 'tradable'] = 0
            cooldown_end = idx + 60
            if cooldown_end >= len(df):
                cooldown_end = len(df) - 1
            df.loc[idx+1:cooldown_end, 'tradable'] = 0

    for idx in short_gaps[short_gaps].index:
        apply_short_gap_logic(idx)

    for idx in large_gaps[large_gaps].index:
        if idx == 0:
            continue
        prev_close = df.loc[idx - 1, 'close']
        curr_open = df.loc[idx, 'open']
        if not np.isclose(prev_close, curr_open, atol=1e-5):
            apply_short_gap_logic(idx)

    return df

def mark_weekend_non_tradable(df):
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df['day_of_week'] = df['timestamp'].dt.dayofweek  
    df['non_tradable_weekend'] = 0

    for i in range(1, len(df)):
        prev_day = df.loc[i - 1, 'day_of_week']
        current_day = df.loc[i, 'day_of_week']
        if prev_day == 4 and current_day == 0:
            last_friday_idx = i - 1
            first_monday_idx = i
            df.at[last_friday_idx, 'non_tradable_weekend'] = 1
            df.at[first_monday_idx, 'non_tradable_weekend'] = 1

    df.loc[df['non_tradable_weekend'] == 1, 'tradable'] = 0
    df.drop(columns=['day_of_week', 'non_tradable_weekend'], inplace=True)
    return df

def update_tradable_for_news(df, news_timestamps, minutes_before, minutes_after):
    df = df.sort_values("timestamp").reset_index(drop=True)

    for raw_news_time in news_timestamps:
        if pd.isna(raw_news_time):
            continue

        news_time = raw_news_time
        start_time = news_time - timedelta(minutes=minutes_before)
        end_time = news_time + timedelta(minutes=minutes_after)

        df.loc[
            (df["tradable"] == 1) &
            (df["timestamp"] >= start_time) &
            (df["timestamp"] <= end_time),
            "tradable"
        ] = 0

    return df

def add_until_invalid_column(df):
    df = df.sort_values("timestamp").reset_index(drop=True)

    invalid_indices = df.index[df["tradable"] == 0]
    next_invalid_time = pd.Series(index=df.index, dtype=df["timestamp"].dtype)
    next_invalid_time.loc[invalid_indices] = df.loc[invalid_indices, "timestamp"]
    next_invalid_time = next_invalid_time.bfill()

    time_diff = (next_invalid_time - df["timestamp"]).dt.total_seconds() // 60
    df["until_invalid"] = time_diff.fillna(-1).astype(int)

    return df

def main():
    try:
        news_timestamps = fetch_market_news()

        data_1m = load_data_from_csv(data_path)

        data_1m = handle_gaps_and_tradable(data_1m, short_gap_threshold, fill_gap_threshold)

        data_1m = mark_weekend_non_tradable(data_1m)

        data_1m = update_tradable_for_news(
            data_1m, news_timestamps, minutes_before, minutes_after
        )

        data_1m = add_until_invalid_column(data_1m)

        insert_updated_data_with_sender(data_1m, table_name)

        print("All steps completed successfully.")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
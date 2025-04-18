import pandas as pd
from questdb.ingress import Sender, IngressError, TimestampNanos
from datetime import datetime

def process_news_csv(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None)
        df.columns = ['date', 'time', 'currency', 'impact', 'news', 'value1', 'value2', 'value3', 'value4', 'extra']
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M')
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 12, 31)
        df = df[(df['impact'] == 'H') & (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        df = df[['timestamp', 'news']]
        print(f"Processed {len(df)} rows of filtered news data.")
        print("Sample data:\n", df.head())
        return df

    except Exception as e:
        print(f"Error processing news CSV: {e}")
        return pd.DataFrame()

def insert_news_into_questdb(df, table_name):
    conf = "http::addr=localhost:9000;auto_flush=on;auto_flush_rows=100000;auto_flush_bytes=1048576;"
    try:
        with Sender.from_conf(conf) as sender:
            for _, row in df.iterrows():
                sender.row(
                    table_name,
                    columns={'news': row['news']},
                    at=TimestampNanos(int(row['timestamp'].timestamp() * 1e9)) 
                )
            print(f"Inserted {len(df)} rows into QuestDB table: {table_name}.")
    except IngressError as e:
        print(f"Error inserting data into QuestDB: {e}")

def main():
    news_csv_path = "market_news.csv"

    print(f"Processing news data from {news_csv_path}...")
    df_news = process_news_csv(news_csv_path)
    if df_news.empty:
        print("No data to process. Exiting...")
        return

    insert_news_into_questdb(df_news, "market_news")

if __name__ == "__main__":
    main()


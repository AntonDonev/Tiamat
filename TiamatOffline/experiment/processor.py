import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

def process_news_csv(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None)
        df.columns = ['date', 'time', 'currency', 'impact', 'news', 'value1', 'value2', 'value3', 'value4', 'extra']

        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)

        if df['timestamp'].dt.tz is None:
            print("News timestamps presumed UTC. Localizing to UTC...")
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            print("News timestamps have timezone info. Converting to UTC...")
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

        start_date = pd.Timestamp('2013-01-01', tz='UTC')
        current_year = 2025 # Based on current date April 28, 2025
        end_date = pd.Timestamp(f'{current_year}-12-31 23:59:59', tz='UTC')

        df_filtered = df[(df['impact'] == 'H') & (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()

        if df_filtered.empty:
            print("No HIGH impact news found within the specified date range.")
            return pd.Series(dtype='datetime64[ns, UTC]')

        print(f"Found {len(df_filtered)} HIGH impact news events between {start_date.date()} and {end_date.date()}.")

        return df_filtered['timestamp'].drop_duplicates().sort_values().reset_index(drop=True)

    except FileNotFoundError:
        print(f"Error: Could not find the news file at the specified path: {file_path}")
        return pd.Series(dtype='datetime64[ns, UTC]')
    except Exception as e:
        print(f"An unexpected error occurred while processing the news CSV ({file_path}): {e}")
        return pd.Series(dtype='datetime64[ns, UTC]')

def process_market_data_csv(file_path):
    try:
        df = pd.read_csv(file_path, delimiter='\t')
        print(f"Read {len(df)} rows from market data file: {file_path}")
        print("Original market data columns:", df.columns.tolist())

        required_input_cols = ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
        missing_cols = [col for col in required_input_cols if col not in df.columns]
        if missing_cols:
             raise ValueError(f"Market data CSV is missing required columns: {missing_cols}")

        # Only load the essential columns now
        cols_to_load = required_input_cols[:]
        df = df[cols_to_load]

        date_col = '<DATE>'
        time_col = '<TIME>'
        df['datetime_str'] = df[date_col] + ' ' + df[time_col]
        datetime_format = '%Y.%m.%d %H:%M:%S'
        df['timestamp_local'] = pd.to_datetime(df['datetime_str'], format=datetime_format, errors='coerce')

        df.dropna(subset=['timestamp_local'], inplace=True)
        if df.empty:
             print("No valid timestamps found after parsing market data.")
             return pd.DataFrame()

        source_tz_name = 'Europe/Bucharest'
        try:
            source_tz = pytz.timezone(source_tz_name)
            print(f"Assuming market data timestamps are in {source_tz_name} (handles DST).")
        except pytz.UnknownTimeZoneError:
            print(f"Warning: Timezone '{source_tz_name}' not found. Falling back to fixed Etc/GMT-2 (UTC+2).")
            source_tz = pytz.timezone('Etc/GMT-2')

        try:
            df['timestamp'] = df['timestamp_local'].dt.tz_localize(source_tz, ambiguous='infer', nonexistent='shift_forward').dt.tz_convert('UTC')
        except Exception as tz_error:
             print(f"Error during timezone localization/conversion: {tz_error}. Check data around DST changes.")
             df.dropna(subset=['timestamp_local'], inplace=True)
             df['timestamp'] = df['timestamp_local'].dt.tz_localize(source_tz, ambiguous='infer', nonexistent='shift_forward').dt.tz_convert('UTC')

        # Define the exact final columns to keep from the market data itself
        final_cols = ['timestamp', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
        df = df[final_cols].copy()

        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"Processed {len(df)} rows of market data with UTC timestamps.")
        print("Kept columns:", df.columns.tolist())
        print("Sample processed market data (UTC):\n", df.head())
        return df

    except FileNotFoundError:
        print(f"Error: Market data file not found at {file_path}")
        return pd.DataFrame()
    except ValueError as ve:
         print(f"Configuration Error: {ve}")
         return pd.DataFrame()
    except Exception as e:
        print(f"Error processing market data CSV ({file_path}): {e}")
        return pd.DataFrame()


def calculate_until_invalid(market_data_df, news_timestamps_series):
    if news_timestamps_series.empty:
        print("No news timestamps provided, cannot calculate 'until_invalid'.")
        market_data_df['until_invalid'] = pd.NA
        market_data_df['until_invalid'] = market_data_df['until_invalid'].astype('Int64')
        return market_data_df

    market_data_df = market_data_df.sort_values('timestamp').reset_index(drop=True)
    news_timestamps_array = news_timestamps_series.to_numpy()
    market_timestamps_array = market_data_df['timestamp'].to_numpy()

    indices = np.searchsorted(news_timestamps_array, market_timestamps_array, side='right')

    calculated_minutes = []

    print("\nCalculating absolute time (minutes) to CLOSEST news event...")

    for i, market_ts in enumerate(market_timestamps_array):
        idx = indices[i]

        diff_next = pd.NaT
        diff_prev = pd.NaT
        next_news_ts = pd.NaT
        prev_news_ts = pd.NaT

        if idx < len(news_timestamps_array):
            next_news_ts = news_timestamps_array[idx]
            diff_next = next_news_ts - market_ts

        if idx > 0:
            prev_news_ts = news_timestamps_array[idx - 1]
            diff_prev = prev_news_ts - market_ts

        chosen_diff = pd.NaT

        if pd.notna(diff_prev) and pd.notna(diff_next):
            if abs(diff_prev) < abs(diff_next):
                chosen_diff = diff_prev
            else:
                chosen_diff = diff_next
        elif pd.notna(diff_prev):
            chosen_diff = diff_prev
        elif pd.notna(diff_next):
            chosen_diff = diff_next

        if pd.notna(chosen_diff):
            minutes_float = abs(chosen_diff).total_seconds() / 60.0
            calculated_minutes.append(minutes_float)
        else:
            calculated_minutes.append(pd.NA)

    market_data_df['until_invalid'] = pd.Series(calculated_minutes, index=market_data_df.index)

    try:
        print("Converting 'until_invalid' column to Int64 (nullable integer)...")
        print("Notice: This will remove any fractional minutes.")
        market_data_df['until_invalid'] = pd.to_numeric(market_data_df['until_invalid'], errors='coerce')
        market_data_df['until_invalid'] = market_data_df['until_invalid'].astype('Int64')
        print(f"Successfully converted 'until_invalid' to {market_data_df['until_invalid'].dtype}.")
    except Exception as e:
        print(f"\nError converting 'until_invalid' to Int64: {e}. Keeping as float.")
        market_data_df['until_invalid'] = pd.to_numeric(market_data_df['until_invalid'], errors='coerce')

    print("\nSample data with 'until_invalid' (absolute minutes to closest news):")
    print("Head:\n", market_data_df.head())
    print("\nTail:\n", market_data_df.tail())

    return market_data_df


def main():
    news_csv_path = "news.csv"
    market_data_csv_path = "market_data.csv"
    output_csv_path = "ready_market_data.csv"

    print("Starting data processing...")
    # Display current time using the detected Kazanluk timezone (Europe/Sofia)
    print(f"Current local time: {datetime.now(pytz.timezone('Europe/Sofia'))}")

    print(f"\nProcessing news data from: {news_csv_path}")
    news_timestamps = process_news_csv(news_csv_path)
    if news_timestamps.empty:
      print("Warning: No high impact news timestamps were loaded. 'until_invalid' will be empty.")

    print(f"\nProcessing market data from: {market_data_csv_path}")
    df_market = process_market_data_csv(market_data_csv_path)
    if df_market.empty:
        print("Critical Error: No market data processed. Exiting.")
        return

    print("\nCalculating absolute time until/since closest news event...")
    df_market_processed = calculate_until_invalid(df_market, news_timestamps)

    try:
        # Define the exact final columns for the output file
        final_expected_cols = ['timestamp', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', 'until_invalid']

        # Ensure only these columns exist before saving
        cols_to_save = [col for col in final_expected_cols if col in df_market_processed.columns]
        # Check if all expected columns are present after filtering
        if len(cols_to_save) != len(final_expected_cols):
             missing_final_cols = set(final_expected_cols) - set(cols_to_save)
             print(f"Warning: Could not find all expected final columns. Missing: {missing_final_cols}")
             # Proceeding with available columns among the expected ones

        df_to_save = df_market_processed[cols_to_save]

        print(f"\nFinal columns being saved: {df_to_save.columns.tolist()}")
        print(f"Saving processed data to: {output_csv_path}")
        df_to_save.to_csv(output_csv_path, index=False)
        print(f"Successfully saved processed data.")
        print(f"Output column 'until_invalid' type: {df_to_save['until_invalid'].dtype}")

    except KeyError as ke:
        print(f"\nError saving data: Missing expected column(s) during final selection - {ke}")
        print(f"Available columns after processing: {df_market_processed.columns.tolist()}")
    except Exception as e:
        print(f"\nError saving data to CSV ({output_csv_path}): {e}")

    print("\nProcessing finished.")

if __name__ == "__main__":
    main()
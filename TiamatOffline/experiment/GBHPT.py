# --- Imports ---
import pandas as pd
import numpy as np
import ta
# from ta.utils import dropna # dropna is part of ta now
from datetime import datetime, timedelta
import lightgbm as lgb
from backtesting import Backtest, Strategy
# from backtesting.lib import crossover # Not used directly in this strategy logic
from sklearn.model_selection import train_test_split # Could be used, but walk-forward is primary
from sklearn.metrics import accuracy_score, classification_report # For evaluating models
# from geneticalgorithm import geneticalgorithm as ga # REMOVED
import os
import joblib # For saving models
import json # For saving config
# import yfinance as yf # REMOVED
import warnings
import logging
import random # For DEAP
import pickle # For Checkpointing
import uuid # For unique run IDs during evaluation

# --- DEAP Imports ---
from deap import base, creator, tools, algorithms

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # Backtesting.py can be noisy
warnings.filterwarnings('ignore', category=RuntimeWarning) # Ignore potential division by zero etc. in stats if few trades


# --- Global Helper Functions ---

# JSON serializer helper (defined globally)
def default_serializer(obj):
    """Converts numpy/pandas/datetime objects to JSON serializable formats."""
    if isinstance(obj, (np.integer, np.int64)): # Handle numpy integers explicitly
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)): # Handle numpy floats
        # Handle potential NaN/Inf
        if np.isnan(obj): return None # Represent NaN as null
        if np.isinf(obj): return str(obj) # Represent Inf as string 'Infinity' or '-Infinity'
        return float(obj)
    elif isinstance(obj, np.ndarray): # Handle numpy arrays
        return obj.tolist()
    elif isinstance(obj, (datetime, pd.Timestamp)): # Handle datetime/timestamps
        return obj.isoformat()
    elif pd.isna(obj): # Handle pandas NA/NaN
        return None # Represent NaN as null in JSON
    # If none of the above, let the default JSON encoder try or raise error
    try:
        return json.JSONEncoder().default(obj)
    except TypeError:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable and not handled by custom serializer: {obj}")

# --- Configuration ---
DATA_FILE_PATH = 'ready_market_data.csv'
RUNS_DIR = 'deap_runs' # Directory to save results for each evaluation
CHECKPOINT_DIR = 'deap_checkpoints' # Directory for GA state checkpoints
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "ga_checkpoint.pkl")
CHECKPOINT_FREQ = 1 # Save checkpoint every N generations (Not used by eaSimple, but kept for potential manual use)

# --- Initial training uses data up to the END of this year (2013-2017) ---
INITIAL_TRAIN_END_YEAR = 2017

# --- First year to simulate (will use model trained on 2013-2017) ---
SIMULATION_START_YEAR = 2018

# --- Setup Logging ---
# Configure the logging system first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- FIX: Define the logger variable *after* basicConfig and *before* first use ---
logger = logging.getLogger()
# --- END FIX ---


# --- Unique ID for this entire GA execution ---
# Used for organizing run directories if not resuming
ga_execution_id = datetime.now().strftime('%Y%m%d_%H%M%S')


# --- Date settings for determining simulation range ---
# Use current time for dynamic range determination
try:
    # Using current system time as requested by context
    # Update this line based on current execution time if needed
    # CURRENT_DATE = datetime(2025, 5, 4, 12, 42, 52) # From context: Sunday, May 4, 2025 at 12:42:52 PM EEST
    # Using current system time instead:
    CURRENT_DATE = datetime.now()
    logger.info(f"Using system current date/time: {CURRENT_DATE}")
except NameError: # Should not happen now, but keep for safety
    CURRENT_DATE = datetime.now() # Fallback if specific time not available
    logger.warning(f"Using system current date (fallback): {CURRENT_DATE}")

# Convert to UTC Timestamp, trying to handle timezone if possible
try:
    # Attempt to determine local timezone and convert to UTC
    # This might be system-dependent
    local_tz = datetime.now().astimezone().tzinfo
    CURRENT_DATE = pd.Timestamp(CURRENT_DATE, tz=local_tz).tz_convert('UTC')
    logger.info(f"Using UTC Timestamp: {CURRENT_DATE}") # Log the final UTC time
except Exception as tz_err:
    logger.warning(f"Automatic timezone conversion failed: {tz_err}. Assuming current time is naive, localizing to UTC.")
    # Fallback: Treat as naive and localize directly to UTC
    CURRENT_DATE = pd.Timestamp(CURRENT_DATE).tz_localize('UTC')
    logger.info(f"Using assumed UTC Timestamp (fallback): {CURRENT_DATE}")

# --- Earliest data to consider overall ---
FILTER_START_DATE = pd.Timestamp('2013-01-01', tz='UTC')


# --- Hardcoded S&P 500 Yearly Returns (%) ---
# Based on user input for years relevant to the simulation period (2018-2024)
SP500_YEARLY_RETURNS = {
    # 2024: 23.31, # 2024 is not yet complete as of CURRENT_DATE, S&P YTD will vary. Use actuals if needed.
    # Example: As of early May 2024, S&P might be up ~8%. Update as needed.
    2024: 8.0,  # Placeholder - ADJUST THIS BASED ON ACTUAL S&P 500 YTD PERFORMANCE FOR 2024
    2023: 24.23,
    2022: -19.44,
    2021: 26.89,
    2020: 16.26,
    2019: 28.88,
    2018: -6.24,
}
# Adjust the stop year based on S&P data and CURRENT_DATE
SIMULATION_STOP_YEAR = min(CURRENT_DATE.year, max(SP500_YEARLY_RETURNS.keys())) # Use latest S&P year or current year


# --- Strategy & Target Configuration ---
TARGET_HORIZON_MINUTES = 30 # How far ahead to look for TP/SL hits
PIP_DEFINITION = 0.0001 # Example for EURUSD (adjust for your asset!)
PIP_LEVELS = [5,6,7,8,9,10,12, 15, -5,-6,-7,-8,-9, -10,-12, -15] # Bins/Targets

# --- GA Configuration (DEAP) ---
GA_POPULATION_SIZE = 10 # Number of individuals (configs) per generation
GA_GENERATIONS = 5      # Number of generations to run (max, respecting checkpoints)
GA_MUTATION_PROB = 0.2  # Probability for mutating an individual
GA_CROSSOVER_PROB = 0.5 # Probability for crossing over individuals
GA_MUTPB = 0.1          # Probability for mutating each gene within an individual (used in custom mutator)
GA_TOURNSIZE = 3        # Tournament size for selection

# --- Define bounds for LightGBM parameters AND Strategy Parameters --- # << MODIFIED
LGBM_PARAM_BOUNDS = {
    'num_leaves': [15, 60],          # Integer
    'learning_rate': [0.01, 0.1],    # Real
    'feature_fraction': [0.6, 0.95], # Real
    'n_estimators': [50, 200],       # Integer
    'reg_alpha': [0.0, 0.5],         # Real (L1 regularization)
    'reg_lambda': [0.0, 0.5],        # Real (L2 regularization)
    'probability_threshold': [0.60, 0.99], # << NEW: Real (Strategy param)
    'min_rr_ratio': [1.0, 3.0],          # << NEW: Real (Strategy param)
}
LGBM_PARAM_ORDER = list(LGBM_PARAM_BOUNDS.keys()) # Maintain consistent order

# How many indicators to select (min, max) - GA will choose a number in this range
INDICATOR_COUNT_BOUNDS = [10, 100] # Min check applied in evaluation

# --- Backtesting Configuration ---
INITIAL_CASH = 100000
COMMISSION_PERC = 0 # Example commission (0.01%) -> Convert to decimal for backtesting: 0.0001
COMMISSION_DECIMAL = COMMISSION_PERC / 100.0 # Use this in Backtest

# --- Risk Management Configuration --- <<< NEW SECTION
RISK_PERCENTAGE_PER_TRADE = 0.02 # Risk 2% of equity per trade

# --- Other Config ---
INVALID_NEWS_THRESHOLD = 30
NUM_CLASSES = len(PIP_LEVELS) + 1 # +1 for "neither hit"
CLASS_LABELS = {i: pips for i, pips in enumerate(PIP_LEVELS)}
CLASS_LABELS[NUM_CLASSES - 1] = 'Neither'
CLASS_MAP_INV = {v: k for k, v in CLASS_LABELS.items()} # Map pips back to class index


# --- Data Loading ---
def load_data(file_path: str) -> pd.DataFrame:
    """Loads market data, parses timestamp, sets index, renames columns, and checks."""
    # logger is now guaranteed to be defined before this function is called
    logger.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)

        # --- FIX START: Explicitly rename columns based on the CSV header provided ---
        rename_map = {
            '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
            '<TICKVOL>': 'Volume', '<TIMESTAMP>': 'timestamp',
            # Add other potential MT5 names if needed, e.g., '<SPREAD>'
        }
        actual_rename_map = {
            csv_name: expected_name for csv_name, expected_name in rename_map.items() if csv_name in df.columns
        }
        if actual_rename_map:
            logger.info(f"Renaming columns: {actual_rename_map}")
            df.rename(columns=actual_rename_map, inplace=True)
        # --- FIX END ---

        # --- Timestamp Processing ---
        if 'timestamp' not in df.columns: raise ValueError("Missing required 'timestamp' column.")
        # Assume timestamp in CSV is already UTC or handle conversion appropriately here
        # If CSV is naive, localize then convert:
        # df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('YourSourceTimeZone').dt.tz_convert('UTC')
        # If CSV is already UTC-aware string:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # --- OHLCV Column Validation (after renaming) ---
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required OHLCV columns after renaming: {missing}. Check file '{file_path}'.")

        # --- until_invalid Handling ---
        if 'until_invalid' not in df.columns:
            logger.warning("'until_invalid' column not found. Defaulting to allows all trades (99999).")
            df['until_invalid'] = 99999
        else:
            # Ensure it's numeric, handle potential errors
            df['until_invalid'] = pd.to_numeric(df['until_invalid'], errors='coerce')
            # Fill any conversion errors with a large value (no news restriction)
            df['until_invalid'].fillna(99999, inplace=True)


        logger.info(f"Data loaded and columns processed successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        exit()
    except ValueError as ve:
        logger.error(f"Data Loading Error: {ve}")
        exit()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback; logger.error(traceback.format_exc())
        exit()

# --- Feature Engineering ---
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds technical indicators."""
    logger.info("Calculating technical indicators...")
    df = df.copy()
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required OHLCV columns for indicator calculation: {missing}")

    open_col, high_col, low_col, close_col, vol_col = 'Open', 'High', 'Low', 'Close', 'Volume'

    # Convert to numeric, coercing errors - crucial before ta library
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where essential OHLCV data is missing *after* conversion attempt
    rows_before_drop = len(df)
    df.dropna(subset=required_cols, inplace=True)
    if len(df) < rows_before_drop:
        logger.warning(f"Dropped {rows_before_drop - len(df)} rows due to non-numeric OHLCV data.")

    if df.empty:
        logger.error("DataFrame is empty after removing non-numeric OHLCV rows. Cannot calculate indicators.")
        return df # Return empty df

    logger.info(f"Columns available for ta: {df.columns.tolist()}")
    # Use error handling for ta features
    try:
        df = ta.add_all_ta_features(
            df, open=open_col, high=high_col, low=low_col, close=close_col, volume=vol_col, fillna=True
        )
        logger.info("  Finished calculating indicators using ta.add_all_ta_features.")
    except Exception as ta_err:
        logger.error(f"Error during ta.add_all_ta_features: {ta_err}. Some indicators might be missing.")
        # Continue even if some TAs fail, the process below handles NaNs/Infs

    try:
        psar_indicator = ta.trend.PSARIndicator(high=df[high_col], low=df[low_col], close=df[close_col], fillna=True)
        df['psar_up'] = psar_indicator.psar_up()
        df['psar_down'] = psar_indicator.psar_down()
        # Calculate direction based on Close relative to PSAR levels
        df['psar_dir'] = 0 # Default no direction
        # Check if psar_down exists and is valid before comparing
        if 'psar_down' in df.columns and not df['psar_down'].isnull().all():
             df.loc[df[close_col] > df['psar_down'], 'psar_dir'] = 1
        # Check if psar_up exists and is valid before comparing
        if 'psar_up' in df.columns and not df['psar_up'].isnull().all():
             df.loc[df[close_col] < df['psar_up'], 'psar_dir'] = -1

        logger.info("  Added custom PSAR direction.")
    except KeyError as ke:
         logger.warning(f"Could not calculate custom PSAR direction due to missing column: {ke}")
    except Exception as e:
        logger.warning(f"Could not calculate custom PSAR: {e}")

    base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'until_invalid' in df.columns: base_cols.append('until_invalid')
    base_cols_count = len([col for col in base_cols if col in df.columns])
    n_indicator_cols = len(df.columns) - base_cols_count
    logger.info(f"  Calculated {n_indicator_cols} potential indicator features.")
    logger.info("  Cleaning and filling NaNs/Infs...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NaNs - consider ffill first then bfill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    # Maybe a final fill with 0 if any remain? Or drop rows/cols?
    # df.fillna(0, inplace=True) # Use carefully, might distort data

    original_cols = df.shape[1]
    df.dropna(axis=1, how='all', inplace=True) # Drop cols that are *entirely* NaN
    if df.shape[1] < original_cols:
        logger.warning(f"Dropped {original_cols - df.shape[1]} columns containing only NaNs after processing.")

    # Final check for any remaining NaNs in numeric columns (potential issue)
    numeric_cols = df.select_dtypes(include=np.number).columns
    nan_check = df[numeric_cols].isnull().sum()
    if nan_check.sum() > 0:
         logger.warning(f"NaN values still present in some numeric columns after fill attempts: \n{nan_check[nan_check > 0]}")
         # Option: Drop rows with any remaining NaNs in features? df.dropna(subset=numeric_cols, inplace=True)

    logger.info(f"Technical indicators added. DataFrame shape: {df.shape}")
    return df


# --- Target Variable Creation ---
def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the multi-class target variable based on future price movements.
    """
    logger.info(f"Creating target variable (horizon: {TARGET_HORIZON_MINUTES} mins)...")
    df = df.copy()
    target_col = 'target_class'
    df[target_col] = NUM_CLASSES - 1 # Default to "Neither"

    required_cols = ['High', 'Low', 'Close']
    if not all(c in df.columns for c in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        raise ValueError(f"Missing required columns for target creation: {missing}")

    # Ensure required columns are numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    rows_before_drop = len(df)
    df.dropna(subset=required_cols, inplace=True)
    if len(df) < rows_before_drop:
        logger.warning(f"Dropped {rows_before_drop - len(df)} rows due to non-numeric High/Low/Close before target creation.")

    if df.empty:
        logger.error("DataFrame is empty after cleaning High/Low/Close. Cannot create targets.")
        return df


    logger.info("  Preparing lookahead data...")
    # Shift requires sorted index, which should be true from load_data
    for i in range(1, TARGET_HORIZON_MINUTES + 1):
        df[f'High_fwd_{i}'] = df['High'].shift(-i)
        df[f'Low_fwd_{i}'] = df['Low'].shift(-i)

    # Calculate absolute TP/SL levels based on current Close
    # Ensure Close is valid before calculation
    tp_levels_abs = {p: df['Close'] + p * PIP_DEFINITION for p in PIP_LEVELS if p > 0}
    sl_levels_abs = {p: df['Close'] + p * PIP_DEFINITION for p in PIP_LEVELS if p < 0}

    logger.info("  Calculating target classes (vectorized)...")
    # Initialize min_hit_time with a large number (infinity representation)
    min_hit_time = pd.Series(np.iinfo(np.int32).max, index=df.index) # Use max int instead of float inf

    # Iterate through future minutes
    for i in range(1, TARGET_HORIZON_MINUTES + 1):
        high_t = df[f'High_fwd_{i}']
        low_t = df[f'Low_fwd_{i}']
        current_time_step = i

        # Mask for rows where target is not yet resolved OR this timestep is earlier than a previous hit
        mask_unresolved = (df[target_col] == (NUM_CLASSES - 1)) | (current_time_step < min_hit_time)

        # Also mask rows where future high/low is NaN (end of dataset)
        nan_mask = high_t.isna() | low_t.isna()
        mask_unresolved &= ~nan_mask

        if not mask_unresolved.any():
            # logger.debug(f"Target resolution complete at step {i}.")
            break # Stop if all targets resolved

        # Check TP levels (smallest pip target first)
        for pips, tp_target_price_series in sorted(tp_levels_abs.items(), key=lambda item: item[0]):
            if not mask_unresolved.any(): break # Optimization
            class_idx = CLASS_MAP_INV.get(pips) # Use .get for safety
            if class_idx is None: continue # Skip if pips not in map (shouldn't happen)
            # Condition: future high hits TP target & target is still unresolved for this row
            hit_tp_condition = (high_t >= tp_target_price_series) & mask_unresolved
            if hit_tp_condition.any():
                df.loc[hit_tp_condition, target_col] = class_idx
                # Update min_hit_time only for newly resolved rows
                min_hit_time.loc[hit_tp_condition] = np.minimum(min_hit_time.loc[hit_tp_condition], current_time_step)
                # Update mask_unresolved for the next iteration within this time step
                mask_unresolved &= ~hit_tp_condition

        # Check SL levels (smallest absolute pip target first, i.e., closest to current price)
        for pips, sl_target_price_series in sorted(sl_levels_abs.items(), key=lambda item: item[0], reverse=True):
            if not mask_unresolved.any(): break # Optimization
            class_idx = CLASS_MAP_INV.get(pips) # Use .get for safety
            if class_idx is None: continue # Skip if pips not in map
            # Condition: future low hits SL target & target is still unresolved for this row
            hit_sl_condition = (low_t <= sl_target_price_series) & mask_unresolved
            if hit_sl_condition.any():
                df.loc[hit_sl_condition, target_col] = class_idx
                 # Update min_hit_time only for newly resolved rows
                min_hit_time.loc[hit_sl_condition] = np.minimum(min_hit_time.loc[hit_sl_condition], current_time_step)
                # Update mask_unresolved for the next iteration
                mask_unresolved &= ~hit_sl_condition

    # Clean up temporary forward columns
    fwd_cols = [f'High_fwd_{i}' for i in range(1, TARGET_HORIZON_MINUTES + 1)] + \
               [f'Low_fwd_{i}' for i in range(1, TARGET_HORIZON_MINUTES + 1)]
    df.drop(columns=fwd_cols, inplace=True, errors='ignore')

    # Rows where target remained "Neither" and hit time is still max_int haven't hit any target within horizon
    # We often drop these rows as the target is effectively unknown/unmet.
    original_len = len(df)
    # Drop rows where target remained the default 'Neither' OR where Close was NaN initially
    rows_to_drop = df[target_col] == (NUM_CLASSES - 1)
    # Keep track of how many are dropped
    unmet_target_count = rows_to_drop.sum()
    # Drop them
    df = df[~rows_to_drop]
    dropped_count = original_len - len(df)

    if not df.empty:
        df[target_col] = df[target_col].astype(int)
        logger.info(f"Target variable created. Dropped {dropped_count} rows (incl. {unmet_target_count} unmet targets).")
        logger.info(f"Target class distribution:\n{df[target_col].value_counts(normalize=True).sort_index()}")
    else:
        logger.warning("Target variable creation resulted in an empty DataFrame after removing unmet targets.")

    return df

# --- Backtesting Strategy ---
class LGBMStrategy(Strategy):
    # Parameters to be set via bt.run()
    model = None
    feature_names = None
    prob_threshold = None # Probability threshold (set via bt.run())
    min_rr = None       # Minimum R:R ratio (set via bt.run()) << NEW
    pip_levels = PIP_LEVELS
    invalid_news_thresh = INVALID_NEWS_THRESHOLD
    pip_def = PIP_DEFINITION
    class_labels = CLASS_LABELS

    # --- Risk Management --- <<< NEW
    risk_percentage = RISK_PERCENTAGE_PER_TRADE # Use the globally defined risk percentage

    def init(self):
        # --- Input Validation ---
        if self.model is None or self.feature_names is None:
            raise ValueError("Strategy requires 'model' and 'feature_names' parameters.")
        if self.prob_threshold is None or not (0 < self.prob_threshold <= 1):
             raise ValueError(f"Strategy requires a valid 'prob_threshold' parameter (0 < p <= 1). Got: {self.prob_threshold}")
        if self.min_rr is None or self.min_rr <= 0:
            raise ValueError(f"Strategy requires a valid 'min_rr' parameter (positive value). Got: {self.min_rr}")
        if not isinstance(self.feature_names, list) or not self.feature_names:
             raise ValueError("Strategy requires a non-empty list of 'feature_names'.")
        if not all(f in self.data.df.columns for f in self.feature_names):
             missing_feats = [f for f in self.feature_names if f not in self.data.df.columns]
             raise ValueError(f"Strategy data missing required features: {missing_feats}")
        if not (0 < self.risk_percentage < 1):
            raise ValueError(f"Strategy requires a valid 'risk_percentage' between 0 and 1. Got: {self.risk_percentage}")

        # --- Prepare Data References ---
        # Select only the necessary feature columns for faster access during backtest
        self.feature_df = self.data.df[self.feature_names]

        # Prepare news filter data
        if 'until_invalid' in self.data.df.columns:
            # Ensure it's numeric; default to large value if conversion fails
            self.until_invalid = pd.to_numeric(self.data.df['until_invalid'], errors='coerce').fillna(99999)
        else:
            logger.warning("LGBMStrategy: 'until_invalid' column missing. Disabling news filter.")
            # Create a series of large values if the column doesn't exist
            self.until_invalid = pd.Series(99999, index=self.data.df.index)

        # Pre-calculate mapping from pip level to class index for faster lookup
        self.pip_to_index_map = {v: k for k, v in self.class_labels.items() if isinstance(v, (int, float))}


    def next(self):
        # --- Pre-Trade Checks ---
        # If already in a position, do nothing
        if self.position: return

        # Get current index/timestamp robustly
        # Backtesting.py usually makes data available as self.data.Close, self.data.Open etc.
        # The length gives the number of bars processed so far. Index is len - 1.
        if len(self.data.Close) < 1 : return # Not enough data yet
        current_idx = len(self.data.Close) - 1
        # It's safer to access features via iloc on the pre-filtered DataFrame
        # Check if current_idx is valid for feature_df (can lag if features have NaNs)
        if current_idx >= len(self.feature_df):
            # This might happen if feature calculation creates NaNs at the start
            # logger.debug(f"Skipping bar {current_idx}: Index out of bounds for feature_df (len {len(self.feature_df)})")
            return

        current_timestamp = self.data.index[current_idx] # Get timestamp from main data index

        try:
            # News filter check
            # Access until_invalid using the same index logic
            if current_idx < len(self.until_invalid):
                time_since_invalid = self.until_invalid.iloc[current_idx]
                if time_since_invalid <= self.invalid_news_thresh:
                    # logger.debug(f"{current_timestamp}: Skipping trade due to news proximity ({time_since_invalid} <= {self.invalid_news_thresh})")
                    return
            else:
                logger.warning(f"{current_timestamp}: Index {current_idx} out of bounds for until_invalid Series.")
                return # Skip if we can't check the news filter

            # Get current features safely using iloc
            current_features = self.feature_df.iloc[[current_idx]]

            # Check for NaNs in features for the current bar
            if current_features.isnull().values.any():
                # logger.debug(f"{current_timestamp}: Skipping trade due to NaN in features.")
                return

            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(current_features)[0]
                # Ensure probabilities array has expected length
                if len(probabilities) != NUM_CLASSES:
                     logger.warning(f"{current_timestamp}: Model predict_proba returned array of unexpected length {len(probabilities)} (expected {NUM_CLASSES}). Skipping.")
                     return
            else:
                logger.warning(f"{current_timestamp}: Model does not support predict_proba. Skipping.")
                return

            # Get current close price and equity
            current_close = self.data.Close[-1]
            current_equity = self.equity # Get current equity from backtesting framework

            if pd.isna(current_close) or current_close <= 0:
                # logger.debug(f"{current_timestamp}: Skipping trade due to invalid close price ({current_close}).")
                return
            if current_equity <= 0:
                 # logger.debug(f"{current_timestamp}: Skipping trade due to non-positive equity ({current_equity}).")
                 return


        except IndexError:
            # logger.warning(f"{current_timestamp}: Index {current_idx} out of bounds during backtest. Skipping.") # Can be noisy
            return
        except Exception as e:
            logger.error(f"{current_timestamp}: Error during pre-trade checks or prediction: {e}")
            import traceback; logger.debug(f"Traceback: {traceback.format_exc()}")
            return

        # --- Trade Selection Logic (Find best TP/SL based on prob diff) ---
        best_trade = None
        max_prob_diff = -float('inf') # Track the highest (winning_prob - losing_prob)

        # Use the instance's probability threshold and min_rr set during init
        current_prob_threshold = self.prob_threshold
        current_min_rr = self.min_rr

        # Iterate through potential Take Profit levels (positive pips)
        for tp_pips in self.pip_levels:
             if not isinstance(tp_pips, (int, float)) or tp_pips <= 0: continue # Skip non-positive TP levels
             tp_idx = self.pip_to_index_map.get(tp_pips)
             # Check if index is valid for the probabilities array
             if tp_idx is None or not (0 <= tp_idx < len(probabilities)): continue

             # Iterate through potential Stop Loss levels (negative pips)
             for sl_pips in self.pip_levels:
                 if not isinstance(sl_pips, (int, float)) or sl_pips >= 0: continue # Skip non-negative SL levels
                 sl_idx = self.pip_to_index_map.get(sl_pips)
                 # Check if index is valid for the probabilities array
                 if sl_idx is None or not (0 <= sl_idx < len(probabilities)): continue

                 # --- Calculate R:R Ratio ---
                 # Avoid division by zero, although sl_pips should be negative here
                 if sl_pips == 0: continue
                 current_rr = abs(tp_pips / sl_pips)

                 # --- Check Minimum R:R ---
                 if current_rr < current_min_rr: continue

                 # --- Get Probabilities ---
                 prob_tp_hit = probabilities[tp_idx]
                 prob_sl_hit = probabilities[sl_idx]

                 # --- Evaluate Potential BUY Trade ---
                 # Check if winning probability (hitting TP) meets threshold
                 if prob_tp_hit >= current_prob_threshold:
                     prob_diff = prob_tp_hit - prob_sl_hit # Difference: P(Win) - P(Loss)
                     # If this difference is the best found so far...
                     if prob_diff > max_prob_diff:
                          # Calculate exact TP/SL prices
                          tp_price = current_close + tp_pips * self.pip_def
                          sl_price = current_close + sl_pips * self.pip_def
                          # Final check: ensure calculated prices are valid (TP > Close, SL < Close) and not NaN
                          if tp_price > current_close and sl_price < current_close and pd.notna(tp_price) and pd.notna(sl_price):
                              max_prob_diff = prob_diff
                              best_trade = {'type': 'buy', 'tp': tp_price, 'sl': sl_price, 'prob': prob_tp_hit, 'prob_diff': prob_diff}


                 # --- Evaluate Potential SELL Trade ---
                 # Check if winning probability (hitting SL, which is TP for sell) meets threshold
                 if prob_sl_hit >= current_prob_threshold:
                     prob_diff = prob_sl_hit - prob_tp_hit # Difference: P(Win) - P(Loss)
                     # If this difference is the best found so far...
                     if prob_diff > max_prob_diff:
                         # Calculate exact TP/SL prices for sell
                         tp_price = current_close + sl_pips * self.pip_def # TP is lower for sell
                         sl_price = current_close + tp_pips * self.pip_def # SL is higher for sell
                         # Final check: ensure calculated prices are valid (TP < Close, SL > Close) and not NaN
                         if tp_price < current_close and sl_price > current_close and pd.notna(tp_price) and pd.notna(sl_price):
                            max_prob_diff = prob_diff
                            best_trade = {'type': 'sell', 'tp': tp_price, 'sl': sl_price, 'prob': prob_sl_hit, 'prob_diff': prob_diff}


        # --- Execute Trade ---
        # If a best trade was found based on max probability difference...
        if best_trade:
            # logger.debug(f"{current_timestamp}: Best trade found: {best_trade['type']} @ {current_close:.5f}, TP={best_trade['tp']:.5f}, SL={best_trade['sl']:.5f}, ProbDiff={best_trade['prob_diff']:.4f}")

            # --- Calculate Position Size based on Fixed Fractional Risk --- <<< MODIFICATION START
            sl_price = best_trade['sl']
            risk_amount_per_trade = current_equity * self.risk_percentage

            if best_trade['type'] == 'buy':
                stop_loss_distance_price = current_close - sl_price
            else: # Sell trade
                stop_loss_distance_price = sl_price - current_close

            # Ensure stop loss distance is positive (sanity check)
            if stop_loss_distance_price <= 1e-9: # Use small epsilon to avoid division by zero
                logger.warning(f"{current_timestamp}: Skipping {best_trade['type']} trade. Calculated stop loss distance is non-positive ({stop_loss_distance_price:.6f}). SL: {sl_price:.5f}, Close: {current_close:.5f}")
                return

            # Calculate position size
            calculated_size = risk_amount_per_trade / stop_loss_distance_price

            # Ensure size is positive and not NaN/inf
            if not np.isfinite(calculated_size) or calculated_size <= 0:
                 logger.warning(f"{current_timestamp}: Skipping {best_trade['type']} trade due to invalid calculated size: {calculated_size} (RiskAmt: {risk_amount_per_trade:.2f}, SLDist: {stop_loss_distance_price:.6f})")
                 return

            # Optional: Adjust size based on minimum tradeable units if necessary (e.g., round down)
            # size = np.floor(calculated_size) # Example if only whole units allowed
            size = calculated_size # Use the calculated size directly for now

            # logger.debug(f"{current_timestamp}: Calculated Size: {size:.4f} for {best_trade['type']} (Equity: {current_equity:.2f}, RiskAmt: {risk_amount_per_trade:.2f}, SLDist: {stop_loss_distance_price:.6f})")
            # --- MODIFICATION END ---

            # --- Execute with calculated size ---
            if best_trade['type'] == 'buy':
                 # Final check for NaN just before execution (should be redundant but safe)
                 if pd.notna(best_trade['sl']) and pd.notna(best_trade['tp']):
                     self.buy(sl=best_trade['sl'], tp=best_trade['tp'], size=size) # Pass calculated size
                 else: logger.warning(f"{current_timestamp}: Skipping BUY trade due to NaN SL/TP detected just before execution.")
            elif best_trade['type'] == 'sell':
                 if pd.notna(best_trade['sl']) and pd.notna(best_trade['tp']):
                     self.sell(sl=best_trade['sl'], tp=best_trade['tp'], size=size) # Pass calculated size
                 else: logger.warning(f"{current_timestamp}: Skipping SELL trade due to NaN SL/TP detected just before execution.")


# --- Fitness Function for DEAP ---
def evaluate_individual(individual: list, *, all_indicator_cols: list, full_data: pd.DataFrame) -> tuple:
    """
    Fitness function for DEAP. Decodes individual, trains, simulates, evaluates.
    Includes fail-fast if a year has 0 trades. Returns a tuple: (fitness_score,).
    """
    instance_start_time = datetime.now()
    # Create a unique ID for this specific evaluation run
    eval_run_id = f"deap_{ga_execution_id}_eval_{uuid.uuid4().hex[:10]}"
    run_dir = os.path.join(RUNS_DIR, eval_run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"--- Starting DEAP Evaluation: {eval_run_id} ---")

    # --- 1. Decode DEAP Individual ---
    try:
        num_indicators = len(all_indicator_cols)
        num_lgbm_and_strategy_params = len(LGBM_PARAM_ORDER)
        expected_len = num_indicators + num_lgbm_and_strategy_params
        if len(individual) != expected_len:
             logger.error(f"Run {eval_run_id}: Individual length mismatch. Got {len(individual)}, expected {expected_len}. Assigning poor fitness.")
             # Ensure results.txt exists for failed runs if possible
             with open(os.path.join(run_dir, 'results.txt'), 'w') as f: f.write("FAILED: Individual length mismatch.\n")
             return (-999999,) # Use a distinct very poor fitness score

        # --- Decode Indicator Selection ---
        indicator_mask = [bool(gene) for gene in individual[:num_indicators]]
        selected_indicator_cols = [col for i, col in enumerate(all_indicator_cols) if indicator_mask[i]]

        # Check if enough indicators were selected
        if len(selected_indicator_cols) < INDICATOR_COUNT_BOUNDS[0]:
            logger.warning(f"Run {eval_run_id}: Too few indicators selected ({len(selected_indicator_cols)} < {INDICATOR_COUNT_BOUNDS[0]}). Assigning poor fitness.")
            with open(os.path.join(run_dir, 'results.txt'), 'w') as f: f.write("FAILED: Too few indicators selected.\n")
            return (-999998,) # Assign poor fitness

        # --- Decode LGBM and Strategy Parameters ---
        lgbm_and_strategy_vals = individual[num_indicators:]
        lgbm_config = { # Base LGBM settings
            'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': NUM_CLASSES,
            'boosting_type': 'gbdt', 'n_jobs': -1, 'verbose': -1, 'seed': 42, 'verbosity': -1,
        }
        decoded_prob_threshold = None
        decoded_min_rr_ratio = None

        # Iterate through the expected parameters based on LGBM_PARAM_ORDER
        for i, param_name in enumerate(LGBM_PARAM_ORDER):
            # Check bounds definition first
            if param_name not in LGBM_PARAM_BOUNDS or len(LGBM_PARAM_BOUNDS[param_name]) != 2:
                 logger.error(f"Run {eval_run_id}: Invalid bounds defined for parameter '{param_name}'. Skipping decoding for this param.")
                 continue # Skip this parameter if bounds are missing/invalid

            bounds = LGBM_PARAM_BOUNDS[param_name]
            lower_bound, upper_bound = bounds[0], bounds[1]

            # Handle potential errors from GA (e.g., NaN, Inf from mutations)
            raw_val = lgbm_and_strategy_vals[i]
            if pd.isna(raw_val) or np.isinf(raw_val):
                 logger.warning(f"Run {eval_run_id}: Invalid raw value '{raw_val}' for param '{param_name}'. Using lower bound '{lower_bound}'.")
                 val = lower_bound # Use lower bound as a fallback
            # Decode based on type (int or float) and clamp to bounds
            elif isinstance(lower_bound, int):
                 # Ensure bounds are logical before rounding/clamping
                 if lower_bound > upper_bound: lower_bound, upper_bound = upper_bound, lower_bound # Swap if order wrong
                 val = max(lower_bound, min(upper_bound, int(round(raw_val))))
            else: # Assume float/real
                 # Ensure bounds are logical before clamping
                 if lower_bound > upper_bound: lower_bound, upper_bound = upper_bound, lower_bound # Swap if order wrong
                 val = max(lower_bound, min(upper_bound, float(raw_val)))

            # Assign decoded value to the correct variable/dictionary
            if param_name == 'probability_threshold':
                decoded_prob_threshold = val
            elif param_name == 'min_rr_ratio':
                decoded_min_rr_ratio = val
            else: # It's an LGBM parameter
                lgbm_config[param_name] = val

        # --- Final Check: Ensure critical strategy params were decoded ---
        if decoded_prob_threshold is None or decoded_min_rr_ratio is None:
             missing_params = []
             if decoded_prob_threshold is None: missing_params.append('probability_threshold')
             if decoded_min_rr_ratio is None: missing_params.append('min_rr_ratio')
             logger.error(f"Run {eval_run_id}: Failed to decode required strategy parameters: {missing_params}. Aborting with poor fitness.")
             with open(os.path.join(run_dir, 'results.txt'), 'w') as f: f.write(f"FAILED: Missing decoded strategy parameters: {missing_params}.\n")
             return (-999997,) # Unique code for critical param decode failure

        logger.info(f"Run {eval_run_id}: Decoded Params: Indicators={len(selected_indicator_cols)}, ProbThresh={decoded_prob_threshold:.4f}, MinRR={decoded_min_rr_ratio:.2f}")
        # logger.debug(f"Run {eval_run_id}: LGBM Config: {lgbm_config}") # Optional debug
        feature_cols = selected_indicator_cols
        target_col = 'target_class'
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'until_invalid'] # Ensure until_invalid is included

    except Exception as e:
        logger.error(f"Run {eval_run_id}: Error decoding DEAP individual: {e}")
        import traceback; logger.debug(f"Decoding traceback: {traceback.format_exc()}")
        with open(os.path.join(run_dir, 'results.txt'), 'w') as f: f.write(f"FAILED: Error decoding individual: {e}.\n")
        return (-999996,) # Unique code for general decoding exception

    # --- Save Configuration for this run ---
    config_to_save = {
        'run_id': eval_run_id,
        'ga_execution_id': ga_execution_id, # Link back to the main GA run
        'selected_indicators': selected_indicator_cols,
        'lightgbm_params': lgbm_config,
        'probability_threshold': decoded_prob_threshold,
        'min_rr_ratio': decoded_min_rr_ratio,
        'risk_percentage_per_trade': RISK_PERCENTAGE_PER_TRADE, # <<< ADDED
        'target_horizon_minutes': TARGET_HORIZON_MINUTES,
        'pip_levels': PIP_LEVELS,
        'invalid_news_threshold': INVALID_NEWS_THRESHOLD,
        'pip_definition': PIP_DEFINITION,
        'num_classes': NUM_CLASSES,
        'indicator_count_bounds': INDICATOR_COUNT_BOUNDS,
        'lgbm_param_bounds': LGBM_PARAM_BOUNDS,
        'initial_train_end_year': INITIAL_TRAIN_END_YEAR,
        'simulation_start_year': SIMULATION_START_YEAR,
    }
    try:
        # Use the globally defined serializer
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config_to_save, f, indent=4, default=default_serializer)
    except Exception as e:
        logger.error(f"Run {eval_run_id}: Failed to save config.json: {e}")
        # Continue evaluation even if config saving fails

    # --- 2. Prepare Data Subset ---
    # Ensure all necessary columns are present before selection
    cols_required_in_full_data = set(base_cols + feature_cols + [target_col])
    missing_in_source = [c for c in cols_required_in_full_data if c not in full_data.columns]
    if missing_in_source:
        logger.error(f"Run {eval_run_id}: Fatal Error - Required columns missing from source full_data: {missing_in_source}. Assigning poor fitness.")
        with open(os.path.join(run_dir, 'results.txt'), 'a') as f: f.write(f"FAILED: Missing required columns in source data: {missing_in_source}.\n")
        return (-999995,)

    # Select only the columns needed for this run
    cols_to_keep = sorted(list(cols_required_in_full_data))
    try:
        data = full_data[cols_to_keep].copy()
        initial_rows = len(data)
        # Clean Infinite values first
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop rows with NaN in features or target column - crucial for ML
        data.dropna(subset=feature_cols + [target_col], inplace=True)
        rows_after_na = len(data)
        if initial_rows != rows_after_na:
            logger.debug(f"Run {eval_run_id}: Dropped {initial_rows - rows_after_na} rows due to NaNs/Infs in selected features/target.")
        if data.empty:
            logger.error(f"Run {eval_run_id}: Data is empty after selecting features/target and dropping NaNs. Assigning poor fitness.")
            with open(os.path.join(run_dir, 'results.txt'), 'a') as f: f.write("FAILED: Data empty after feature/target selection and NaN drop.\n")
            return (-999994,) # Data prep error code
    except KeyError as e:
       logger.error(f"Run {eval_run_id}: KeyError during data subset preparation: {e}. Assigning poor fitness.")
       with open(os.path.join(run_dir, 'results.txt'), 'a') as f: f.write(f"FAILED: KeyError during data prep: {e}.\n")
       return (-999993,)
    except Exception as e:
        logger.error(f"Run {eval_run_id}: Error preparing data subset: {e}. Assigning poor fitness.")
        with open(os.path.join(run_dir, 'results.txt'), 'a') as f: f.write(f"FAILED: Error during data prep: {e}.\n")
        return (-999992,)

    # --- 3. Initial Training ---
    initial_train_end_ts = pd.Timestamp(f'{INITIAL_TRAIN_END_YEAR + 1}-01-01', tz='UTC')
    train_data = data[data.index < initial_train_end_ts]

    # Define minimum training samples based on config
    min_train_samples = max(100, lgbm_config.get('n_estimators', 50) * 3) # Adjusted min samples
    if train_data.empty or len(train_data) < min_train_samples:
        logger.error(f"Run {eval_run_id}: Insufficient data ({len(train_data)} < {min_train_samples}) for initial training (before {initial_train_end_ts.date()}). Assigning poor fitness.")
        with open(os.path.join(run_dir, 'results.txt'), 'a') as f: f.write("FAILED: Insufficient initial training data.\n")
        return (-999991,) # Training error code

    logger.info(f"Run {eval_run_id}: Initial Training on {len(train_data)} rows (Features: {len(feature_cols)}) until {initial_train_end_ts.date()}...")
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]

    # Final check for NaNs/Infs before fitting (should be redundant)
    if X_train.isnull().values.any() or np.isinf(X_train.values).any() or y_train.isnull().values.any():
        logger.error(f"Run {eval_run_id}: NaNs/Infs detected in X_train/y_train just before initial fitting. Aborting.")
        with open(os.path.join(run_dir, 'results.txt'), 'a') as f: f.write("FAILED: NaNs/Infs in training data before fit.\n")
        return (-999990,) # Training error code

    try:
        lgbm_model = lgb.LGBMClassifier(**lgbm_config)
        lgbm_model.fit(X_train, y_train)
        logger.info(f"Run {eval_run_id}: Initial training complete.")
    except Exception as e:
        logger.error(f"Run {eval_run_id}: Initial training failed: {e}. Assigning poor fitness.")
        with open(os.path.join(run_dir, 'results.txt'), 'a') as f: f.write(f"FAILED: Initial training error: {e}\n")
        return (-999989,) # Training error code

    # --- 4. Simulation and Retraining Loop ---
    current_model = lgbm_model
    yearly_results = {}
    all_trades = pd.DataFrame()
    total_valid_simulation_years = 0 # Count years with successful backtest

    # --- Determine Simulation Date Range ---
    sim_data_available = data[data.index >= initial_train_end_ts]
    if sim_data_available.empty:
        logger.warning(f"Run {eval_run_id}: No data available for simulation period (after {initial_train_end_ts.date()}). Assigning default poor fitness.")
        with open(os.path.join(run_dir, 'results.txt'), 'a') as f: f.write("\nFAILED: No simulation data available after initial training period.\n")
        return (-999988,) # Simulation error code

    # Actual start year is the later of configured start or first available data year
    actual_simulation_start_year = max(SIMULATION_START_YEAR, sim_data_available.index.min().year)
    # Use the globally determined SIMULATION_STOP_YEAR
    actual_simulation_end_year = min(CURRENT_DATE.year, SIMULATION_STOP_YEAR)


    if actual_simulation_start_year > actual_simulation_end_year:
         logger.warning(f"Run {eval_run_id}: Calculated simulation start year ({actual_simulation_start_year}) is after end year ({actual_simulation_end_year}). No simulation possible. Assigning poor fitness.")
         with open(os.path.join(run_dir, 'results.txt'), 'a') as f: f.write("\nWARNING: No simulation years possible based on date ranges.\n")
         return (-999987,)

    logger.info(f"Run {eval_run_id}: Simulation loop will run from {actual_simulation_start_year} up to and including year: {actual_simulation_end_year}")

    # --- Yearly Simulation Loop ---
    for year in range(actual_simulation_start_year, actual_simulation_end_year + 1):
        loop_year_start_time = datetime.now()
        logger.info(f"="*10 + f" Run {eval_run_id} Simulating Year: {year} " + "="*10)

        # Define time boundaries for this simulation year slice
        sim_year_start_ts = max(pd.Timestamp(f'{year}-01-01', tz='UTC'), sim_data_available.index.min())
        sim_year_end_ts = pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
        # Ensure end timestamp doesn't exceed the absolute max timestamp in the data
        sim_year_end_ts = min(sim_year_end_ts, data.index.max() + timedelta(microseconds=1)) # Include the last bar if needed

        # Get data slice for this year
        simulation_data_slice = data[(data.index >= sim_year_start_ts) & (data.index < sim_year_end_ts)].copy()

        # --- Checks before backtesting the year ---
        if simulation_data_slice.empty or sim_year_start_ts >= sim_year_end_ts:
            logger.warning(f"Run {eval_run_id}, Year {year}: No data available for simulation in this period. Skipping year.")
            yearly_results[year] = None # Mark year as failed/skipped
            continue

        # Check for NaNs/Infs in the features for this slice
        if simulation_data_slice[feature_cols].isnull().values.any() or np.isinf(simulation_data_slice[feature_cols].values).any():
            logger.error(f"Run {eval_run_id}, Year {year}: NaNs/Infs detected in feature data slice before backtest. Skipping year.")
            yearly_results[year] = None
            continue

        logger.info(f"Run {eval_run_id}: Backtesting {year} ({simulation_data_slice.index.min().date()} to {simulation_data_slice.index.max().date()}, {len(simulation_data_slice)} rows)...")

        # --- Prepare Backtest Parameters ---
        # Pass the decoded strategy parameters specific to this individual
        bt_params = {
            'model': current_model,
            'feature_names': feature_cols,
            'pip_levels': PIP_LEVELS,
            'min_rr': decoded_min_rr_ratio,
            'prob_threshold': decoded_prob_threshold,
            'invalid_news_thresh': INVALID_NEWS_THRESHOLD,
            'pip_def': PIP_DEFINITION,
            'class_labels': CLASS_LABELS,
            # Risk percentage is now handled internally by the strategy class attribute
        }

        # --- Execute Backtest ---
        try:
            # Ensure unique index - critical for backtesting framework
            if not simulation_data_slice.index.is_unique:
                logger.warning(f"Run {eval_run_id}, Year {year}: Deduplicating index before backtest.")
                simulation_data_slice = simulation_data_slice[~simulation_data_slice.index.duplicated(keep='first')]

            if simulation_data_slice.empty:
                 logger.error(f"Run {eval_run_id}, Year {year}: Data slice empty after deduplication. Skipping year.")
                 yearly_results[year] = None; continue

            # Initialize Backtest instance (using COMMISSION_DECIMAL)
            bt = Backtest(simulation_data_slice, LGBMStrategy, cash=INITIAL_CASH, commission=COMMISSION_DECIMAL, margin=0.01,
                          exclusive_orders=True, trade_on_close=False, hedging=False) # trade_on_close=False is typical for M1

            # Run the simulation for this year
            stats = bt.run(**bt_params) # Pass the decoded parameters

            # --- FAIL FAST CHECK: Abort if 0 trades executed in the year ---
            num_trades_year = stats.get('# Trades', 0) # Default to 0 if key missing
            if num_trades_year == 0:
                logger.error(f"Run {eval_run_id}, Year {year}: Evaluation aborted - 0 trades executed. Assigning poor fitness.")
                results_file_path = os.path.join(run_dir, 'results.txt')
                mode = 'a' if os.path.exists(results_file_path) else 'w'
                with open(results_file_path, mode) as f:
                     f.write(f"\nFAILED: Run aborted due to 0 trades in year {year}.\n")
                return (-999986,) # Return distinct poor fitness for zero trades
            # --- End Fail Fast Check ---

            # Store results and trades
            yearly_results[year] = stats
            total_valid_simulation_years += 1 # Increment count of successful years
            trades_year = stats['_trades']
            if not trades_year.empty:
                trades_year['Year'] = year
                all_trades = pd.concat([all_trades, trades_year], ignore_index=True)

            # Attempt to plot results
            plot_file = os.path.join(run_dir, f'backtest_{year}.html')
            try:
                 # Only plot if there are trades and stats are available
                 if stats is not None and stats.get('# Trades', 0) > 0:
                     bt.plot(filename=plot_file, open_browser=False)
                 # else: logger.info(f"Skipping plot for {year} (no trades).") # Handled by 0 trades check
            except ImportError:
                 logger.error(f"Run {eval_run_id}, Year {year}: Plotting failed (ImportError). Install 'bokeh'.")
            except Exception as plot_err:
                logger.error(f"Run {eval_run_id}, Year {year}: Backtest plot failed: {plot_err}")

            # Log yearly summary
            logger.info(f"Run {eval_run_id}: Backtest {year} Stats: Return={stats.get('Return [%]', 'N/A'):.2f}%, WinRate={stats.get('Win Rate [%]', 'N/A'):.2f}%, #Trades={stats.get('# Trades', 'N/A')}")

        except Exception as e:
            logger.error(f"Run {eval_run_id}, Year {year}: Backtesting failed: {e}")
            import traceback; logger.debug(f"Backtesting traceback: {traceback.format_exc()}") # Debug level for traceback
            yearly_results[year] = None # Mark year as failed


        # --- Yearly Retraining (if not the last simulation year) ---
        if year < actual_simulation_end_year:
            retrain_end_ts = min(pd.Timestamp(f'{year + 1}-01-01', tz='UTC'), data.index.max() + timedelta(microseconds=1))
            # Use data *up to* the end of the year just simulated
            retrain_data = data[data.index < retrain_end_ts].copy()

            if retrain_data.empty or len(retrain_data) < min_train_samples:
                logger.warning(f"Run {eval_run_id}: Insufficient data ({len(retrain_data)} < {min_train_samples}) for retraining up to {retrain_end_ts.date()}. Skipping retraining, using previous model.")
            else:
                logger.info(f"Run {eval_run_id}: Retraining model using {len(retrain_data)} rows (up to {retrain_end_ts.date()})...")
                X_retrain = retrain_data[feature_cols]
                y_retrain = retrain_data[target_col]

                # Final check before retraining
                if X_retrain.isnull().values.any() or np.isinf(X_retrain.values).any() or y_retrain.isnull().values.any():
                    logger.error(f"Run {eval_run_id}: NaNs/Infs detected before retraining for year {year}. Skipping retraining.")
                    # Continue with the existing model (current_model is not updated)
                else:
                    try:
                        # Create a new model instance for retraining
                        lgbm_model_retrained = lgb.LGBMClassifier(**lgbm_config)
                        lgbm_model_retrained.fit(X_retrain, y_retrain)
                        current_model = lgbm_model_retrained # Update the model for the next year
                        logger.info(f"Run {eval_run_id}: Yearly retraining complete for year {year+1}.")
                    except Exception as e:
                        logger.error(f"Run {eval_run_id}: Yearly retraining failed for year {year+1}: {e}. Continuing with previous model.")
                        # Keep using the non-retrained model

        logger.debug(f"Run {eval_run_id}: Year {year} processing took: {datetime.now() - loop_year_start_time}")

    # --- End Yearly Simulation Loop ---


    # --- 5. Final Training (Optional: train on all available data) ---
    logger.info(f"Run {eval_run_id}: Attempting Final Training...")
    final_train_data = data.copy() # Use all cleaned data for final model
    final_model = None
    if not final_train_data.empty:
        if len(final_train_data) >= min_train_samples:
            logger.info(f"Run {eval_run_id}: Final Training on {len(final_train_data)} rows ({final_train_data.index.min().date()} to {final_train_data.index.max().date()})...")
            X_final = final_train_data[feature_cols]
            y_final = final_train_data[target_col]

            # Final check
            if X_final.isnull().values.any() or np.isinf(X_final.values).any() or y_final.isnull().values.any():
                 logger.error(f"Run {eval_run_id}: NaNs or Infs detected before final training. Skipping.")
            else:
                try:
                    final_model = lgb.LGBMClassifier(**lgbm_config)
                    final_model.fit(X_final, y_final)
                    model_path = os.path.join(run_dir, 'final_model.joblib')
                    joblib.dump(final_model, model_path)
                    logger.info(f"Run {eval_run_id}: Final model trained and saved to {model_path}")
                except Exception as e:
                    logger.error(f"Run {eval_run_id}: Final model training failed: {e}.")
                    # Append failure message to results file
                    with open(os.path.join(run_dir, 'results.txt'), 'a') as f: f.write(f"\nFAILED: Final model training error: {e}\n")
        else:
             logger.warning(f"Run {eval_run_id}: Skipping final training: insufficient data ({len(final_train_data)} < {min_train_samples}).")
    else:
         logger.warning(f"Run {eval_run_id}: Skipping final training: prepared data was empty.")


    # --- 6. Evaluate Overall Performance & Calculate Fitness ---
    if not yearly_results or total_valid_simulation_years == 0:
        # Check if the failure was due to 0 trades earlier
        results_file_path = os.path.join(run_dir, 'results.txt')
        zero_trade_fail = False
        if os.path.exists(results_file_path):
             try:
                  with open(results_file_path, 'r') as f:
                       if "0 trades" in f.read():
                           zero_trade_fail = True
             except Exception as read_err: logger.warning(f"Could not read results file {results_file_path} to check for 0 trades: {read_err}")

        # If not the 0-trade failure, log generic failure
        if not zero_trade_fail:
            logger.error(f"Run {eval_run_id}: No valid yearly results generated ({total_valid_simulation_years} successful years). Evaluation failed. Assigning poor fitness.")
            mode = 'a' if os.path.exists(results_file_path) else 'w'
            try:
                with open(results_file_path, mode) as f: f.write("\nFAILED: No valid yearly simulation results generated.\n")
            except Exception as write_err: logger.error(f"Failed to write failure message to results file: {write_err}")
        # Return a specific code if no valid years, different from 0 trades code
        return (-999985,) if not zero_trade_fail else (-999986,) # Keep 0 trades code distinct

    # --- Aggregate results from successful years ---
    total_return = 1.0; years_beat_sp500 = 0; total_trades = 0
    results_summary = []; sharpe_ratios = []; sortino_ratios = []
    max_drawdowns = [] ; win_rates = []

    # Iterate through the years that were *expected* to run
    expected_sim_years = list(range(actual_simulation_start_year, actual_simulation_end_year + 1))
    total_expected_sim_years = len(expected_sim_years)

    for year in expected_sim_years:
        stats = yearly_results.get(year) # Get stats if year ran, else None

        # Check if the year ran successfully and produced stats
        if stats is None or not isinstance(stats, pd.Series) or stats.empty:
            results_summary.append(f"Year {year}: FAILED or SKIPPED (No Stats)")
            continue # Skip years that failed or were skipped

        # --- Process stats for valid years ---
        return_pct = stats.get('Return [%]', 0.0)
        num_trades = stats.get('# Trades', 0)
        win_rate = stats.get('Win Rate [%]', 0.0)
        sharpe = stats.get('Sharpe Ratio', np.nan)
        sortino = stats.get('Sortino Ratio', np.nan)
        max_dd = stats.get('Max. Drawdown [%]', np.nan) # Usually negative

        # Handle NaN/inf values from stats - replace with 0 for aggregation? Or skip?
        # For Sharpe/Sortino, 0 seems reasonable if undefined (e.g., 0 trades or 0 std dev return)
        sharpe = 0.0 if pd.isna(sharpe) or not np.isfinite(sharpe) else sharpe
        sortino = 0.0 if pd.isna(sortino) or not np.isfinite(sortino) else sortino
        # Max Drawdown NaN could mean no loss occurred. Treat as 0?
        max_dd = 0.0 if pd.isna(max_dd) else max_dd # Keep it negative/zero

        # Aggregate metrics
        sharpe_ratios.append(sharpe)
        sortino_ratios.append(sortino)
        max_drawdowns.append(max_dd)
        win_rates.append(win_rate)
        total_trades += num_trades

        # Calculate cumulative return
        if pd.notna(return_pct):
            total_return *= (1 + return_pct / 100.0)
        else:
            logger.warning(f"Run {eval_run_id}, Year {year}: Return [%] is NaN. Treating as 0% for cumulative calculation.")
            return_pct = 0.0 # Use 0 for display if NaN

        # Compare vs S&P 500 for this year
        sp_return = SP500_YEARLY_RETURNS.get(year, np.nan)
        beat_sp_str = "N/A (S&P data missing)"
        beat_sp_bool = False
        if not pd.isna(sp_return):
            if return_pct > sp_return:
                years_beat_sp500 += 1
                beat_sp_str = f"Yes (SP500: {sp_return:.2f}%)"
                beat_sp_bool = True
            else:
                beat_sp_str = f"No (SP500: {sp_return:.2f}%)"
                beat_sp_bool = False
        elif year in SP500_YEARLY_RETURNS: # Handle case where year exists but value is NaN
            logger.warning(f"Run {eval_run_id}: S&P 500 return is NaN for year {year} in SP500_YEARLY_RETURNS dict.")
            beat_sp_str = "N/A (S&P data is NaN)"


        # Format summary line for this year
        summary_line = (f"Year {year}: Return={return_pct:.2f}%, "
                        f"WinRate={win_rate:.2f}%, Trades={num_trades}, "
                        f"Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, "
                        f"MaxDD={max_dd:.2f}%, Beat S&P500={beat_sp_str}")
        results_summary.append(summary_line)

    # --- Calculate Overall Aggregated Metrics ---
    # Only average over years that actually produced valid stats
    num_valid_years_aggregated = len(sharpe_ratios) # Should equal total_valid_simulation_years

    if num_valid_years_aggregated > 0:
        overall_return_pct = (total_return - 1) * 100
        avg_trades_per_year = total_trades / num_valid_years_aggregated
        avg_sharpe = np.mean(sharpe_ratios)
        avg_sortino = np.mean(sortino_ratios)
        avg_win_rate = np.mean(win_rates)
        # Average Max Drawdown (less meaningful than overall max DD, but can indicate consistency)
        avg_max_dd = np.mean(max_drawdowns)
        # Overall Max Drawdown would require combining equity curves - Backtesting lib doesn't easily provide this across runs
        # We'll use average max drawdown and number of failed years as proxies
    else: # Should not happen if we reach here due to earlier checks, but safety first
        logger.error(f"Run {eval_run_id}: Reached final aggregation with 0 valid years. This indicates a logic error.")
        overall_return_pct = 0.0
        avg_trades_per_year = 0.0
        avg_sharpe = 0.0
        avg_sortino = 0.0
        avg_win_rate = 0.0
        avg_max_dd = 0.0


    # --- Define Success Criteria ---
    # Criteria 1: All expected simulation years must have run successfully without critical errors
    all_expected_years_succeeded = (num_valid_years_aggregated == total_expected_sim_years) if total_expected_sim_years > 0 else False
    # Criteria 2: Beat S&P 500 threshold (e.g., > 50% of the simulated years)
    success_threshold_fraction = 0.5
    # Calculate required number of S&P beats based on *successfully simulated* years
    beat_sp_required_count = np.ceil(success_threshold_fraction * num_valid_years_aggregated) if num_valid_years_aggregated > 0 else 0
    beat_sp_threshold_met = (years_beat_sp500 >= beat_sp_required_count) if num_valid_years_aggregated > 0 else False

    # Overall success status
    is_successful_run = all_expected_years_succeeded and beat_sp_threshold_met

    # --- Prepare Final Summary Text ---
    final_summary_text = [
        f"="*15 + f" Run {eval_run_id} Final Summary " + f"="*15,
        f"Config: Prob Threshold={decoded_prob_threshold:.4f}, Min R:R={decoded_min_rr_ratio:.2f}, Indicators={len(feature_cols)}",
        f"Risk Per Trade: {RISK_PERCENTAGE_PER_TRADE * 100:.2f}%", # <<< ADDED
        f"Overall Cumulative Return: {overall_return_pct:.2f}% ({num_valid_years_aggregated} valid sim years)",
        f"Total Trades: {total_trades}", f"Avg Trades/Valid Sim Year: {avg_trades_per_year:.1f}",
        f"Avg Win Rate (Valid Years): {avg_win_rate:.2f}%",
        f"Avg Sharpe (Valid Years, NaN/inf=0): {avg_sharpe:.3f}",
        f"Avg Sortino (Valid Years, NaN/inf=0): {avg_sortino:.3f}",
        f"Avg Max Drawdown (Valid Years): {avg_max_dd:.2f}%",
        f"Years Successfully Simulated: {num_valid_years_aggregated} / {total_expected_sim_years} (Expected Range: {expected_sim_years})",
        f"Years Beat S&P 500: {years_beat_sp500} / {num_valid_years_aggregated} (Required: >= {int(beat_sp_required_count)})",
        f"Run Status Based on Sim Criteria: {'SUCCESSFUL' if is_successful_run else 'UNSUCCESSFUL'}",
        f"Final Model Trained: {'Yes' if final_model is not None else 'No'}",
        f"Total Evaluation Time: {datetime.now() - instance_start_time}"
    ]
    final_summary_text.extend(["-"*10 + " Yearly Simulation Details " + "-"*10])
    final_summary_text.extend(results_summary)
    logger.info("\n".join(final_summary_text))

    # --- Save final results ---
    try:
        results_file_path = os.path.join(run_dir, 'results.txt')
        mode = 'a' if os.path.exists(results_file_path) else 'w' # Append if 0-trade fail message exists
        with open(results_file_path, mode) as f: f.write("\n" + "\n".join(final_summary_text))
        if not all_trades.empty:
            all_trades.to_csv(os.path.join(run_dir, 'all_trades.csv'), index=False)
    except Exception as e: logger.error(f"Run {eval_run_id}: Failed to save results/trades files: {e}")


    # --- Define Fitness Score (higher is better) ---
    # Base fitness on key performance metrics
    fitness = 0.0

    # 1. Risk-Adjusted Return (Sharpe Ratio) - Primary driver
    # Scale Sharpe: e.g., Sharpe of 1 maps to 100 points, max 200, min -100
    # Sharpe might be lower now due to smaller position sizes, adjust scaling if needed
    sharpe_contribution = max(-100, min(avg_sharpe * 100, 200)) if num_valid_years_aggregated > 0 else -200
    fitness += sharpe_contribution

    # 2. Consistency / Beating Benchmark
    # Bonus points for each year beating S&P 500
    fitness += years_beat_sp500 * 30 # Award points per beat

    # 3. Robustness / Stability
    # Penalize heavily for failed/skipped simulation years
    if total_expected_sim_years > 0:
         failed_or_skipped_years = total_expected_sim_years - num_valid_years_aggregated
         fitness -= (failed_or_skipped_years / total_expected_sim_years) * 500 # Max penalty -500 if all fail

    # Penalize if didn't meet the S&P beat threshold
    if num_valid_years_aggregated > 0 and not beat_sp_threshold_met:
        shortfall_pct = (beat_sp_required_count - years_beat_sp500) / beat_sp_required_count if beat_sp_required_count > 0 else 1.0
        fitness -= shortfall_pct * 200 # Max penalty -200 if 0 beats when required > 0

    # 4. Trade Activity
    # Penalize very low average trade counts (but 0 trades already handled by fail-fast)
    min_avg_trades_threshold = 5 # Min avg trades per *valid* year
    if num_valid_years_aggregated > 0:
        if avg_trades_per_year < 1:
            fitness -= 300 # Heavy penalty for < 1 avg trade/year
        elif avg_trades_per_year < min_avg_trades_threshold:
             # Proportional penalty for being below threshold
             fitness -= 200 * (1 - (avg_trades_per_year / min_avg_trades_threshold))


    # 5. Final Model Penalty
    # Penalize if final model training was expected but failed
    final_train_should_have_run = (not final_train_data.empty and len(final_train_data) >= min_train_samples)
    if final_train_should_have_run and final_model is None:
        fitness -= 100 # Penalty if final model failed despite enough data

    # 6. Overall Success Penalty
    # Apply a large penalty if the basic success criteria (all years ran + S&P beats) were not met
    if not is_successful_run and total_expected_sim_years > 0:
         fitness -= 500 # Large penalty for not meeting basic success criteria

    # 7. Drawdown Penalty (more relevant with fixed risk)
    # Penalize high average drawdown
    # Example: -10% DD -> -10 penalty, -30% DD -> -90 penalty
    if num_valid_years_aggregated > 0 and avg_max_dd < 0: # max_dd is negative
        fitness += avg_max_dd * 3 # Multiply negative drawdown by 3 (e.g., -10% DD -> -30 penalty)


    # Final check for NaN/Inf fitness
    if pd.isna(fitness) or np.isinf(fitness):
        logger.error(f"Run {eval_run_id}: Fitness calculation resulted in NaN/Inf ({fitness}). Assigning fallback poor fitness.")
        fitness = -999999 # Fallback poor fitness

    logger.info(f"--- DEAP Eval {eval_run_id} End --- Fitness={fitness:.4f} ---")
    # DEAP requires fitness to be returned as a tuple
    return (fitness,)


# --- Data Loading and Preparation (Global Scope) ---
logger.info("--- Starting Data Preparation ---")
try:
    market_data_raw = load_data(DATA_FILE_PATH)
    market_data_features = add_technical_indicators(market_data_raw)
    if market_data_features.empty: raise ValueError("Feature calculation resulted in empty DataFrame.")
    market_data_processed = create_target_variable(market_data_features)
    if market_data_processed.empty: raise ValueError("Target creation resulted in empty DataFrame.")

    logger.info(f"Filtering data to start from {FILTER_START_DATE.date()}...")
    original_rows = len(market_data_processed)
    market_data_final = market_data_processed[market_data_processed.index >= FILTER_START_DATE].copy()

    # Final cleanup before GA starts - redundant? Maybe necessary after filtering.
    market_data_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Apply ffill/bfill again? Or drop? Let's drop rows with NaNs in *any* column at this stage.
    initial_rows_final = len(market_data_final)
    market_data_final.dropna(inplace=True)
    if len(market_data_final) < initial_rows_final:
         logger.warning(f"Dropped {initial_rows_final - len(market_data_final)} rows with NaNs during final pre-GA cleanup.")


    rows_after_filter = len(market_data_final)
    logger.info(f"Rows before filter: {original_rows} / Rows after filter & final NaN drop: {rows_after_filter}")

    if market_data_final.empty: raise ValueError("Filtering or final NaN drop resulted in empty DataFrame.")
    if 'target_class' not in market_data_final.columns: raise ValueError("'target_class' missing post-processing.")

    logger.info(f"Final data range for analysis: {market_data_final.index.min()} to {market_data_final.index.max()}")
    logger.info(f"Data shape for GA: {market_data_final.shape}")

    # Identify Feature Columns dynamically after all processing
    non_feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target_class', 'until_invalid', 'index'] # Add 'index' if it sneaks in
    # Select numeric columns that aren't the base OHLCV or target/utility columns
    potential_feature_cols = []
    if not market_data_final.empty:
        potential_feature_cols = [
            col for col in market_data_final.columns
            if col not in non_feature_cols
               and pd.api.types.is_numeric_dtype(market_data_final[col])
               # Ensure column has variance (more than 1 unique value)
               and market_data_final[col].nunique(dropna=True) > 1
               # Ensure column doesn't contain ONLY NaNs (already handled by dropna, but safe check)
               and not market_data_final[col].isnull().all()
        ]
    logger.info(f"Identified {len(potential_feature_cols)} potential numeric features for GA.")
    # logger.debug(f"Potential features: {potential_feature_cols}") # Optional: Print feature list

    if not potential_feature_cols: raise ValueError("No potential feature columns found after processing.")
    if len(potential_feature_cols) < INDICATOR_COUNT_BOUNDS[0]:
         raise ValueError(f"Found only {len(potential_feature_cols)} valid features, which is less than the minimum required ({INDICATOR_COUNT_BOUNDS[0]}).")

    logger.info("--- Data Preparation Complete ---")

except Exception as data_prep_error:
      logger.error(f"--- FATAL ERROR DURING DATA PREPARATION ---")
      logger.error(data_prep_error)
      import traceback; logger.error(traceback.format_exc())
      logger.error("Exiting script. Cannot proceed without valid data.")
      exit()


# --- DEAP Setup ---
logger.info("--- Setting up DEAP ---")
# Define Fitness (Maximize single objective)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Define Individual (List of genes with the FitnessMax attribute)
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- Initialize Toolbox ---
toolbox = base.Toolbox()

# --- Gene Generators ---
# Generator for indicator selection bit (0 or 1)
def gen_indicator(): return random.randint(0, 1)

# Generator for LGBM and Strategy parameters (respecting bounds and type)
def gen_lgbm_or_strategy_param(param_name):
    # Ensure bounds are valid
    if param_name not in LGBM_PARAM_BOUNDS or len(LGBM_PARAM_BOUNDS[param_name]) != 2:
        raise ValueError(f"Invalid or missing bounds for parameter '{param_name}' in LGBM_PARAM_BOUNDS.")
    bounds = LGBM_PARAM_BOUNDS[param_name]
    lower, upper = bounds[0], bounds[1]
    if lower > upper: lower, upper = upper, lower # Ensure lower <= upper
    # Generate based on type hint from bounds
    if isinstance(lower, int):
        return random.randint(lower, upper)
    else: # Assume float/real
        return random.uniform(lower, upper)

# --- Individual Structure Generator ---
# Defines how a complete individual is created
def individual_generator():
    # 1. Generate indicator selection genes (list of 0s and 1s)
    num_indicators = len(potential_feature_cols)
    indicator_genes = [gen_indicator() for _ in range(num_indicators)]

    # 2. Generate LGBM and Strategy parameter genes (list of numbers)
    param_genes = [gen_lgbm_or_strategy_param(param_name) for param_name in LGBM_PARAM_ORDER]

    # Combine into a single list [indicators..., params...]
    return indicator_genes + param_genes

# Register the generator for creating a raw individual
toolbox.register("individual_raw", tools.initIterate, creator.Individual, individual_generator)
# Register the generator for creating a population (list of individuals)
toolbox.register("population", tools.initRepeat, list, toolbox.individual_raw)

# --- Genetic Operators ---
# Evaluation Function (already defined above)
toolbox.register("evaluate", evaluate_individual, all_indicator_cols=potential_feature_cols, full_data=market_data_final)

# Crossover Operator (Uniform Crossover)
toolbox.register("mate", tools.cxUniform, indpb=0.5) # indpb is prob of swapping each gene

# Mutation Operator (Custom function to handle different gene types and bounds)
def mutate_individual(individual, indpb, num_indicators, param_bounds, param_order):
    """Mutates an individual respecting bounds and types. Returns tuple: (mutated_individual,)"""
    # Iterate through each gene in the individual
    for i in range(len(individual)):
        # Check if this specific gene should be mutated based on indpb
        if random.random() < indpb:
            # --- Mutate Indicator Gene ---
            if i < num_indicators:
                individual[i] = 1 - individual[i] # Flip the bit (0 -> 1, 1 -> 0)
            # --- Mutate Parameter Gene ---
            else:
                param_idx = i - num_indicators
                # Ensure the index corresponds to a known parameter
                if param_idx < len(param_order):
                    param_name = param_order[param_idx]
                    # Check bounds definition
                    if param_name not in param_bounds or len(param_bounds[param_name]) != 2:
                        logger.warning(f"Mutation skipped for gene {i} ('{param_name}'): Invalid bounds defined.")
                        continue
                    bounds = param_bounds[param_name]
                    lower, upper = bounds[0], bounds[1]
                    if lower > upper: lower, upper = upper, lower # Ensure lower <= upper

                    # Mutate based on type
                    if isinstance(lower, int):
                        # Resample integer within bounds
                        if lower == upper: individual[i] = lower # Handle fixed param case
                        else: individual[i] = random.randint(lower, upper)
                    else: # Float/Real parameter - Apply Gaussian mutation & clamp
                        current_value = individual[i]
                        # Handle potential NaN/Inf in gene before mutation
                        if pd.isna(current_value) or np.isinf(current_value):
                             current_value = random.uniform(lower, upper) # Resample if invalid

                        # Determine mutation strength (sigma) based on range
                        range_width = upper - lower
                        sigma = range_width * 0.1 if range_width > 1e-9 else 0.1 # Scale sigma, avoid 0
                        # Generate new value using Gaussian distribution centered around current value
                        new_val = random.gauss(current_value, sigma)
                        # Clamp the mutated value to the defined bounds
                        individual[i] = max(lower, min(upper, new_val))
                else:
                     logger.warning(f"Mutation index {i} out of bounds for parameter list (len {len(param_order)}). Skipping.")
    # DEAP mutation functions must return the modified individual in a tuple
    return individual,

# Register the custom mutation function
# GA_MUTPB = gene-level mutation probability passed as 'indpb' to the function
toolbox.register("mutate", mutate_individual, indpb=GA_MUTPB, num_indicators=len(potential_feature_cols),
                 param_bounds=LGBM_PARAM_BOUNDS, param_order=LGBM_PARAM_ORDER)

# Selection Operator (Tournament Selection)
toolbox.register("select", tools.selTournament, tournsize=GA_TOURNSIZE)

logger.info("--- DEAP Setup Complete ---")


# --- Main Execution Logic ---
if __name__ == "__main__":
    main_start_time = datetime.now()
    logger.info(f"--- Starting Main Execution with DEAP ({ga_execution_id}) ---")
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Initialize GA State Variables ---
    start_gen = 0
    population = None
    # HallOfFame stores the best individual found so far
    halloffame = tools.HallOfFame(1) # Store only the single best
    # Logbook records statistics for each generation
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "nevals", "avg", "std", "min", "max" # Standard headers
    # Statistics object to compute stats from population fitness
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else np.nan)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # --- Load Checkpoint if exists ---
    if os.path.exists(CHECKPOINT_FILE):
        try:
            logger.info(f"Attempting to load checkpoint from: {CHECKPOINT_FILE}")
            with open(CHECKPOINT_FILE, "rb") as cp_file:
                cp = pickle.load(cp_file)
            population = cp["population"]
            start_gen = cp["generation"] + 1 # Start from the next generation
            halloffame = cp["halloffame"]
            logbook = cp["logbook"]
            # Restore Random Number Generator state for reproducibility
            if "random_state" in cp: random.setstate(cp["random_state"])
            else: logger.warning("Checkpoint missing 'random_state'. Using fresh RNG seed.")

            # --- Compatibility Check (essential if code structure/params changed) ---
            num_expected_genes = len(potential_feature_cols) + len(LGBM_PARAM_ORDER)
            checkpoint_compatible = True
            if not population:
                 logger.warning("Checkpoint loaded, but population is empty. Starting fresh.")
                 checkpoint_compatible = False
            elif len(population[0]) != num_expected_genes:
                 length_found = len(population[0])
                 logger.error(f"Checkpoint individual length mismatch ({length_found}). Expected {num_expected_genes}. Checkpoint is incompatible. Starting fresh.")
                 checkpoint_compatible = False

            if checkpoint_compatible:
                 logger.info(f"Resumed from checkpoint. Starting generation {start_gen}/{GA_GENERATIONS}.")
                 # Report fitness of best individual from checkpoint
                 if halloffame and len(halloffame) > 0 and halloffame[0].fitness.valid:
                     logger.info(f"Best individual from checkpoint: Fitness={halloffame[0].fitness.values[0]:.4f}")
                 else:
                     logger.info("Best individual from checkpoint has invalid fitness (will be re-evaluated).")
                 # Invalidate fitness of the loaded population - forces re-evaluation
                 # Ensures changes in the evaluation function are applied
                 logger.info("Invalidating fitness of loaded population for re-evaluation.")
                 for ind in population:
                     ind.fitness.valid = False
            else:
                 # Reset state if checkpoint incompatible
                 start_gen = 0; population = None; halloffame = tools.HallOfFame(1); logbook = tools.Logbook()
                 logbook.header = "gen", "evals", "nevals", "avg", "std", "min", "max"
                 random.seed(datetime.now().timestamp()) # Re-seed RNG

        except FileNotFoundError:
             logger.info("Checkpoint file not found. Starting fresh.")
             start_gen = 0; population = None
        except EOFError: # Handle file corruption / incomplete write
            logger.error(f"Could not load checkpoint due to EOFError (likely corrupted file '{CHECKPOINT_FILE}'). Starting fresh.")
            start_gen = 0; population = None; halloffame = tools.HallOfFame(1); logbook = tools.Logbook()
            logbook.header = "gen", "evals", "nevals", "avg", "std", "min", "max"
            random.seed(datetime.now().timestamp())
        except Exception as e: # Broad exception catch for other loading issues
            logger.error(f"Could not load checkpoint from '{CHECKPOINT_FILE}': {e}. Starting fresh.")
            start_gen = 0; population = None; halloffame = tools.HallOfFame(1); logbook = tools.Logbook()
            logbook.header = "gen", "evals", "nevals", "avg", "std", "min", "max"
            random.seed(datetime.now().timestamp())

    # --- Initialize population if not loaded from checkpoint ---
    if population is None:
        logger.info(f"Initializing new population (Size: {GA_POPULATION_SIZE})...")
        try:
            population = toolbox.population(n=GA_POPULATION_SIZE)
            logger.info(f"Initialized population with {len(population)} individuals.")
            # Fitness is invalid by default for new individuals
        except Exception as pop_init_err:
             logger.error(f"FATAL: Failed to initialize population: {pop_init_err}")
             import traceback; logger.error(traceback.format_exc())
             exit()

    # --- Evaluate Initial Population (or re-evaluate loaded population) ---
    # Select individuals whose fitness is not valid
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    num_to_evaluate = len(invalid_ind)

    if num_to_evaluate > 0:
        logger.info(f"Evaluating {num_to_evaluate} individuals for generation {start_gen}...")
        evaluation_failed = False
        try:
            # Use toolbox.map for potential parallelization (if setup)
            # For sequential evaluation:
            fitnesses = []
            for i, ind in enumerate(invalid_ind):
                logger.info(f"Evaluating individual {i+1}/{num_to_evaluate} (Gen {start_gen})...")
                try:
                    fit_tuple = toolbox.evaluate(ind)
                    if not isinstance(fit_tuple, tuple) or len(fit_tuple) == 0:
                         raise TypeError(f"Evaluation function did not return a valid tuple fitness. Got: {fit_tuple}")
                    fitnesses.append(fit_tuple)
                except Exception as eval_err: # Catch errors during individual evaluation
                     logger.error(f"ERROR evaluating individual {i+1} (Gen {start_gen}): {eval_err}")
                     # Assign a very poor fitness tuple to penalize failed evaluations
                     fitnesses.append((-999999,)) # Must match the expected fitness structure (tuple)
                     evaluation_failed = True # Flag that at least one evaluation failed

            # Assign calculated fitnesses back to individuals
            if len(fitnesses) != num_to_evaluate:
                 logger.error(f"FATAL: Fitness assignment mismatch. Expected {num_to_evaluate}, got {len(fitnesses)}. Aborting.")
                 exit()
            for ind, fit in zip(invalid_ind, fitnesses):
                 ind.fitness.values = fit # Assign the tuple

            logger.info(f"Evaluation for generation {start_gen} complete.")
            if evaluation_failed: logger.warning("WARNING: One or more evaluations failed during initial/resumed generation. Assigned poor fitness.")

        except Exception as map_eval_err: # Catch errors in the mapping/looping process itself
            logger.error(f"FATAL ERROR during population evaluation: {map_eval_err}")
            import traceback; logger.error(traceback.format_exc())
            exit()

    # --- Update HallOfFame and Logbook after initial evaluation ---
    if population:
        # Ensure HoF only gets valid individuals
        valid_pop = [ind for ind in population if ind.fitness.valid]
        if valid_pop:
             halloffame.update(valid_pop)
             try:
                 record = stats.compile(valid_pop) # Compile stats from valid individuals
                 # Record stats for this generation (gen=start_gen)
                 # 'evals' = total evaluations done *in this specific step* (could be 0 if all loaded were valid)
                 # 'nevals' = number of *new* evaluations (same as evals here as we invalidated all loaded)
                 logbook.record(gen=start_gen, evals=num_to_evaluate, nevals=num_to_evaluate, **record)
                 logger.info(f"Stats after evaluation (Generation {start_gen}): {logbook.stream}")
             except Exception as log_err:
                 logger.error(f"Error recording stats for generation {start_gen}: {log_err}")
        else:
             logger.warning(f"No valid individuals found after evaluation for generation {start_gen}. Cannot update HoF or logbook.")


    # --- Main Evolutionary Loop (using DEAP's eaSimple algorithm) ---
    logger.info(f"Starting DEAP evolution from generation {start_gen} up to generation {GA_GENERATIONS-1}...")

    # Check if we already completed the target number of generations
    if start_gen >= GA_GENERATIONS:
         logger.info(f"Checkpoint generation {start_gen} >= target generations {GA_GENERATIONS}. Skipping evolution loop.")
    else:
        # Run the evolutionary algorithm
        try:
            # Ensure population is valid before starting loop
            if not population: raise ValueError("Population is empty or None before starting eaSimple.")

            # eaSimple runs for 'ngen' generations *starting from the state passed in*
            # So if start_gen=1 and GA_GENERATIONS=5, it runs for ngen=4 (gens 1, 2, 3, 4)
            generations_to_run = GA_GENERATIONS - start_gen

            population, logbook = algorithms.eaSimple(
                population,          # Current population state
                toolbox,             # DEAP toolbox with registered operators
                cxpb=GA_CROSSOVER_PROB,  # Probability of mating two individuals
                mutpb=GA_MUTATION_PROB, # Probability of mutating an individual
                ngen=generations_to_run, # Number of generations to run *from start_gen*
                stats=stats,         # Statistics object to compile
                halloffame=halloffame, # HallOfFame object to update
                verbose=True         # Print logbook output each generation
            )

        except KeyboardInterrupt:
             logger.warning("KeyboardInterrupt detected during evolution. Saving checkpoint and exiting.")
             # Save checkpoint on interrupt
             if population is not None:
                  # Determine the last successfully completed generation
                  current_gen = logbook[-1]['gen'] if logbook and len(logbook) > 0 else start_gen -1
                  cp = { 'population': population, 'generation': current_gen, 'halloffame': halloffame,
                        'logbook': logbook, 'random_state': random.getstate() }
                  try:
                      with open(CHECKPOINT_FILE, "wb") as cp_file: pickle.dump(cp, cp_file)
                      logger.info(f"Checkpoint saved successfully to {CHECKPOINT_FILE} at generation {current_gen}.")
                  except Exception as e: logger.error(f"Error saving checkpoint on interrupt: {e}")
             exit() # Exit after attempting to save
        except Exception as ea_err:
            logger.error(f"FATAL ERROR during DEAP algorithm execution (eaSimple): {ea_err}")
            import traceback; logger.error(traceback.format_exc())
            # Attempt to save checkpoint even on error?
            if population is not None:
                 current_gen = logbook[-1]['gen'] if logbook and len(logbook) > 0 else start_gen -1
                 cp = { 'population': population, 'generation': current_gen, 'halloffame': halloffame,
                        'logbook': logbook, 'random_state': random.getstate() }
                 try:
                     with open(CHECKPOINT_FILE, "wb") as cp_file: pickle.dump(cp, cp_file)
                     logger.info(f"Checkpoint saved successfully to {CHECKPOINT_FILE} at generation {current_gen} after error.")
                 except Exception as e: logger.error(f"Error saving checkpoint after error: {e}")
            # Exit after error
            exit()

    # --- Process DEAP Results ---
    logger.info("=" * 60); logger.info("DEAP Genetic Algorithm Finished.")
    logger.info(f"Total Execution Time: {datetime.now() - main_start_time}")

    # --- Save Final State (Checkpoint) ---
    # Useful even if completed normally, allows inspecting final population
    if population is not None:
        # Get the generation number of the last completed generation
        final_gen = logbook[-1]['gen'] if logbook and len(logbook) > 0 else GA_GENERATIONS - 1
        cp = {
            'population': population,
            'generation': final_gen,
            'halloffame': halloffame,
            'logbook': logbook,
            'random_state': random.getstate()
        }
        try:
            final_cp_path = os.path.join(CHECKPOINT_DIR, f"ga_checkpoint_final_gen{final_gen}.pkl")
            with open(final_cp_path, "wb") as cp_file: pickle.dump(cp, cp_file)
            # Also save to the standard checkpoint file name
            with open(CHECKPOINT_FILE, "wb") as cp_file: pickle.dump(cp, cp_file)
            logger.info(f"Final state saved successfully to {final_cp_path} (and {CHECKPOINT_FILE}) at generation {final_gen}.")
        except Exception as e: logger.error(f"Error saving final checkpoint: {e}")


    # --- Report Best Individual Found ---
    if halloffame is not None and len(halloffame) > 0:
        best_ind = halloffame[0] # Get the best individual from HoF
        if best_ind.fitness.valid:
            best_fitness = best_ind.fitness.values[0]
            logger.info(f"Best Individual Found (Fitness Score): {best_fitness:.4f}")

            # --- Decode the best individual again for final reporting ---
            try:
                num_indicators = len(potential_feature_cols)
                num_params = len(LGBM_PARAM_ORDER)
                if len(best_ind) != num_indicators + num_params:
                     raise ValueError(f"Best individual length mismatch. Got {len(best_ind)}, expected {num_indicators + num_params}")

                # Decode indicators
                best_indicator_mask = [bool(gene) for gene in best_ind[:num_indicators]]
                best_selected_indicators = [col for i, col in enumerate(potential_feature_cols) if best_indicator_mask[i]]

                # Decode parameters
                best_lgbm_and_strategy_vals = best_ind[num_indicators:]
                best_lgbm_config = { # Reset for clarity
                    'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': NUM_CLASSES,
                    'boosting_type': 'gbdt', 'n_jobs': -1, 'verbose': -1, 'seed': 42, 'verbosity': -1,
                }
                best_decoded_prob_threshold = None
                best_decoded_min_rr_ratio = None

                for i, param_name in enumerate(LGBM_PARAM_ORDER):
                     bounds = LGBM_PARAM_BOUNDS[param_name]
                     lower, upper = bounds[0], bounds[1]
                     if lower > upper: lower, upper = upper, lower # Swap if needed
                     raw_val = best_lgbm_and_strategy_vals[i]

                     # Handle potential NaN/inf in the *best* individual's genes
                     if pd.isna(raw_val) or np.isinf(raw_val):
                         val = lower # Fallback to lower bound
                         logger.warning(f"Best individual had invalid gene for '{param_name}'. Using lower bound: {val}")
                     elif isinstance(lower, int): val = max(lower, min(upper, int(round(raw_val))))
                     else: val = max(lower, min(upper, float(raw_val)))

                     # Separate strategy params from LGBM config
                     if param_name == 'probability_threshold': best_decoded_prob_threshold = val
                     elif param_name == 'min_rr_ratio': best_decoded_min_rr_ratio = val
                     else: best_lgbm_config[param_name] = val

                logger.info("--- Best Configuration Found ---")
                logger.info(f"Number of Indicators: {len(best_selected_indicators)}")
                # logger.info(f"Selected Indicators:\n{best_selected_indicators}") # Optional: Print list
                logger.info(f"Best LightGBM Params (decoded): {best_lgbm_config}")
                if best_decoded_prob_threshold is not None: logger.info(f"Best Probability Threshold (decoded): {best_decoded_prob_threshold:.4f}")
                else: logger.warning("Could not decode best probability threshold.")
                if best_decoded_min_rr_ratio is not None: logger.info(f"Best Min R:R Ratio (decoded): {best_decoded_min_rr_ratio:.2f}")
                else: logger.warning("Could not decode best min R:R ratio.")
                logger.info(f"Risk Per Trade Used: {RISK_PERCENTAGE_PER_TRADE * 100:.2f}%") # <<< ADDED


                # --- Save Best Config Summary to JSON ---
                best_config_summary = {
                    'best_fitness_score': best_fitness,
                    'best_selected_indicators': best_selected_indicators,
                    'best_lightgbm_params_decoded': best_lgbm_config,
                    'best_probability_threshold_decoded': best_decoded_prob_threshold,
                    'best_min_rr_ratio_decoded': best_decoded_min_rr_ratio,
                    'best_individual_raw_genes': list(best_ind), # Save the raw gene list for reference
                    'ga_parameters': {
                         'population_size': GA_POPULATION_SIZE, 'generations_configured': GA_GENERATIONS,
                         'crossover_prob': GA_CROSSOVER_PROB, 'mutation_prob_individual': GA_MUTATION_PROB,
                         'mutation_prob_gene': GA_MUTPB, 'tournament_size': GA_TOURNSIZE,
                    },
                    'other_config': { # Include key settings for reproducibility
                        'data_file': DATA_FILE_PATH, 'filter_start_date': FILTER_START_DATE.isoformat(),
                        'initial_train_end_year': INITIAL_TRAIN_END_YEAR,
                        'simulation_start_year': SIMULATION_START_YEAR,
                        'current_date_used': CURRENT_DATE.isoformat(),
                        'target_horizon': TARGET_HORIZON_MINUTES,
                        'pip_levels': PIP_LEVELS,
                        'pip_definition': PIP_DEFINITION, 'news_threshold': INVALID_NEWS_THRESHOLD,
                        'initial_cash': INITIAL_CASH, 'commission': COMMISSION_PERC, # Storing original %
                        'commission_decimal': COMMISSION_DECIMAL, # Storing decimal used
                        'risk_percentage_per_trade': RISK_PERCENTAGE_PER_TRADE, # <<< ADDED
                        'lgbm_and_strategy_param_bounds': LGBM_PARAM_BOUNDS,
                    },
                    'ga_execution_id': ga_execution_id, # ID of the overall GA run
                     # Add link to the best individual's run directory? Requires finding it.
                }

                # --- Find the run directory of the best individual (requires searching results) ---
                # This part is tricky as the HoF individual might be from an earlier generation.
                # A robust way is to search the 'results.txt' files in RUNS_DIR for the best fitness.
                best_run_id = None
                best_run_fitness = -float('inf')
                try:
                    logger.info("Searching run directories for the best individual's results...")
                    # Check if RUNS_DIR exists before trying to list its contents
                    if not os.path.isdir(RUNS_DIR):
                         logger.warning(f"Run directory '{RUNS_DIR}' not found. Cannot search for best run ID.")
                    else:
                        for item in os.listdir(RUNS_DIR):
                            item_path = os.path.join(RUNS_DIR, item)
                            # Check if it's a directory and matches the expected naming pattern for this GA run
                            if os.path.isdir(item_path) and item.startswith(f"deap_{ga_execution_id}_eval_"):
                                results_path = os.path.join(item_path, 'results.txt')
                                if os.path.exists(results_path):
                                    try:
                                        with open(results_path, 'r') as f:
                                            content = f.read()
                                        # Extract fitness (assuming format like "--- Fitness=...")
                                        fit_line = [line for line in content.split('\n') if "--- Fitness=" in line] # Adjusted search string
                                        if fit_line:
                                             try:
                                                  # Extract float after the equals sign
                                                  current_fitness = float(fit_line[0].split('Fitness=')[-1].split('---')[0].strip())
                                                  # Use np.isclose for robust float comparison
                                                  if np.isclose(current_fitness, best_fitness):
                                                       best_run_id = item # Found the matching run
                                                       best_config_summary['best_individual_run_id'] = best_run_id
                                                       logger.info(f"Found matching run directory for best fitness: {best_run_id}")
                                                       break # Stop searching once found
                                             except (ValueError, IndexError) as parse_err:
                                                  logger.debug(f"Could not parse fitness from line in {results_path}: '{fit_line[0]}' - {parse_err}")
                                                  continue # Ignore lines where Fitness isn't extractable
                                    except Exception as read_err:
                                         logger.warning(f"Error reading results file {results_path}: {read_err}")
                        if best_run_id is None:
                             logger.warning("Could not automatically find the specific run directory for the best HoF individual. It might be from a previous GA execution if resuming, or the results.txt format changed.")

                except Exception as search_err:
                    logger.error(f"Error searching for best individual's run directory: {search_err}")


                # --- Save the final best config JSON ---
                best_config_file = os.path.join(RUNS_DIR, f"deap_{ga_execution_id}_BEST_CONFIG_SUMMARY.json")
                try:
                    # Use the globally defined serializer
                    with open(best_config_file, 'w') as f:
                        json.dump(best_config_summary, f, indent=4, default=default_serializer)
                    logger.info(f"Best configuration summary saved to: {best_config_file}")
                except Exception as e:
                    logger.error(f"Failed to save best config summary JSON: {e}")

                logger.info("---")
                if best_run_id:
                    logger.info(f"Detailed results (trades, model, plots) for the best individual are in directory:")
                    logger.info(f"'{os.path.join(RUNS_DIR, best_run_id)}'")
                else:
                    logger.info("To get detailed results for the best individual, manually check logs/results.txt")
                    logger.info("in the subdirectories within 'deap_runs' for the run with the highest fitness score.")

            except Exception as decode_err:
                logger.error(f"Error decoding or reporting best individual: {decode_err}")
                import traceback; logger.error(traceback.format_exc())

        else:
             logger.warning("DEAP finished, but the best individual in HallOfFame has invalid fitness. Check logs for evaluation errors.")
    else:
        logger.warning("DEAP finished, but HallOfFame was empty or None. No best individual found.")

    logger.info(f"--- Main execution finished ({ga_execution_id}). ---")
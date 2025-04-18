import os
import logging
import warnings
import requests
import pandas as pd
import numpy as np
import ta
import lightgbm as lgb
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
import joblib
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from datetime import datetime, time
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.INFO)

QUESTDB_URL       = "http://localhost:9000/exec"
TABLE_NAME        = "XAUUSD_1M_TRAINING_WITH_FEATURES"
TRAIN_START_DATE  = "2018-01-01"
TRAIN_END_DATE    = "2022-12-31"
VAL_START_DATE    = "2023-01-01"
VAL_END_DATE      = "2023-12-31"
TEST_2024_START   = "2024-01-01"
TEST_2024_END     = "2024-12-31"
INITIAL_CAPITAL    = 10_000
MAX_RISK_PER_TRADE = 0.015
MAX_LEVERAGE       = 50
TRANSACTION_COST   = 0.001
SLIPPAGE           = 0.0005
RANDOM_SEED = 42
N_TRIALS = 50

TIMEFRAME = 5
N_DOLLARS = 15
BIN_SIZE_DOLLARS = 1
ALPHA_DOLLARS = 5
PROB_THRESHOLD = 0.65
MIN_RR = 1.5

BASE_LGBM_PARAMS = {
    'objective': 'multiclass',
    'boosting_type': 'gbdt',
    'feature_fraction_seed': 42,
    'bagging_seed': 42,
    'boost_from_average': True,
    'is_unbalance': False,
    'verbose': -1
}

def fetch_table_data():
    query = f"""
    SELECT * FROM {TABLE_NAME}
    WHERE timestamp >= '{TRAIN_START_DATE}'
      AND timestamp <= '{TEST_2024_END}'
    """
    response = requests.get(QUESTDB_URL, params={"query": query})
    response.raise_for_status()
    data_json = response.json()

    columns = [col["name"] for col in data_json["columns"]]
    rows    = data_json["dataset"]
    df      = pd.DataFrame(rows, columns=columns)

    for col in ["GAPFLAG", "tradable"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True, ignore_index=True)

    numeric_cols = ["open", "high", "low", "close", "tick_volume", "until_invalid"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    return df

def split_data(df):
    df_train = df[(df['timestamp'] >= TRAIN_START_DATE) & (df['timestamp'] <= TRAIN_END_DATE)].copy()
    df_val   = df[(df['timestamp'] >= VAL_START_DATE)    & (df['timestamp'] <= VAL_END_DATE)].copy()
    df_test  = df[(df['timestamp'] >= TEST_2024_START)   & (df['timestamp'] <= TEST_2024_END)].copy()

    logger.info(f"Train set:      {len(df_train)} rows  [{TRAIN_START_DATE} -> {TRAIN_END_DATE}]")
    logger.info(f"Validation set: {len(df_val)} rows    [{VAL_START_DATE} -> {VAL_END_DATE}]")
    logger.info(f"Test 2024 set:  {len(df_test)} rows    [{TEST_2024_START} -> {TEST_2024_END}]")
    return df_train, df_val, df_test

def engineer_features_relative(df):
    df = df.copy()
    df.sort_values('timestamp', inplace=True)
    
    df['hour'] = df['timestamp'].dt.hour
    
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['london_ny_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
    
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    df['ma_cross_9_21'] = df['ema_9'] / df['ema_21'] - 1
    df['ma_cross_50_200'] = df['sma_50'] / df['sma_200'] - 1
    
    df['price_to_ema_9'] = df['close'] / df['ema_9'] - 1
    df['price_to_ema_21'] = df['close'] / df['ema_21'] - 1
    df['price_to_sma_50'] = df['close'] / df['sma_50'] - 1
    df['price_to_sma_200'] = df['close'] / df['sma_200'] - 1
    
    df['ema_9_slope'] = df['ema_9'].pct_change(5)
    df['ema_21_slope'] = df['ema_21'].pct_change(5)
    df['sma_50_slope'] = df['sma_50'].pct_change(10)
    df['sma_200_slope'] = df['sma_200'].pct_change(20)
    
    df['above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
    df['below_sma_50'] = (df['close'] < df['sma_50']).astype(int)
    
    df['trend_direction'] = np.sign(df['close'] - df['sma_50'])
    df['trend_direction_change'] = df['trend_direction'].diff().ne(0).astype(int)
    
    df['trend_streak'] = df.groupby(df['trend_direction_change'].cumsum())['trend_direction_change'].cumcount()
    df['trend_streak'] = np.minimum(df['trend_streak'], 20)
    
    macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    df['macd_line_pct'] = df['macd_line'] / df['close'] * 100
    df['macd_signal_pct'] = df['macd_signal'] / df['close'] * 100
    df['macd_histogram_pct'] = df['macd_histogram'] / df['close'] * 100
    
    df['macd_cross'] = np.sign(df['macd_line'] - df['macd_signal'])
    df['macd_cross_change'] = df['macd_cross'].diff().ne(0).astype(int)
    df['macd_hist_direction'] = np.sign(df['macd_histogram'])
    df['macd_hist_change'] = df['macd_hist_direction'].diff().ne(0).astype(int)
    
    psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'], step=0.02, max_step=0.2)
    df['psar'] = psar.psar()
    
    df['psar_position'] = np.where(df['close'] > df['psar'], 1, -1)
    df['psar_distance'] = (df['close'] - df['psar']) / df['close']
    
    ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    
    df['price_to_kijun'] = df['close'] / df['ichimoku_base'] - 1
    df['tenkan_kijun_cross'] = df['ichimoku_conv'] / df['ichimoku_base'] - 1
    
    df['cloud_thickness'] = (df['ichimoku_a'] - df['ichimoku_b']) / df['close']
    df['above_cloud'] = ((df['close'] > df['ichimoku_a']) & (df['close'] > df['ichimoku_b'])).astype(int)
    df['below_cloud'] = ((df['close'] < df['ichimoku_a']) & (df['close'] < df['ichimoku_b'])).astype(int)
    df['in_cloud'] = (~df['above_cloud'].astype(bool) & ~df['below_cloud'].astype(bool)).astype(int)
    
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
    
    df['roc_5'] = ta.momentum.ROCIndicator(df['close'], window=5).roc()
    df['roc_10'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
    df['roc_20'] = ta.momentum.ROCIndicator(df['close'], window=20).roc()
    
    rsi_scaled = 2 * (df['rsi_14'] - 50) / 100
    df['fisher_rsi_14'] = 0.5 * np.log((1 + rsi_scaled) / (1 - rsi_scaled + 1e-9))
    
    boll = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['boll_pct_b'] = boll.bollinger_pband()
    df['boll_width'] = boll.bollinger_wband()
    
    atr_raw = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()
    df['atr_pct'] = atr_raw / df['close'] * 100
    
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    df['adx_trend_strength'] = df['adx'] * np.sign(df['adx_pos'] - df['adx_neg'])
    df['di_spread'] = (df['adx_pos'] - df['adx_neg']) / (df['adx_pos'] + df['adx_neg'])
    
    adx5 = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=5)
    df['adx5'] = adx5.adx()
    
    donch55 = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=55)
    high_band55 = donch55.donchian_channel_hband()
    low_band55 = donch55.donchian_channel_lband()
    mid_band55 = donch55.donchian_channel_mband()
    df['donchian_pos_55'] = (df['close'] - low_band55) / (high_band55 - low_band55 + 1e-9)
    df['donchian_width_55'] = (high_band55 - low_band55) / mid_band55
    
    df['donchian_upper_break'] = (df['close'] > high_band55.shift(1)).astype(int)
    df['donchian_lower_break'] = (df['close'] < low_band55.shift(1)).astype(int)
    df['donchian_breakout'] = df['donchian_upper_break'] - df['donchian_lower_break']
    
    donch = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=20)
    high_band = donch.donchian_channel_hband()
    low_band = donch.donchian_channel_lband()
    mid_band = donch.donchian_channel_mband()
    df['donchian_pos'] = (df['close'] - low_band) / (high_band - low_band + 1e-9)
    df['donchian_width'] = (high_band - low_band) / mid_band
    
    df['volume_price_corr'] = df['close'].rolling(window=20).corr(df['tick_volume'])
    
    df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(window=20).mean()
    
    obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['tick_volume'])
    df['obv_raw'] = obv.on_balance_volume()
    df['obv_ema'] = df['obv_raw'].ewm(span=20, adjust=False).mean()
    df['obv_change'] = df['obv_raw'].pct_change(20)
    df['obv_slope'] = (df['obv_raw'] - df['obv_raw'].shift(5)) / df['obv_raw'].shift(5)
    
    for lag in range(1, 6):
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
    
    for window in [10, 20, 50, 100]:
        df[f'price_to_ma_{window}'] = df['close'] / df['close'].rolling(window=window).mean() - 1
    
    df['volatility_10'] = df['close'].rolling(window=10).std() / df['close'].rolling(window=10).mean()
    df['volatility_20'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_20d'] = df['close'].pct_change(20)
    
    df['high_low_pct'] = (df['high'] - df['low']) / df['close']
    df['open_close_pct'] = (df['close'] - df['open']) / df['open']
    
    df['pct_from_20d_high'] = df['close'] / df['high'].rolling(20).max() - 1
    df['pct_from_20d_low'] = df['close'] / df['low'].rolling(20).min() - 1
    
    rolling_mean10 = df['close'].rolling(window=10).mean()
    rolling_std10 = df['close'].rolling(window=10).std()
    df['z_score_10'] = (df['close'] - rolling_mean10) / rolling_std10
    
    for window in [20, 50]:
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        df[f'z_score_{window}'] = (df['close'] - rolling_mean) / rolling_std
    
    if 'until_invalid' in df.columns:
        df['is_event_near'] = df['until_invalid'].shift(1) < 1
    else:
        df['is_event_near'] = False

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def compute_bin_edges(n_dollars: float,
                      bin_size: float,
                      alpha: float) -> np.ndarray:
    neg_edges = np.arange(-n_dollars, -alpha + bin_size, bin_size)
    if neg_edges.size == 0 or neg_edges[-1] < -alpha:
        neg_edges = np.append(neg_edges, -alpha)

    pos_edges = np.arange(alpha, n_dollars + bin_size, bin_size)
    if pos_edges.size > 0 and pos_edges[-1] > n_dollars:
        pos_edges = pos_edges[:-1]
        pos_edges = np.append(pos_edges, n_dollars)

    bin_edges = np.concatenate([neg_edges, pos_edges])
    return bin_edges

def get_neg_pos_bin_indices(bin_edges: np.ndarray):
    neg_indices = []
    pos_indices = []
    num_bins = len(bin_edges) - 1
    for i in range(num_bins):
        left_edge  = bin_edges[i]
        right_edge = bin_edges[i+1]
        if right_edge <= 0:
            neg_indices.append(i)
        elif left_edge >= 0:
            pos_indices.append(i)
    return neg_indices, pos_indices

def create_labels(df: pd.DataFrame,
                  timeframe: int,
                  bin_edges: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["future_return_dollars"] = df["close"].shift(-timeframe) - df["close"]
    df["movement_bin"] = pd.cut(
        df["future_return_dollars"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        right=False
    )
    df.dropna(subset=["movement_bin"], inplace=True)
    df["movement_bin"] = df["movement_bin"].astype(int)
    return df

def prune_features(df_train, model=None, feature_imp=None, prune_percent=20):
    ignore_cols = {"timestamp", "future_return_dollars", "movement_bin",
                   "tradable", "GAPFLAG", "is_event_near", "until_invalid",
                   "hour"}
    
    all_features = [c for c in df_train.columns if c not in ignore_cols]
    
    if model is None and feature_imp is None:
        return all_features
    
    if feature_imp is None:
        feature_imp = pd.DataFrame({
            'Feature': model.feature_name_,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
    
    keep_count = int(len(all_features) * (100 - prune_percent) / 100)
    features_to_keep = feature_imp['Feature'].head(keep_count).tolist()
    
    removed_features = [f for f in all_features if f not in features_to_keep]
    logger.info(f"Pruned {len(removed_features)} features, keeping {len(features_to_keep)} features")
    logger.info(f"Removed features: {removed_features}")
    
    return features_to_keep

def train_with_selected_features(df_train, df_val, bin_edges, features_to_use, timeframe, params):
    df_train_labeled = create_labels(df_train, timeframe, bin_edges)
    unique_train_bins = sorted(df_train_labeled["movement_bin"].unique())
    num_classes = len(unique_train_bins)
    df_val_labeled = create_labels(df_val, timeframe, bin_edges)
    
    X_train = df_train_labeled[features_to_use]
    y_train = df_train_labeled["movement_bin"]
    X_val = df_val_labeled[features_to_use]
    y_val = df_val_labeled["movement_bin"]
    
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        min_child_samples=params['min_child_samples'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        subsample_freq=params.get('subsample_freq', 0),
        reg_alpha=params.get('reg_alpha', 0.0),
        reg_lambda=params.get('reg_lambda', 0.0),
        min_split_gain=params.get('min_split_gain', 0.0),
        max_bin=params.get('max_bin', 255),
        boosting_type=params.get('boosting_type', 'gbdt'),
        path_smooth=params.get('path_smooth', 0.0),
        class_weight='balanced',
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1
    )
    
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100)
            ]
        )
    except KeyboardInterrupt:
        print("Training interrupted. Exiting gracefully.")
        raise
    
    return model, features_to_use

def train_incrementally(df_train, df_val, bin_edges, features_to_use, timeframe, params):
    df_combined = pd.concat([df_train, df_val], ignore_index=True)
    df_combined.sort_values('timestamp', inplace=True)
    
    logger.info(f"Incremental model training on combined data: {len(df_combined)} rows")
    logger.info(f"Period: [{df_combined['timestamp'].min()} -> {df_combined['timestamp'].max()}]")
    
    df_combined_labeled = create_labels(df_combined, timeframe, bin_edges)
    unique_bins = sorted(df_combined_labeled["movement_bin"].unique())
    num_classes = len(unique_bins)
    
    split_idx = int(len(df_combined_labeled) * 0.8)
    df_train_split = df_combined_labeled.iloc[:split_idx]
    df_val_split = df_combined_labeled.iloc[split_idx:]
    
    X_train = df_train_split[features_to_use]
    y_train = df_train_split["movement_bin"]
    X_val = df_val_split[features_to_use]
    y_val = df_val_split["movement_bin"]
    
    logger.info(f"Incremental model: using {len(X_train)} samples for training, {len(X_val)} for validation")
    
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        min_child_samples=params['min_child_samples'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        subsample_freq=params.get('subsample_freq', 0),
        reg_alpha=params.get('reg_alpha', 0.0),
        reg_lambda=params.get('reg_lambda', 0.0),
        min_split_gain=params.get('min_split_gain', 0.0),
        max_bin=params.get('max_bin', 255),
        boosting_type=params.get('boosting_type', 'gbdt'),
        path_smooth=params.get('path_smooth', 0.0),
        class_weight='balanced',
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1
    )
    
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100)
            ]
        )
    except KeyboardInterrupt:
        print("Training interrupted. Exiting gracefully.")
        raise
    
    return model, features_to_use

def predict_proba_with_features(df_input, model, bin_edges, timeframe, features_to_use):
    df_input_labeled = create_labels(df_input, timeframe, bin_edges)
    X_input = df_input_labeled[features_to_use]
    y_proba = model.predict_proba(X_input)
    return y_proba, df_input_labeled

def build_signal_dataframe(df_test: pd.DataFrame,
                           y_pred_proba: np.ndarray,
                           model: lgb.LGBMClassifier,
                           bin_edges: np.ndarray,
                           prob_threshold: float,
                           min_rr: float) -> pd.DataFrame:
    df_out = df_test.copy().reset_index(drop=True)

    train_bin_labels = model.classes_
    bin_label_to_col = {lbl: idx for idx, lbl in enumerate(train_bin_labels)}

    neg_indices, pos_indices = get_neg_pos_bin_indices(bin_edges)

    neg_bins = {}
    for i in neg_indices:
        if i in bin_label_to_col:
            neg_bins[i] = abs(bin_edges[i])

    pos_bins = {}
    for j in pos_indices:
        if j in bin_label_to_col:
            pos_bins[j] = bin_edges[j+1]

    signals = []
    weighted_pos_moves = []
    weighted_neg_moves = []
    prob_diffs = []

    for row_probs in y_pred_proba:
        agg_pos = 0.0
        agg_neg = 0.0
        sum_pos_move = 0.0
        sum_neg_move = 0.0

        for p_bin, move_val in pos_bins.items():
            p_idx = bin_label_to_col[p_bin]
            p_prob = row_probs[p_idx]
            agg_pos += p_prob
            sum_pos_move += p_prob * move_val

        for n_bin, move_val in neg_bins.items():
            n_idx = bin_label_to_col[n_bin]
            n_prob = row_probs[n_idx]
            agg_neg += n_prob
            sum_neg_move += n_prob * move_val

        weighted_pos_move = sum_pos_move / agg_pos if agg_pos > 0 else 0
        weighted_neg_move = sum_neg_move / agg_neg if agg_neg > 0 else 0
        prob_diff = agg_pos - agg_neg

        i = len(signals)
        
        if prob_diff >= prob_threshold:
            signal = 1
        elif -prob_diff >= prob_threshold:
            signal = -1
        else:
            signal = 0
        
        if signal != 0 and i < len(df_out):
            if signal == -1 and df_out["price_to_sma_50"].iloc[i] > 0:
                if -prob_diff < (prob_threshold * 1.2):
                    signal = 0
            
            if signal == 1 and df_out["price_to_sma_50"].iloc[i] < 0:
                if prob_diff < (prob_threshold * 1.2):
                    signal = 0
        
        signals.append(signal)
        weighted_pos_moves.append(weighted_pos_move)
        weighted_neg_moves.append(weighted_neg_move)
        prob_diffs.append(prob_diff)

    df_out["Signal"] = signals
    df_out["prob_diff"] = prob_diffs

    sl_list = []
    tp_list = []
    for i, sig in enumerate(signals):
        price = df_out["close"].iloc[i]
        if sig == 1:
            tp_distance = weighted_pos_moves[i]
            tp = price + tp_distance
            sl = price - (tp_distance / min_rr)
        elif sig == -1:
            tp_distance = weighted_neg_moves[i]
            tp = price - tp_distance
            sl = price + (tp_distance / min_rr)
        else:
            tp = np.nan
            sl = np.nan
        tp_list.append(tp)
        sl_list.append(sl)

    df_out["SL_price"] = sl_list
    df_out["TP_price"] = tp_list
    df_out["weighted_pos_move"] = weighted_pos_moves
    df_out["weighted_neg_move"] = weighted_neg_moves

    return df_out

def calculate_position_size(strategy, entry_price, stop_loss):
    if np.isnan(stop_loss) or stop_loss <= 0:
        return 0

    risk_amount = INITIAL_CAPITAL * MAX_RISK_PER_TRADE
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance < 1e-8:
        return 0
    raw_size = risk_amount / sl_distance

    max_size = (strategy.equity * MAX_LEVERAGE) / entry_price
    final_size = min(raw_size, max_size)
    if final_size >= 1:
        final_size = int(final_size)
    return final_size

class MySignalStrategy(Strategy):
    def init(self):
        self.Signal_col = self.data.Signal
        self.SL_col     = self.data.SL_price
        self.TP_col     = self.data.TP_price
        self.current_sl = None

    def next(self):
        i = len(self.data) - 1
        price = self.data.Close[i]
        signal = self.Signal_col[i]
        sl_signal = self.SL_col[i]
        tp_signal = self.TP_col[i]

        if not self.position:
            if self.data.until_invalid[i] == 0:
                return
            self.current_sl = None
            if signal == 1:
                size = calculate_position_size(self, price, sl_signal)
                if size > 0:
                    self.buy(size=size, sl=sl_signal, tp=tp_signal)
                    self.current_sl = sl_signal
            elif signal == -1:
                size = calculate_position_size(self, price, sl_signal)
                if size > 0:
                    self.sell(size=size, sl=sl_signal, tp=tp_signal)
                    self.current_sl = sl_signal

def plot_prob_diff_with_trades(df, backtest_instance, period_name, out_path=None):
    dates = df['timestamp']
    prob_diffs = df['prob_diff'].values
    prices = df['close'].values
    signals = df['Signal'].values
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    ax1 = plt.subplot(gs[0])
    ax1.plot(dates, prob_diffs, color='black', alpha=0.7, linewidth=0.8)
    
    ax1.axhline(y=PROB_THRESHOLD, color='green', linestyle='--', alpha=0.5, label=f'+Threshold: {PROB_THRESHOLD}')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axhline(y=-PROB_THRESHOLD, color='red', linestyle='--', alpha=0.5, label=f'-Threshold: {-PROB_THRESHOLD}')
    
    long_signals = np.where(signals == 1)[0]
    short_signals = np.where(signals == -1)[0]
    
    if len(long_signals) > 0:
        ax1.scatter(dates.iloc[long_signals], prob_diffs[long_signals], 
                   color='green', marker='^', s=80, alpha=0.7, label='Long Signal')
    
    if len(short_signals) > 0:
        ax1.scatter(dates.iloc[short_signals], prob_diffs[short_signals], 
                   color='red', marker='v', s=80, alpha=0.7, label='Short Signal')
    
    ax1.set_title(f'Probability Difference and Signals - {period_name}', fontsize=14)
    ax1.set_ylabel('Probability Difference (Pos - Neg)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(dates, prices, color='blue', alpha=0.8, linewidth=0.8)
    
    if hasattr(backtest_instance, '_trades') and len(backtest_instance._trades) > 0:
        trades = backtest_instance._trades
        
        long_entries = [(t.EntryTime, t.EntryPrice) for t in trades if t.Size > 0]
        short_entries = [(t.EntryTime, t.EntryPrice) for t in trades if t.Size < 0]
        
        if long_entries:
            long_times, long_prices = zip(*long_entries)
            ax2.scatter(long_times, long_prices, color='green', marker='^', s=100, label='Long Entry', zorder=5)
        
        if short_entries:
            short_times, short_prices = zip(*short_entries)
            ax2.scatter(short_times, short_prices, color='red', marker='v', s=100, label='Short Entry', zorder=5)
    
    ax2.set_title(f'Price Chart with Trade Entries - {period_name}', fontsize=14)
    ax2.set_ylabel('Gold Price (USD)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        logger.info(f"Probability difference plot saved to {out_path}")
    
    return fig

def run_backtest(df, strategy_params=None, save_plot=False, plot_filename=None):
    df_bt = df.copy()
    df_bt.rename(
        columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        },
        inplace=True
    )
    df_bt.set_index('timestamp', inplace=True)
    df_bt.index.name = 'Date'
    
    bt = Backtest(
        df_bt,
        MySignalStrategy,
        cash=INITIAL_CAPITAL,
        commission=0,
        margin=(1.0 / MAX_LEVERAGE),
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True
    )
    stats = bt.run()
    
    if save_plot and plot_filename:
        fig = bt.plot(filename=plot_filename)
        logger.info(f"Backtest plot saved to {plot_filename}")
    
    return stats, bt

def save_model_bundle(model, bin_edges, timeframe, prob_threshold, min_rr, features_to_use, output_path):
    model_bundle = {
        "model": model,
        "bin_edges": bin_edges,
        "timeframe": timeframe,
        "prob_threshold": prob_threshold,
        "min_rr": min_rr,
        "features": features_to_use
    }
    joblib.dump(model_bundle, output_path)
    logger.info(f"Model bundle saved to {output_path}")

def objective(trial, df_train, df_val, df_test, bin_edges):
    params = {
        'n_estimators': 600,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03),
        'max_depth': trial.suggest_int('max_depth', 5, 7),
        'num_leaves': trial.suggest_int('num_leaves', 30, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
        'path_smooth': trial.suggest_float('path_smooth', 5, 10),
        'subsample_freq': 3,
        'min_split_gain': 0.01,
        'max_bin': 255,
        'boosting_type': 'gbdt',
        'is_unbalance': False,
        'boost_from_average': True,
        'feature_fraction_seed': 42,
        'bagging_seed': 42
    }
    
    timeframe = TIMEFRAME
    prob_threshold = PROB_THRESHOLD
    min_rr = MIN_RR
    
    trial.set_user_attr('trial_id', trial.number)
    logger.info(f"Starting trial {trial.number} with params: {params}")
    
    train_sample = df_train.sample(frac=0.3, random_state=RANDOM_SEED)
    val_sample = df_val.sample(frac=0.3, random_state=RANDOM_SEED)
    
    initial_model, _ = train_with_selected_features(
        train_sample, 
        val_sample, 
        bin_edges, 
        prune_features(train_sample),
        timeframe, 
        params
    )
    
    pruned_features = prune_features(
        df_train, 
        model=initial_model,
        prune_percent=20
    )
    
    logger.info(f"Training model on 2018-2022 with {len(pruned_features)} features...")
    val_model, selected_features = train_with_selected_features(
        df_train, 
        df_val, 
        bin_edges, 
        pruned_features,
        timeframe, 
        params
    )
    
    y_proba_val, df_val_labeled = predict_proba_with_features(
        df_val, 
        val_model, 
        bin_edges, 
        timeframe, 
        selected_features
    )
    
    df_val_signals = build_signal_dataframe(
        df_val_labeled,
        y_proba_val,
        val_model,
        bin_edges,
        prob_threshold,
        min_rr
    )
    
    stats_val, _ = run_backtest(df_val_signals)
    val_sharpe = stats_val['Sharpe Ratio']
    val_return = stats_val['Return [%]']
    val_max_dd = stats_val['Max. Drawdown [%]']
    val_trades = stats_val['# Trades']
    
    logger.info(f"Validation results: Sharpe={val_sharpe:.2f}, Return={val_return:.2f}%, "
                f"MaxDD={val_max_dd:.2f}%, Trades={val_trades}")
    
    if val_sharpe < 0.5 or val_trades < 5:
        logger.info(f"Validation results too poor, skipping incremental model training")
        return -100.0
    
    logger.info(f"Training incremental model on 2018-2023...")
    incremental_model, incr_features = train_incrementally(
        df_train, 
        df_val, 
        bin_edges, 
        selected_features,
        timeframe, 
        params
    )
    
    y_proba_test, df_test_labeled = predict_proba_with_features(
        df_test, 
        incremental_model, 
        bin_edges, 
        timeframe, 
        incr_features
    )
    
    df_test_signals = build_signal_dataframe(
        df_test_labeled,
        y_proba_test,
        incremental_model,
        bin_edges,
        prob_threshold,
        min_rr
    )
    
    stats_test, _ = run_backtest(df_test_signals)
    test_sharpe = stats_test['Sharpe Ratio']
    test_return = stats_test['Return [%]']
    test_max_dd = stats_test['Max. Drawdown [%]']
    test_trades = stats_test['# Trades']
    
    logger.info(f"Test results: Sharpe={test_sharpe:.2f}, Return={test_return:.2f}%, "
                f"MaxDD={test_max_dd:.2f}%, Trades={test_trades}")
    
    if test_sharpe > 1.0 and test_trades >= 10:
        logger.info(f"Found good model! Saving...")
        os.makedirs("best_models", exist_ok=True)
        model_filename = f"best_models/trial_{trial.number}_sharpe_{test_sharpe:.2f}.pkl"
        save_model_bundle(
            model=incremental_model,
            bin_edges=bin_edges,
            timeframe=timeframe,
            prob_threshold=prob_threshold,
            min_rr=min_rr,
            features_to_use=incr_features,
            output_path=model_filename
        )
    
    trial.set_user_attr('val_sharpe', val_sharpe)
    trial.set_user_attr('val_return', val_return)
    trial.set_user_attr('val_max_dd', val_max_dd)
    trial.set_user_attr('val_trades', val_trades)
    trial.set_user_attr('test_sharpe', test_sharpe)
    trial.set_user_attr('test_return', test_return)
    trial.set_user_attr('test_max_dd', test_max_dd)
    trial.set_user_attr('test_trades', test_trades)
    
    composite_score = test_sharpe - (test_max_dd / 100.0)
    
    if test_trades < 10:
        composite_score -= (10 - test_trades) * 0.1
    
    if test_return <= 0:
        composite_score -= 2.0
    
    return composite_score

if __name__ == "__main__":
    models_dir = "Models"
    charts_dir = "charts"
    best_models_dir = "best_models"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(best_models_dir, exist_ok=True)
    
    logger.info("Fetching data...")
    df_all = fetch_table_data()
    df_all = engineer_features_relative(df_all)
    df_train_global, df_val_global, df_test_2024 = split_data(df_all)
    
    bin_edges = compute_bin_edges(N_DOLLARS, BIN_SIZE_DOLLARS, ALPHA_DOLLARS)
    
    logger.info("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name="gold_trading_strategy"
    )
    
    study.optimize(
        lambda trial: objective(
            trial, 
            df_train_global, 
            df_val_global, 
            df_test_2024, 
            bin_edges
        ),
        n_trials=N_TRIALS,
        timeout=None,
        gc_after_trial=True,
        show_progress_bar=True
    )
    
    logger.info("Optimization completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value (composite score): {study.best_trial.value}")
    logger.info(f"Best parameters: {study.best_trial.params}")
    
    best_val_sharpe = study.best_trial.user_attrs.get('val_sharpe', 'N/A')
    best_test_sharpe = study.best_trial.user_attrs.get('test_sharpe', 'N/A')
    best_test_return = study.best_trial.user_attrs.get('test_return', 'N/A')
    best_test_max_dd = study.best_trial.user_attrs.get('test_max_dd', 'N/A')
    
    logger.info(f"Best trial validation Sharpe: {best_val_sharpe}")
    logger.info(f"Best trial test Sharpe: {best_test_sharpe}")
    logger.info(f"Best trial test Return: {best_test_return}%")
    logger.info(f"Best trial test Max Drawdown: {best_test_max_dd}%")
    
    study_file = os.path.join(models_dir, "optuna_study_results.pkl")
    joblib.dump(study, study_file)
    logger.info(f"Study saved to {study_file}")
    
    plt.figure(figsize=(12, 8))
    
    optimization_history_plot = optuna.visualization.plot_optimization_history(study)
    param_importances_plot = optuna.visualization.plot_param_importances(study)
    parallel_coordinate_plot = optuna.visualization.plot_parallel_coordinate(study)
    
    optimization_history_plot.write_image(os.path.join(charts_dir, "optimization_history.png"))
    param_importances_plot.write_image(os.path.join(charts_dir, "param_importances.png"))
    parallel_coordinate_plot.write_image(os.path.join(charts_dir, "parallel_coordinate.png"))
    
    good_trials = [t for t in study.trials if t.user_attrs.get('test_sharpe', 0) > 1.0]
    if good_trials:
        logger.info(f"Found {len(good_trials)} trials with Sharpe ratio > 1.0")
        
        good_trials.sort(key=lambda t: t.user_attrs.get('test_sharpe', 0), reverse=True)
        
        logger.info("Top 5 trials by Sharpe ratio:")
        for i, trial in enumerate(good_trials[:5]):
            trial_id = trial.number
            sharpe = trial.user_attrs.get('test_sharpe')
            ret = trial.user_attrs.get('test_return')
            dd = trial.user_attrs.get('test_max_dd')
            trades = trial.user_attrs.get('test_trades')
            
            logger.info(f"{i+1}. Trial {trial_id}: Sharpe={sharpe:.2f}, Return={ret:.2f}%, "
                       f"MaxDD={dd:.2f}%, Trades={trades}")
        
        with open(os.path.join(models_dir, "good_trials_report.txt"), "w") as f:
            f.write(f"Trials with Sharpe ratio > 1.0 (Total: {len(good_trials)})\n")
            f.write("="*80 + "\n\n")
            
            for i, trial in enumerate(good_trials):
                trial_id = trial.number
                sharpe = trial.user_attrs.get('test_sharpe')
                ret = trial.user_attrs.get('test_return')
                dd = trial.user_attrs.get('test_max_dd')
                trades = trial.user_attrs.get('test_trades')
                params = trial.params
                
                f.write(f"Trial {trial_id}: Sharpe={sharpe:.2f}, Return={ret:.2f}%, "
                       f"MaxDD={dd:.2f}%, Trades={trades}\n")
                f.write("Parameters:\n")
                for k, v in params.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n" + "-"*50 + "\n\n")
    else:
        logger.info("No trials found with Sharpe ratio > 1.0")
    
    logger.info("Hyperparameter tuning complete!")
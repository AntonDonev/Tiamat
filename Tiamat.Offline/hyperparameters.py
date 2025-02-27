import os
import logging
import warnings
import requests
import pandas as pd
import numpy as np
import ta
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTDB_URL       = "http://localhost:9000/exec"
TABLE_NAME        = "XAUUSD_1M_TRAINING_WITH_FEATURES"
TRAIN_START_DATE  = "2018-01-01"
TRAIN_END_DATE    = "2022-12-31"
VAL_START_DATE    = "2023-01-01"
VAL_END_DATE      = "2023-12-31"
TEST_2024_START   = "2024-01-01"
TEST_2024_END     = "2024-12-31"
INITIAL_CAPITAL    = 10_000
MAX_RISK_PER_TRADE = 0.02
MAX_LEVERAGE       = 50
TRANSACTION_COST   = 0.001
SLIPPAGE           = 0.0005
RANDOM_SEED = 42

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

def engineer_features(df):
    df = df.copy()
    df.sort_values('timestamp', inplace=True)
    
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
    
    boll = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['boll_upper']  = boll.bollinger_hband()
    df['boll_middle'] = boll.bollinger_mavg()
    df['boll_lower']  = boll.bollinger_lband()
    
    df['atr'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()
    
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=5)
    df['adx']     = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()

    obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['tick_volume'])
    df['obv']     = obv.on_balance_volume()
    df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
    
    donch = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=20)
    df['donchian_high'] = donch.donchian_channel_hband()
    df['donchian_low']  = donch.donchian_channel_lband()
    df['donchian_mid']  = donch.donchian_channel_mband()
    
    for lag in range(1, 6):
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    
    df['rolling_mean_10'] = df['close'].rolling(window=10).mean()
    df['rolling_std_10']  = df['close'].rolling(window=10).std()
    
    if 'until_invalid' in df.columns:
        df['is_event_near'] = df['until_invalid'].shift(1) < 1
    else:
        df['is_event_near'] = False

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def compute_bin_edges(n_percent: float,
                      bin_size: float,
                      alpha: float) -> np.ndarray:
    neg_edges = np.arange(-n_percent, -alpha, bin_size)
    if neg_edges.size == 0 or neg_edges[-1] < -alpha:
        neg_edges = np.append(neg_edges, -alpha)

    pos_edges = np.arange(alpha, n_percent + bin_size, bin_size)
    if pos_edges[-1] > n_percent:
        pos_edges[-1] = n_percent

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
    df["future_return"] = (
        (df["close"].shift(-timeframe) - df["close"]) / df["close"]
    )
    df["movement_bin"]  = pd.cut(
        df["future_return"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        right=False
    )
    df.dropna(subset=["movement_bin"], inplace=True)
    df["movement_bin"] = df["movement_bin"].astype(int)
    return df

def train_lgbm_with_val(df_train: pd.DataFrame,
                        df_val: pd.DataFrame,
                        bin_edges: np.ndarray,
                        timeframe: int = 5,
                        params=None):
    df_train_labeled = create_labels(df_train, timeframe, bin_edges)
    unique_train_bins = sorted(df_train_labeled["movement_bin"].unique())
    num_classes = len(unique_train_bins)
    df_val_labeled = create_labels(df_val, timeframe, bin_edges)

    ignore_cols = {"timestamp", "future_return", "movement_bin",
                   "tradable", "GAPFLAG", "is_event_near", "until_invalid"}
    feature_cols = [c for c in df_train_labeled.columns if c not in ignore_cols]

    X_train = df_train_labeled[feature_cols]
    y_train = df_train_labeled["movement_bin"]
    X_val   = df_val_labeled[feature_cols]
    y_val   = df_val_labeled["movement_bin"]

    if params is None:
        params = {
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': 5,
            'n_estimators': 200,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

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
                lgb.log_evaluation(period=100),
                lgb.early_stopping(stopping_rounds=100)
            ]
        )
    except KeyboardInterrupt:
        print("Training interrupted. Exiting gracefully.")
        raise

    assert (model.classes_ == unique_train_bins).all()

    return model

def predict_proba(df_input: pd.DataFrame,
                  model: lgb.LGBMClassifier,
                  bin_edges: np.ndarray,
                  timeframe: int):
    df_input_labeled = create_labels(df_input, timeframe, bin_edges)

    ignore_cols = {"timestamp", "future_return", "movement_bin",
                   "tradable", "GAPFLAG", "is_event_near", "until_invalid"}
    feature_cols = [c for c in df_input_labeled.columns if c not in ignore_cols]
    X_input = df_input_labeled[feature_cols]

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

        if (agg_pos - agg_neg) >= prob_threshold:
            signal = 1
        elif (agg_neg - agg_pos) >= prob_threshold:
            signal = -1
        else:
            signal = 0

        signals.append(signal)
        weighted_pos_moves.append(weighted_pos_move)
        weighted_neg_moves.append(weighted_neg_move)

    df_out["Signal"] = signals

    sl_list = []
    tp_list = []
    for i, sig in enumerate(signals):
        price = df_out["close"].iloc[i]
        if sig == 1:
            tp_distance = weighted_pos_moves[i]
            tp = price * (1 + tp_distance)
            sl = price * (1 - (tp_distance / min_rr))
        elif sig == -1:
            tp_distance = weighted_neg_moves[i]
            tp = price * (1 - tp_distance)
            sl = price * (1 + (tp_distance / min_rr))
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

    capital = strategy.equity
    risk_amount = capital * MAX_RISK_PER_TRADE
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance < 1e-8:
        return 0
    raw_size = risk_amount / sl_distance

    max_size = (capital * MAX_LEVERAGE) / entry_price
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
        else:
            if self.position.is_long:
                if self.current_sl is None or (not np.isnan(sl_signal) and sl_signal > self.current_sl):
                    for trade in self.trades:
                        if trade.sl is None or (not np.isnan(sl_signal) and sl_signal > trade.sl):
                            trade.sl = sl_signal
                    self.current_sl = sl_signal
            elif self.position.is_short:
                if self.current_sl is None or (not np.isnan(sl_signal) and sl_signal < self.current_sl):
                    for trade in self.trades:
                        if trade.sl is None or (not np.isnan(sl_signal) and sl_signal < trade.sl):
                            trade.sl = sl_signal
                    self.current_sl = sl_signal

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
        import matplotlib.pyplot as plt
        fig = bt.plot(filename=plot_filename)
        plt.close(fig)
        logger.info(f"Backtest plot saved to {plot_filename}")
    
    return stats, bt

def save_model_bundle(model, bin_edges, timeframe, prob_threshold, output_path="lgb_model.pkl"):
    model_bundle = {
        "model": model,
        "bin_edges": bin_edges,
        "timeframe": timeframe,
        "prob_threshold": prob_threshold
    }
    import joblib
    joblib.dump(model_bundle, output_path)
    print(f"Model bundle saved to {output_path}")

def objective(trial):
    timeframe = trial.suggest_int('timeframe', 3, 10)
    n_percent = trial.suggest_float('n_percent', 0.01, 0.03)
    bin_size = trial.suggest_float('bin_size', 0.0005, 0.002)
    alpha = bin_size
    bin_edges = compute_bin_edges(n_percent, bin_size, alpha)
    
    lgbm_params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    
    prob_threshold = trial.suggest_float('prob_threshold', 0.5, 0.8)
    min_rr = trial.suggest_float('min_rr', 1.0, 2.0)
    
    try:
        model = train_lgbm_with_val(df_train_global, df_val_global, bin_edges, timeframe, lgbm_params)
        y_proba_val, df_val_labeled = predict_proba(df_val_global, model, bin_edges, timeframe)
        df_val_signals = build_signal_dataframe(
            df_val_labeled,
            y_pred_proba=y_proba_val,
            model=model,
            bin_edges=bin_edges,
            prob_threshold=prob_threshold,
            min_rr=min_rr
        )
        
        stats, _ = run_backtest(df_val_signals)
        
        return_pct = stats['Return [%]']
        max_drawdown = abs(stats['Max. Drawdown [%]'])  
        total_trades = stats['# Trades']
        
        score_a = return_pct  
        score_b = -max_drawdown  
        score_c = total_trades
        
        composite_score = (
            0.6 * score_a +  
            0.3 * score_b +  
            0.1 * score_c   
        )
        
        return composite_score
    except Exception as e:
        logger.error(f"Error in objective function: {e}")
        return -100.0

if __name__ == "__main__":
    import os
    
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)
    
    df_all = fetch_table_data()
    df_all = engineer_features(df_all)
    df_train_global, df_val_global, df_test_2024 = split_data(df_all)
    
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner()
    )
    
    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("Optimization interrupted.")
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    best_params = trial.params
    timeframe = best_params['timeframe']
    n_percent = best_params['n_percent']
    bin_size = best_params['bin_size']
    alpha = bin_size
    bin_edges = compute_bin_edges(n_percent, bin_size, alpha)
    
    lgbm_params = {
        'learning_rate': best_params['learning_rate'],
        'num_leaves': best_params['num_leaves'],
        'max_depth': best_params['max_depth'],
        'n_estimators': best_params['n_estimators'],
        'min_child_samples': best_params['min_child_samples'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree']
    }
    
    prob_threshold = best_params['prob_threshold']
    min_rr = best_params['min_rr']
    
    final_model = train_lgbm_with_val(
        df_train_global, 
        df_val_global, 
        bin_edges, 
        timeframe, 
        lgbm_params
    )
    
    save_model_bundle(
        model=final_model,
        bin_edges=bin_edges,
        timeframe=timeframe,
        prob_threshold=prob_threshold,
        output_path="lgb_model.pkl"
    )
    
    y_proba_val, df_val_labeled = predict_proba(df_val_global, final_model, bin_edges, timeframe)
    df_val_signals = build_signal_dataframe(
        df_val_labeled,
        y_pred_proba=y_proba_val,
        model=final_model,
        bin_edges=bin_edges,
        prob_threshold=prob_threshold,
        min_rr=min_rr
    )
    
    stats_val, bt_val = run_backtest(
        df_val_signals, 
        save_plot=True, 
        plot_filename=os.path.join(results_dir, "validation_backtest_plot.html")
    )
    print("Validation Results for 2023 with optimized parameters:")
    print(stats_val)
    
    y_proba_2024, df_test_2024_labeled = predict_proba(df_test_2024, final_model, bin_edges, timeframe)
    df_test_2024_signals = build_signal_dataframe(
        df_test_2024_labeled,
        y_pred_proba=y_proba_2024,
        model=final_model,
        bin_edges=bin_edges,
        prob_threshold=prob_threshold,
        min_rr=min_rr
    )
    
    stats_2024, bt_2024 = run_backtest(
        df_test_2024_signals,
        save_plot=True,
        plot_filename=os.path.join(results_dir, "test_2024_backtest_plot.html")
    )
    print("Final Test Results for 2024 with optimized parameters:")
    print(stats_2024)
    
    print("\nServer Configuration Values:")
    print(f"PROB_THRESHOLD = {prob_threshold}")
    print(f"BIN_SIZE = {bin_size}")
    print(f"MODEL_BUNDLE_PATH = \"lgb_model.pkl\"")
    print(f"\nBacktest plots saved in directory: {results_dir}")
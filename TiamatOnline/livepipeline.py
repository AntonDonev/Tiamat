import socket
import threading
import time
import logging
import json
import os
import requests
import uuid
from datetime import datetime, timedelta, timezone
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import numpy as np
import ta
import joblib

# --- Configuration ---
HOST = "0.0.0.0"
HTTP_PORT = 8020 # Port for receiving market data (e.g., from MT5)
API_PORT = 8000  # Port for receiving commands (e.g., from ASP.NET)
SOCKET_PORT = 12345 # Port for communicating with MQL4 DLLs

ASPNET_API_URL = "https://tiamat.kzpmg.com/api/python"
API_KEY = "soPibUUmQmYWfCs3IA9BwrjBEI8qQkSeq7wxQP00q7mkw4UBlSV5zekYr3iTqanmVkSUsaapIfc79wWteD6yoOpSUaryh2pUacToU2BaHjyz9tCDQprJMLAPXqb0Marc" # API Key for authenticating requests *to* Python

MODEL_BUNDLE_PATH = "model.pkl"
HIGH_IMPACT_NEWS_PATH = "high_impact_news.csv"
BUFFER_SIZE = 400 # Number of bars to keep for feature calculation

# --- Logging Setup ---
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- XOR Encryption Key and Function ---
# IMPORTANT: This key MUST match the key in the C++ DLL's 'encryption.txt' file
ENCRYPTION_KEY = b"MMgAZWQi788D8238TjqgPgMhx7XYX4CC" # Use bytes for the key.

def xor_cipher(data_bytes, key_bytes):
    """ Simple XOR cipher for bytes. """
    if not key_bytes:
        logger.warning("XOR cipher called with an empty key. Data will not be encrypted/decrypted.")
        return data_bytes
    if not data_bytes:
        return data_bytes
    key_len = len(key_bytes)
    if key_len == 0:
        logger.warning("XOR cipher key_len is 0. Data will not be encrypted/decrypted.")
        return data_bytes
    return bytes([data_bytes[i] ^ key_bytes[i % key_len] for i in range(len(data_bytes))])
# --- End of XOR Encryption ---

# --- Helper Functions ---
def load_high_impact_news_csv(file_path):
    """ Loads and parses high-impact news events from a CSV file. """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"News file not found: {file_path}. Continuing without news data.")
            return []
        df = pd.read_csv(file_path, delimiter=',', header=None)
        df.columns = ['date','time','currency','impact','news','v1','v2','v3','v4','extra']
        # Ensure correct parsing, assuming UTC times in the file
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M', utc=True)
        df = df[df['impact'] == 'H'] # Filter for high impact only
        news_events = df['timestamp'].sort_values().tolist()
        logger.info(f"Loaded {len(news_events)} high-impact news events from CSV (UTC).")
        return news_events
    except Exception as e:
        logger.error(f"Error loading news CSV '{file_path}': {e}", exc_info=True)
        return []

def calc_until_invalid(
    bar_ts, news_events=None, pre_news_buffer=60, post_news_buffer=60,
    maintenance_start=22, maintenance_end=0,
):
    """ Calculates minutes until the next invalid trading period (news or maintenance). """
    if news_events is None: news_events = []
    # Ensure bar_ts is timezone-aware (UTC expected)
    if bar_ts.tzinfo is None:
        logger.warning(f"calc_until_invalid received naive timestamp: {bar_ts}. Assuming UTC.")
        bar_ts = bar_ts.replace(tzinfo=timezone.utc)
    else:
        bar_ts = bar_ts.astimezone(timezone.utc) # Convert to UTC if it's aware but different

    # Calculate current maintenance window
    maint_start_dt = bar_ts.replace(hour=maintenance_start, minute=0, second=0, microsecond=0)
    maint_end_dt = bar_ts.replace(hour=maintenance_end, minute=0, second=0, microsecond=0)
    if maint_end_dt <= maint_start_dt: # Handle overnight maintenance (e.g., 22:00 to 00:00)
        maint_end_dt += timedelta(days=1)

    invalid_windows = [(maint_start_dt, maint_end_dt)]
    # Add news event windows
    for ne in news_events:
        # Ensure news event times are timezone-aware UTC
        if ne.tzinfo is None: ne = ne.replace(tzinfo=timezone.utc)
        else: ne = ne.astimezone(timezone.utc)
        invalid_windows.append((ne - timedelta(minutes=pre_news_buffer), ne + timedelta(minutes=post_news_buffer)))

    invalid_windows.sort(key=lambda x: x[0])

    # Merge overlapping/contiguous windows
    merged = []
    if not invalid_windows: # Should not happen if maintenance is always added
        # Fallback: calculate until next theoretical maintenance start
        next_maint = bar_ts.replace(hour=maintenance_start, minute=0, second=0, microsecond=0)
        if bar_ts >= next_maint: next_maint += timedelta(days=1)
        return max(0, int((next_maint - bar_ts).total_seconds() // 60))

    for win_start, win_end in invalid_windows:
        if not merged or win_start > merged[-1][1]:
            merged.append([win_start, win_end])
        else:
            merged[-1][1] = max(merged[-1][1], win_end)

    # Check current time against merged windows
    for wstart, wend in merged:
        if wstart <= bar_ts < wend: return 0 # Inside an invalid window
        if bar_ts < wstart: # Before the next window
            return max(0, int((wstart - bar_ts).total_seconds() // 60))

    # If past all currently relevant windows, calculate until next maintenance cycle
    next_maint_calc = maint_start_dt
    if bar_ts >= maint_end_dt: # If already past today's maintenance end
        next_maint_calc += timedelta(days=1)
    elif bar_ts >= maint_start_dt: # If inside maintenance (should have returned 0) or between start and end if window spans midnight
        # Find the *next* start, which might be tomorrow
        if bar_ts >= maint_start_dt.replace(hour=maintenance_start, minute=0, second=0, microsecond=0):
            next_maint_calc += timedelta(days=1)

    # Ensure calculation is for the future start time
    while next_maint_calc <= bar_ts:
        next_maint_calc += timedelta(days=1) # Should calculate time until the *next* window starts

    return max(0, int((next_maint_calc - bar_ts).total_seconds() // 60))


def engineer_features(df):
    """ Engineers technical features from OHLCV data. """
    df = df.copy()
    df.sort_values('timestamp', inplace=True)
    df['hour'] = df['timestamp'].dt.hour
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9, fillna=False)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21, fillna=False)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50, fillna=False)
    df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200, fillna=False)
    df['ma_cross_9_21'] = df['ema_9'] / (df['ema_21'] + 1e-9) - 1
    df['ma_cross_50_200'] = df['sma_50'] / (df['sma_200'] + 1e-9) - 1
    df['price_to_ema_9'] = df['close'] / (df['ema_9'] + 1e-9) - 1
    df['price_to_ema_21'] = df['close'] / (df['ema_21'] + 1e-9) - 1
    df['price_to_sma_50'] = df['close'] / (df['sma_50'] + 1e-9) - 1
    df['price_to_sma_200'] = df['close'] / (df['sma_200'] + 1e-9) - 1
    df['ema_9_slope'] = df['ema_9'].pct_change(5)
    df['ema_21_slope'] = df['ema_21'].pct_change(5)
    df['sma_50_slope'] = df['sma_50'].pct_change(10)
    df['sma_200_slope'] = df['sma_200'].pct_change(20)
    df['trend_direction'] = np.sign(df['close'] - df['sma_50'])
    df['trend_direction_change'] = df['trend_direction'].diff().fillna(0).ne(0).astype(int)
    df['trend_streak'] = df.groupby(df['trend_direction_change'].cumsum()).cumcount()
    df['trend_streak'] = np.minimum(df['trend_streak'], 20) # Cap streak
    macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9, fillna=False)
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    df['macd_line_pct'] = df['macd_line'] / (df['close'] + 1e-9) * 100
    df['macd_signal_pct'] = df['macd_signal'] / (df['close'] + 1e-9) * 100
    df['macd_histogram_pct'] = df['macd_histogram'] / (df['close'] + 1e-9) * 100
    psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'], step=0.02, max_step=0.2, fillna=False)
    df['psar'] = psar.psar()
    df['psar_distance'] = (df['close'] - df['psar']) / (df['close'] + 1e-9)
    ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52, fillna=False)
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line() # Kijun Sen
    df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line() # Tenkan Sen
    df['price_to_kijun'] = df['close'] / (df['ichimoku_base'] + 1e-9) - 1
    df['tenkan_kijun_cross'] = df['ichimoku_conv'] / (df['ichimoku_base'] + 1e-9) - 1
    df['cloud_thickness'] = (df['ichimoku_a'] - df['ichimoku_b']) / (df['close'] + 1e-9)
    df['rsi_7'] = ta.momentum.rsi(df['close'], window=7, fillna=False)
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14, fillna=False)
    df['roc_5'] = ta.momentum.roc(df['close'], window=5, fillna=False)
    df['roc_10'] = ta.momentum.roc(df['close'], window=10, fillna=False)
    df['roc_20'] = ta.momentum.roc(df['close'], window=20, fillna=False)
    rsi_scaled = 2 * (df['rsi_14'].fillna(50) - 50) / 100 # Scale RSI to [-1, 1], fill NaN with neutral 50
    df['fisher_rsi_14'] = 0.5 * np.log((1 + rsi_scaled + 1e-9) / (1 - rsi_scaled + 1e-9))
    boll = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2, fillna=False)
    df['boll_pct_b'] = boll.bollinger_pband()
    df['boll_width'] = boll.bollinger_wband()
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14, fillna=False)
    df['atr_pct'] = atr.average_true_range() / (df['close'] + 1e-9) * 100
    adx5 = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=5, fillna=False)
    df['adx5'] = adx5.adx()
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=False)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    df['adx_trend_strength'] = df['adx'].fillna(0) * np.sign(df['adx_pos'].fillna(0) - df['adx_neg'].fillna(0))
    df['di_spread'] = (df['adx_pos'].fillna(0) - df['adx_neg'].fillna(0)) / (df['adx_pos'].fillna(0) + df['adx_neg'].fillna(0) + 1e-9)
    donch = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=20, fillna=False)
    df['donchian_high'] = donch.donchian_channel_hband()
    df['donchian_low'] = donch.donchian_channel_lband()
    df['donchian_mid'] = donch.donchian_channel_mband()
    df['donchian_pos'] = (df['close'] - df['donchian_low']) / (df['donchian_high'] - df['donchian_low'] + 1e-9)
    df['donchian_width'] = (df['donchian_high'] - df['donchian_low']) / (df['donchian_mid'] + 1e-9)
    donch55 = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=55, fillna=False)
    df['donchian_pos_55'] = (df['close'] - donch55.donchian_channel_lband()) / (donch55.donchian_channel_hband() - donch55.donchian_channel_lband() + 1e-9)
    df['donchian_width_55'] = (donch55.donchian_channel_hband() - donch55.donchian_channel_lband()) / (donch55.donchian_channel_mband() + 1e-9)
    if 'tick_volume' in df.columns and not df['tick_volume'].isnull().all():
        df['volume_price_corr'] = df['close'].rolling(window=20).corr(df['tick_volume'])
        df['volume_ratio'] = df['tick_volume'] / (df['tick_volume'].rolling(window=20).mean() + 1e-9)
        obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['tick_volume'], fillna=False)
        df['obv_raw'] = obv.on_balance_volume()
        df['obv_ema'] = df['obv_raw'].ewm(span=20, adjust=False).mean()
        df['obv_change'] = df['obv_raw'].pct_change(20)
        df['obv_slope'] = (df['obv_raw'] - df['obv_raw'].shift(5)) / (df['obv_raw'].shift(5).replace(0, 1e-9) + 1e-9)
    else:
        for col in ['volume_price_corr', 'volume_ratio', 'obv_raw', 'obv_ema', 'obv_change', 'obv_slope']: df[col] = 0.0
    for lag in range(1, 6): df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
    for window in [10, 20, 100]: df[f'price_to_ma_{window}'] = df['close'] / (df['close'].rolling(window=window).mean() + 1e-9) - 1
    df['volatility_10'] = df['close'].rolling(window=10).std() / (df['close'].rolling(window=10).mean() + 1e-9)
    df['volatility_20'] = df['close'].rolling(window=20).std() / (df['close'].rolling(window=20).mean() + 1e-9)
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_20d'] = df['close'].pct_change(20)
    df['high_low_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-9)
    df['open_close_pct'] = (df['close'] - df['open']) / (df['open'] + 1e-9)
    df['pct_from_20d_high'] = df['close'] / (df['high'].rolling(20).max() + 1e-9) - 1
    df['pct_from_20d_low'] = df['close'] / (df['low'].rolling(20).min() + 1e-9) - 1
    rolling_mean10 = df['close'].rolling(window=10).mean()
    rolling_std10 = df['close'].rolling(window=10).std()
    df['z_score_10'] = (df['close'] - rolling_mean10) / (rolling_std10 + 1e-9)
    for window in [20, 50]:
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        df[f'z_score_{window}'] = (df['close'] - rolling_mean) / (rolling_std + 1e-9)
    if 'until_invalid' in df.columns: df['is_event_near'] = (df['until_invalid'].shift(1) < 1).astype(int)
    else: df['is_event_near'] = 0
    df.replace([np.inf, -np.inf], 0, inplace=True) # Replace infs with 0, or consider NaN and dropna
    df.fillna(0, inplace=True) # Fill remaining NaNs with 0 - review if this is appropriate for all features
    # df.dropna(inplace=True) # Alternative: drop rows with any NaNs after calculation
    df.reset_index(drop=True, inplace=True)
    return df


def get_neg_pos_bin_indices(bin_edges: np.ndarray):
    """ Identifies bin indices corresponding to negative and positive moves. """
    neg_indices, pos_indices, num_bins = [], [], len(bin_edges) - 1
    for i in range(num_bins):
        if bin_edges[i+1] <= 0: neg_indices.append(i)
        elif bin_edges[i] >= 0: pos_indices.append(i)
    return neg_indices, pos_indices

def aggregate_signal(proba, price, bin_edges, model_classes, prob_threshold, price_to_sma_50):
    """ Aggregates model probabilities into a final signal with trend filtering. """
    bin_label_to_proba_idx = {label: idx for idx, label in enumerate(model_classes)}
    neg_indices, pos_indices = get_neg_pos_bin_indices(bin_edges)
    valid_neg_bins = [i for i in neg_indices if i in model_classes]
    valid_pos_bins = [i for i in pos_indices if i in model_classes]
    agg_pos_prob, sum_pos_move = 0.0, 0.0
    for bin_label in valid_pos_bins:
        p = proba[bin_label_to_proba_idx[bin_label]]
        agg_pos_prob += p; sum_pos_move += p * ((bin_edges[bin_label] + bin_edges[bin_label+1]) / 2.0)
    agg_neg_prob, sum_neg_move = 0.0, 0.0
    for bin_label in valid_neg_bins:
        p = proba[bin_label_to_proba_idx[bin_label]]
        agg_neg_prob += p; sum_neg_move += p * abs((bin_edges[bin_label] + bin_edges[bin_label+1]) / 2.0)
    exp_pos_mag = sum_pos_move / (agg_pos_prob + 1e-9)
    exp_neg_mag = sum_neg_move / (agg_neg_prob + 1e-9)
    signal, pred_tp, prob_diff = 0, None, agg_pos_prob - agg_neg_prob
    trend_factor = 1.2
    if prob_diff >= prob_threshold: # Potential BUY
        if price_to_sma_50 < 0 and prob_diff < (prob_threshold * trend_factor): logger.info("BUY rejected: Against trend.")
        else: signal, pred_tp = 1, price + exp_pos_mag
    elif -prob_diff >= prob_threshold: # Potential SELL
        if price_to_sma_50 > 0 and (-prob_diff) < (prob_threshold * trend_factor): logger.info("SELL rejected: Against trend.")
        else: signal, pred_tp = -1, price - exp_neg_mag
    return signal, exp_pos_mag, exp_neg_mag, pred_tp


# --- Data Buffer Class ---
class LiveDataBuffer:
    """ Holds recent bar data and calculates 'until_invalid'. """
    def __init__(self, max_size=BUFFER_SIZE, news_events=None):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.news_events = news_events if news_events is not None else []
        self.last_timestamp_processed = None

    def append(self, bar):
        """ Adds a new bar, calculates time until invalid period. """
        try:
            bar_ts_naive = pd.to_datetime(bar['timestamp'].replace('.', '-'))
        except Exception as e: # Catch more general parsing errors
            logger.error(f"Could not parse timestamp: {bar.get('timestamp')}. Error: {e}. Skipping bar.")
            return
        # Assume incoming naive timestamp is UTC, make it timezone-aware
        bar_ts_utc = bar_ts_naive.replace(tzinfo=timezone.utc)

        if bar_ts_utc == self.last_timestamp_processed:
            return # Skip duplicate timestamps

        self.last_timestamp_processed = bar_ts_utc
        until_inv = calc_until_invalid(bar_ts_utc, self.news_events, 1, 30, 22, 0) # Args: pre_news, post_news, maint_start_hr, maint_end_hr
        row = {'timestamp': bar_ts_utc, 'open': float(bar['open']), 'high': float(bar['high']),
               'low': float(bar['low']), 'close': float(bar['close']), 'volume': float(bar['volume']),
               'tick_volume': float(bar['volume']), # Use 'volume' as 'tick_volume' if real volume not separate
               'until_invalid': until_inv}
        self.buffer.append(row)
        # logger.debug(f"Added bar: {row['timestamp']} C={row['close']:.5f}, InvIn={until_inv}m")

    def to_dataframe(self):
        """ Converts the buffer to a Pandas DataFrame. """
        return pd.DataFrame(list(self.buffer))


# --- DLL Socket Server Class ---
class DllSocketServer:
    """ Handles socket communication with MQL4 DLLs, including auth and encryption. """
    def __init__(self, pipeline_trader, host, port, allowed_hwids_map):
        self.pipeline_trader = pipeline_trader
        self.host, self.port = host, port
        self.allowed_hwids_map = allowed_hwids_map # Reference to shared dict
        self.server_socket, self.running = None, True
        self.connections_lock = threading.Lock()
        self.connections = [] # Stores [socket, address, hwid, authenticated_flag]
        self.device_trades_lock = threading.Lock()
        self.device_trades = {} # Stores {hwid: {trade_id: trade_info}}

    def start(self):
        """ Binds the server socket and starts listening threads. """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            logger.info(f"DLL Socket Server listening on {self.host}:{self.port}")
        except OSError as e:
            logger.critical(f"DLL Socket Server FAILED to bind to port {self.port}: {e}")
            self.running = False
            return
        threading.Thread(target=self._accept_loop, daemon=True, name="DLLAcceptThread").start()
        threading.Thread(target=self._recv_loop, daemon=True, name="DLLRecvThread").start()

    def stop(self):
        """ Stops the server and closes all connections. """
        self.running = False
        if self.server_socket:
            try: self.server_socket.shutdown(socket.SHUT_RDWR) # Attempt graceful shutdown
            except OSError: pass # Socket might already be closed
            self.server_socket.close()
            self.server_socket = None
        with self.connections_lock:
            for conn_details in self.connections:
                sock = conn_details[0]
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                sock.close()
            self.connections.clear()
        logger.info("DLL Socket Server stopped and connections closed.")

    def _accept_loop(self):
        """ Accepts incoming client connections. """
        while self.running and self.server_socket:
            try:
                conn, addr = self.server_socket.accept()
                logger.info(f"DLL connection attempt from {addr[0]}:{addr[1]}")
                with self.connections_lock:
                    self.connections.append([conn, addr, None, False]) # Add connection initially unauthenticated
            except OSError: # Handles socket closing during shutdown
                if self.running: logger.warning("Accept loop error or socket closed.")
                break
            except Exception as e:
                if self.running: logger.error(f"Unexpected accept error: {e}", exc_info=True)
                break
        logger.info("DLL accept loop finished.")

    def _remove_connection(self, index_to_remove):
        """ Safely removes a connection from the list and closes its socket. """
        with self.connections_lock:
            if 0 <= index_to_remove < len(self.connections):
                conn, addr, hwid, _ = self.connections.pop(index_to_remove)
                logger.info(f"Removed connection from {addr[0]} (HWID: {hwid}).")
                try: conn.close()
                except OSError: pass
                # Optionally clear trades associated with this HWID upon disconnect
                # with self.device_trades_lock:
                #     if hwid and hwid in self.device_trades:
                #         logger.info(f"Clearing tracked trades for disconnected HWID: {hwid}")
                #         del self.device_trades[hwid]
                return True
        return False

    def _recv_loop(self):
        """ Handles receiving data from all connected clients. """
        while self.running:
            indices_to_remove = []
            # Iterate safely using index for removal later
            with self.connections_lock:
                for i in range(len(self.connections) - 1, -1, -1):
                    conn_details = self.connections[i]
                    conn, addr, current_hwid, is_authenticated = conn_details
                    client_ip = addr[0]
                    try:
                        conn.settimeout(0.01) # Non-blocking check
                        data = conn.recv(1024)
                        if data:
                            # Assume DLL sends plain text, Python encrypts outgoing only
                            # If DLL also encrypts, decryption needed here:
                            # data = xor_cipher(data, ENCRYPTION_KEY) # Example if DLL encrypts
                            msg = data.decode('utf-8', errors='replace').strip()
                            logger.info(f"Socket RECV from {client_ip} (HWID: {current_hwid}, Auth: {is_authenticated}): {msg}")

                            if not is_authenticated:
                                if msg.startswith("AUTH|HWID="):
                                    try:
                                        received_hwid = msg.split("=", 1)[1]
                                        # Check against the dynamically updated map
                                        if received_hwid in self.pipeline_trader.allowed_devices:
                                            conn_details[2] = received_hwid # Set HWID
                                            conn_details[3] = True      # Set authenticated
                                            current_hwid = received_hwid # Update local scope var
                                            is_authenticated = True  # Update local scope var
                                            with self.device_trades_lock: # Ensure dict exists for hwid
                                                if current_hwid not in self.device_trades:
                                                    self.device_trades[current_hwid] = {}
                                            acc_id = self.pipeline_trader.allowed_devices[received_hwid]
                                            logger.info(f"HWID {received_hwid} authenticated for AccountID {acc_id} from IP {client_ip}.")
                                        else:
                                            logger.warning(f"Auth FAIL: HWID '{received_hwid}' from {client_ip} not in allowed devices. Closing.")
                                            indices_to_remove.append(i)
                                    except Exception as e: # Catch potential index errors etc.
                                        logger.error(f"Error processing AUTH msg '{msg}' from {client_ip}: {e}. Closing.", exc_info=True)
                                        indices_to_remove.append(i)
                                else:
                                    logger.warning(f"First msg from {client_ip} not AUTH: '{msg}'. Closing.")
                                    indices_to_remove.append(i)
                            else: # Already authenticated, process trade confirmations etc.
                                if msg.startswith("OPEN_CONFIRM|"):
                                    # Extract trade ID (Python ID) and potentially MQL ticket
                                    py_id = ""
                                    for part in msg.split('|'):
                                        if part.startswith("ID="):
                                            py_id = part.split("=",1)[1]
                                            break
                                    if py_id:
                                        with self.device_trades_lock:
                                            if current_hwid in self.device_trades:
                                                # Store confirmation details, maybe link to original trade_pkg if needed
                                                self.device_trades[current_hwid][py_id] = {"id": py_id, "status": "open_confirmed", "confirmed_at": datetime.now(timezone.utc).isoformat()}
                                                logger.info(f"Trade ID={py_id} OPEN_CONFIRMED for HWID={current_hwid}.")
                                            else: logger.warning(f"Received OPEN_CONFIRM for HWID {current_hwid} which isn't tracked?")
                                    else: logger.warning(f"Could not parse ID from OPEN_CONFIRM: {msg}")

                                elif msg.startswith("CLOSED_CONFIRM|"):
                                    py_id = ""
                                    for part in msg.split('|'):
                                        if part.startswith("ID="):
                                            py_id = part.split("=",1)[1]
                                            break
                                    if py_id:
                                        with self.device_trades_lock:
                                            # Remove from active device trades upon confirmation
                                            if current_hwid in self.device_trades and py_id in self.device_trades[current_hwid]:
                                                del self.device_trades[current_hwid][py_id]
                                                logger.info(f"Trade ID={py_id} CLOSED_CONFIRMED and removed for HWID={current_hwid}.")
                                            else:
                                                logger.info(f"Trade ID={py_id} CLOSED_CONFIRMED for HWID={current_hwid}, but not in local device track.")
                                        # Notify main trader logic about the closure confirmation
                                        self.pipeline_trader.on_position_closed(py_id, "CLOSED_CONFIRM_DLL")
                                    else: logger.warning(f"Could not parse ID from CLOSED_CONFIRM: {msg}")

                                elif msg.startswith("EDIT_CONFIRM|"):
                                    logger.info(f"Received EDIT_CONFIRM from HWID={current_hwid}: {msg}")
                                    # Potentially update internal state if needed based on confirmation

                                # Forward relevant messages to ASP.NET
                                self.pipeline_trader.forward_to_aspnet(msg, client_ip, current_hwid)

                        elif not data: # Client closed connection cleanly
                            logger.info(f"Connection closed by {client_ip} (HWID: {current_hwid}).")
                            indices_to_remove.append(i)

                    except socket.timeout: continue # Normal for non-blocking recv
                    except (ConnectionResetError, BrokenPipeError, OSError) as e:
                        logger.warning(f"Socket ERR for {client_ip} (HWID: {current_hwid}): {e}. Marking for removal.")
                        indices_to_remove.append(i)
                    except Exception as e:
                        logger.error(f"Unexpected RECV loop ERR for {client_ip} (HWID: {current_hwid}): {e}. Marking for removal.", exc_info=True)
                        indices_to_remove.append(i)

            # Remove disconnected clients outside the loop iteration
            if indices_to_remove:
                # Sort indices descending to avoid shifting issues during removal
                for index in sorted(list(set(indices_to_remove)), reverse=True):
                    self._remove_connection(index)

            time.sleep(0.05) # Small sleep to prevent high CPU usage
        logger.info("DLL receive loop finished.")

    def send_signal_to_clients(self, message: str):
        """ Encrypts and sends a message to all authenticated DLL clients. """
        indices_to_remove = []
        with self.connections_lock:
            if not any(auth for _, _, _, auth in self.connections):
                # logger.warning("No authenticated DLL clients to send signal to.") # Can be noisy
                return

            # Encrypt the message using XOR cipher
            encrypted_message_bytes = xor_cipher(message.encode('utf-8'), ENCRYPTION_KEY)
            if not encrypted_message_bytes and message: # Check if encryption failed
                logger.error(f"XOR Encryption failed for message: {message}. Check key. Aborting send.")
                return

            logger.debug(f"Plain message: '{message}', Encrypted length: {len(encrypted_message_bytes)}")

            for i, conn_details in enumerate(self.connections):
                conn, addr, hwid, authenticated = conn_details
                if authenticated:
                    try:
                        conn.sendall(encrypted_message_bytes)
                        # Log the plain message for server readability, but indicate encryption
                        logger.info(f"Sent (encrypted) to {addr[0]} (HWID: {hwid}): {message}")
                    except (ConnectionResetError, BrokenPipeError, OSError) as e:
                        logger.error(f"Send error to {addr[0]} (HWID: {hwid}): {e}. Marking for removal.")
                        indices_to_remove.append(i)
                    except Exception as e:
                        logger.error(f"Unexpected send error to {addr[0]} (HWID: {hwid}): {e}. Marking for removal.", exc_info=True)
                        indices_to_remove.append(i)

        # Process removals outside the main connections_lock critical section
        if indices_to_remove:
            for index in sorted(list(set(indices_to_remove)), reverse=True):
                self._remove_connection(index)

    def get_device_trades(self, hwid=None):
        """ Returns a copy of the tracked trades for a specific HWID or all HWIDs. """
        with self.device_trades_lock:
            if hwid:
                return self.device_trades.get(hwid, {}).copy() # Return copy of specific HWID's trades
            return self.device_trades.copy() # Return copy of the entire dictionary


# --- API Request Handler Class ---
class ApiRequestHandler(BaseHTTPRequestHandler):
    """ Handles incoming HTTP requests for commands (START, EDIT, OPEN, CLOSE). """
    pipeline_trader = None # Static variable shared across handlers

    def log_message(self, format, *args): pass # Suppress default logging

    def send_json_response(self, status_code, data):
        """ Sends a JSON response with appropriate headers. """
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*') # CORS header
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def check_api_key(self):
        """ Validates the x-api-key header. """
        api_key = self.headers.get('x-api-key')
        if not api_key or api_key != API_KEY:
            logger.warning(f"Unauthorized API access from {self.client_address[0]}. Path: {self.path}. Key: '{api_key}'")
            self.send_json_response(401, {"error": "Unauthorized"})
            return False
        return True

    def do_OPTIONS(self):
        """ Handles CORS preflight requests. """
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-API-Key')
        self.end_headers()

    def do_GET(self):
        """ Handles GET requests (currently only /health). """
        if self.path == '/health':
            self.send_json_response(200, {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})
        else:
            self.send_json_response(404, {"error": "Endpoint not found"})

    def do_POST(self):
        """ Handles POST requests for /command and legacy /edit. """
        if not self.check_api_key(): return
        content_length = int(self.headers.get('Content-Length', 0));
        if content_length == 0: self.send_json_response(400, {"error": "Empty request body"}); return
        post_data_bytes = self.rfile.read(content_length)
        try:
            json_data = json.loads(post_data_bytes.decode('utf-8'))
            logger.info(f"API POST to {self.path} from {self.client_address[0]}: {json.dumps(json_data)}")
        except Exception as e:
            logger.error(f"Invalid JSON or decode error from {self.client_address[0]}: {e}", exc_info=True)
            self.send_json_response(400, {"error": "Invalid JSON format"})
            return

        if not ApiRequestHandler.pipeline_trader:
            logger.error("ApiRequestHandler.pipeline_trader is not set. Cannot process command.")
            self.send_json_response(503, {"error": "Service temporarily unavailable - trader not initialized."})
            return

        try:
            if self.path == '/command':
                command = json_data.get('command')
                if command == 'START':
                    hwid = json_data.get('hwid')
                    account_id = str(json_data.get('accountId', 'UNKNOWN_ACCOUNT'))
                    if not hwid:
                        self.send_json_response(400, {"error": "HWID is required for START."})
                        return
                    # Update the shared allowed_devices map in pipeline_trader
                    ApiRequestHandler.pipeline_trader.allowed_devices[hwid] = account_id
                    logger.info(f"API: Registered/Updated HWID={hwid} for AccountID={account_id}")
                    self.send_json_response(200, {"status": "ok", "message": f"HWID {hwid} registered for Account ID {account_id}"})

                elif command == 'EDIT':
                    # --- EDIT Command Correction for MQL4 ---
                    # Extract specific fields expected by MQL4: accountId, maxRisk, untradablePeriod
                    account_id_str = str(json_data.get('accountId', '0')) # Default to '0' if missing
                    max_risk_str = str(json_data.get('maxRisk', '1.5')) # Default if missing
                    untradable_period_str = str(json_data.get('untradablePeriod', '60')) # Default if missing

                    # Construct message in the exact format MQL4 expects: EDIT|accountId|maxRisk|untradablePeriod
                    msg_to_dll = f"EDIT|{account_id_str}|{max_risk_str}|{untradable_period_str}"
                    # --- End of Correction ---
                    logger.info(f"API: Sending {command.upper()} to DLLs: {msg_to_dll}")
                    ApiRequestHandler.pipeline_trader.send_signal(msg_to_dll)
                    self.send_json_response(200, {"status": "ok", "message": f"{command.upper()} command sent to DLLs."})

                elif command in ['OPEN', 'CLOSE']:
                    # Build message generically, but be mindful of MQL expectations if using API for OPEN
                    params_list = []
                    if command == 'OPEN':
                        # Format for OPEN needs to match MQL expectation if API is used to open trades:
                        # OPEN|ID=val|TYPE|SYMBOL|TP=val|MINUTESUNTILINVALID=val|API_ID=val
                        # We need specific keys in the JSON for this.
                        py_id = json_data.get('ID', f'API_OPEN_{int(time.time())}') # API needs to send ID
                        ttype = json_data.get('TYPE', 'BUY')
                        symbol = json_data.get('SYMBOL', 'XAUUSD')
                        tp = json_data.get('TP', '0.0')
                        mins = json_data.get('MINUTESUNTILINVALID', '0')
                        api_id = json_data.get('API_ID', py_id) # API should provide this for mapping
                        # Ensure required fields are present?
                        if tp == '0.0':
                            logger.error("API OPEN command missing TP value.")
                            self.send_json_response(400, {"error": "TP value required for OPEN command via API"})
                            return
                        # Build specific MQL OPEN format
                        msg_to_dll = f"OPEN|ID={py_id}|{ttype}|{symbol}|TP={tp}|MINUTESUNTILINVALID={mins}|API_ID={api_id}"
                    elif command == 'CLOSE':
                        # CLOSE|ID=val|REASON=val|... (MQL doesn't handle this input)
                        # Build generically for now
                        params_list = [f"{k.upper()}={v}" for k,v in json_data.items() if k != 'command']
                        msg_to_dll = f"{command.upper()}|{'|'.join(params_list)}"

                    logger.info(f"API: Sending {command.upper()} to DLLs: {msg_to_dll}")
                    ApiRequestHandler.pipeline_trader.send_signal(msg_to_dll)
                    self.send_json_response(200, {"status": "ok", "message": f"{command.upper()} command sent to DLLs."})

                else:
                    self.send_json_response(400, {"error": f"Unknown command: {command}"})

            elif self.path == '/edit': # Legacy endpoint
                logger.warning(f"API: Deprecated /edit endpoint used by {self.client_address[0]}. Use /command.")
                # --- Legacy EDIT Correction for MQL4 ---
                account_id_str = str(json_data.get('account_id', '0')) # Use lowercase keys
                max_risk_str = str(json_data.get('max_risk', '1.5'))
                untradable_period_str = str(json_data.get('untradable_period', '60'))
                msg_to_dll = f"EDIT|{account_id_str}|{max_risk_str}|{untradable_period_str}"
                # --- End of Correction ---
                logger.info(f"API /edit: Sending EDIT command to DLLs: {msg_to_dll}")
                ApiRequestHandler.pipeline_trader.send_signal(msg_to_dll)
                self.send_json_response(200, {"status": "ok", "message": "EDIT (legacy) command sent."})
            else:
                self.send_json_response(404, {"error": "API Endpoint not found"})

        except Exception as e:
            logger.error(f"Error processing API command '{json_data.get('command')}' at '{self.path}': {e}", exc_info=True)
            self.send_json_response(500, {"error": f"Internal server error processing command."})


# --- Bar Data Request Handler Class ---
class BarsRequestHandler(BaseHTTPRequestHandler):
    """ Handles incoming HTTP POST requests containing market bar data. """
    pipeline_trader = None # Static variable

    def log_message(self, format, *args): pass # Suppress default logging

    def do_POST(self):
        """ Processes POST request, decodes JSON bar data, passes to trader. """
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"Empty request body"}')
            return
        post_data_bytes = self.rfile.read(content_length)
        try:
            json_data = json.loads(post_data_bytes.decode('utf-8'))
            # Expect keys like "time", "open", "high", "low", "close", "volume"
            bar_data = {"timestamp": json_data.get("time", ""), "open": json_data.get("open", 0.0),
                        "high": json_data.get("high", 0.0), "low": json_data.get("low", 0.0),
                        "close": json_data.get("close", 0.0), "volume": json_data.get("volume", 0.0)}

            if not bar_data["timestamp"]: # Basic validation
                logger.error(f"Received bar data with missing timestamp from {self.client_address[0]}.")
                self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"Missing timestamp"}'); return

            if BarsRequestHandler.pipeline_trader:
                BarsRequestHandler.pipeline_trader.on_new_bar(bar_data)
            else:
                logger.error("BarsRequestHandler.pipeline_trader not set. Cannot process new bar.")
                self.send_response(503); self.end_headers(); self.wfile.write(b'{"error":"Service unavailable - trader not initialized"}')
                return

            self.send_response(200); self.send_header('Content-type', 'application/json'); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON bar data from {self.client_address[0]}: {post_data_bytes.decode('utf-8', errors='ignore')}")
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"Invalid JSON format for bar data"}')
        except Exception as e:
            logger.error(f"Error in BarsRequestHandler POST from {self.client_address[0]}: {e}", exc_info=True)
            self.send_response(500); self.end_headers(); self.wfile.write(b'{"error":"Internal server error processing bar data"}')


# --- ASP.NET API Client Class ---
class AspNetApiClient:
    """ Handles communication with the ASP.NET backend API. """
    def __init__(self, api_url, api_key):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'PythonTradingSystemClient/1.2', # Updated agent
            'Accept': '*/*',
            'Cache-Control': 'no-cache',
            'x-api-key': self.api_key # Send API key in header
        })

    def _send_request(self, method, endpoint, data=None, timeout=10):
        """ Sends a request to the specified API endpoint. """
        url = f"{self.api_url}/{endpoint}"
        try:
            payload_str = json.dumps(data) if data else None
            response = self.session.request(method, url, data=payload_str, timeout=timeout)
            response_text_preview = response.text[:200] if response.text else "N/A"
            if 200 <= response.status_code < 300:
                logger.info(f"ASP.NET API {method} {endpoint} OK ({response.status_code}). Resp: {response_text_preview}")
                try: return True, response.json() if response.content else {}
                except json.JSONDecodeError: logger.warning(f"ASP.NET API sent non-JSON 2xx response: {response_text_preview}"); return True, {}
            else:
                logger.warning(f"ASP.NET API {method} {endpoint} FAILED ({response.status_code}). Resp: {response_text_preview}")
                return False, {"error_code": response.status_code, "details": response_text_preview}
        except requests.exceptions.Timeout: logger.error(f"ASP.NET API {method} {endpoint} TIMEOUT."); return False, {"error": "timeout"}
        except requests.exceptions.RequestException as e: logger.error(f"ASP.NET API {method} {endpoint} RequestException: {e}"); return False, {"error": str(e)}
        except Exception as e: logger.error(f"Unexpected error during ASP.NET API call: {e}", exc_info=True); return False, {"error": "unexpected", "details": str(e)}

    def send_open_confirm(self, message_parts, client_ip, hwid=None):
        """ Sends trade open confirmation details to the ASP.NET API. """
        try:
            # MQL sends: OPEN_CONFIRM|ID=PY_xxx|TYPE|SYMBOL|SIZE|RISK_PCT|DATE_STR|TICKET_STR
            py_trade_id, trade_type, symbol, size_str, mql_ticket_str = "", "", "", "0.0", ""
            opened_at_mql_str = ""

            for part in message_parts:
                if part.startswith("ID="): py_trade_id = part.split("=",1)[1]
            if len(message_parts) > 2: trade_type = message_parts[2]
            if len(message_parts) > 3: symbol = message_parts[3]
            if len(message_parts) > 4: size_str = message_parts[4]
            # parts[5] is MQL risk_pct - not typically sent to backend confirm
            if len(message_parts) > 6: opened_at_mql_str = message_parts[6]
            if len(message_parts) > 7: mql_ticket_str = message_parts[7] # MQL Ticket ID

            opened_at_iso = datetime.now(timezone.utc).isoformat()
            if opened_at_mql_str:
                try: opened_at_iso = pd.to_datetime(opened_at_mql_str, format='%Y.%m.%d %H:%M').tz_localize(None).tz_localize('UTC').isoformat()
                except Exception as e: logger.warning(f"Could not parse opened_at '{opened_at_mql_str}'. Error: {e}")

            payload = {"id": py_trade_id, "symbol": symbol, "type": trade_type, "size": float(size_str),
                       "openedAt": opened_at_iso, "fromIp": client_ip or "N/A", "hwid": hwid or "N/A",
                       "internalDllId": mql_ticket_str } # Send MQL ticket as internalDllId
            logger.info(f"Forwarding OPEN_CONFIRM to ASP.NET: {payload}")
            success, _ = self._send_request("POST", "open-confirm", data=payload)
            return success
        except Exception as e: logger.error(f"Error formatting/sending OPEN_CONFIRM: {e}", exc_info=True); return False

    def send_closed_confirm(self, message_parts, client_ip, hwid=None):
        """ Sends trade close confirmation details to the ASP.NET API. """
        try:
            # MQL sends: CLOSED_CONFIRM|ID=PY_xxx|PROFIT|CURRENTCAPITAL|DATE_STR|ACCOUNT_ID_MQL|TICKET_STR
            py_trade_id, profit_str, capital_str, closed_at_mql_str, mql_ticket_str = "", "0.0", "0.0", "", ""

            for part in message_parts:
                if part.startswith("ID="): py_trade_id = part.split("=",1)[1]
            if len(message_parts) > 2: profit_str = message_parts[2]
            if len(message_parts) > 3: capital_str = message_parts[3]
            if len(message_parts) > 4: closed_at_mql_str = message_parts[4]
            # parts[5] is MQL account ID
            if len(message_parts) > 6: mql_ticket_str = message_parts[6] # MQL Closing Deal Ticket

            closed_at_iso = datetime.now(timezone.utc).isoformat()
            if closed_at_mql_str:
                try: closed_at_iso = pd.to_datetime(closed_at_mql_str, format='%Y.%m.%d %H:%M').tz_localize(None).tz_localize('UTC').isoformat()
                except Exception as e: logger.warning(f"Could not parse closed_at '{closed_at_mql_str}'. Error: {e}")

            payload = {"id": py_trade_id, "profit": float(profit_str), "currentCapital": float(capital_str),
                       "closedAt": closed_at_iso, "fromIp": client_ip or "N/A", "hwid": hwid or "N/A",
                       "internalDllId": mql_ticket_str } # Send MQL closing deal ticket
            logger.info(f"Forwarding CLOSED_CONFIRM to ASP.NET: {payload}")
            success, _ = self._send_request("POST", "closed-confirm", data=payload)
            return success
        except Exception as e: logger.error(f"Error formatting/sending CLOSED_CONFIRM: {e}", exc_info=True); return False

    def check_health(self):
        """ Checks the health endpoint of the ASP.NET API. """
        logger.info(f"Checking ASP.NET API health at {self.api_url}/health")
        success, response_data = self._send_request("GET", "health", timeout=5)
        if success: logger.info(f"ASP.NET API Health check OK: {response_data}")
        else: logger.error(f"ASP.NET API Health check FAILED. Details: {response_data}")
        return success


# --- Main Trading Logic Class ---
class LivePipelineTrader:
    """ Orchestrates data buffering, feature engineering, model prediction, and trade signaling. """
    def __init__(self, model_bundle_path):
        self.model_bundle = self.load_model(model_bundle_path)
        if not self.model_bundle:
            logger.critical(f"Halting: Model bundle '{model_bundle_path}' could not be loaded.")
            raise SystemExit(f"Model bundle load failed: {model_bundle_path}")

        # --- Model and Parameters ---
        self.model = self.model_bundle["model"]
        self.bin_edges = self.model_bundle["bin_edges"]
        self.timeframe = self.model_bundle["timeframe"] # e.g., "M1", "M5"
        self.prob_threshold = self.model_bundle.get("prob_threshold", 0.65) # Confidence threshold
        self.features_to_use = self.model_bundle.get("features", None) # List of feature names
        logger.info(f"Model loaded. TF: {self.timeframe}, ProbThresh: {self.prob_threshold}, Features: {'Specific' if self.features_to_use else 'Auto'}")

        # --- Data and State ---
        self.news_events = load_high_impact_news_csv(HIGH_IMPACT_NEWS_PATH)
        self.data_buffer = LiveDataBuffer(BUFFER_SIZE, self.news_events)
        self.active_trades_lock = threading.Lock()
        self.active_trades = [] # Trades initiated by this Python instance {id: ..., api_id:..., type:..., symbol:..., etc.}
        self.allowed_devices = {} # Shared dict: {hwid: accountId} - populated by API START command

        # --- Communication Components ---
        self.socket_server = DllSocketServer(self, HOST, SOCKET_PORT, self.allowed_devices)
        self.socket_server.start() # Start listening for DLL connections
        self.asp_net_client = AspNetApiClient(ASPNET_API_URL, API_KEY)
        ApiRequestHandler.pipeline_trader = self # Make instance available to API handlers
        BarsRequestHandler.pipeline_trader = self # Make instance available to Bar handlers

        # --- API Server ---
        self.api_httpd = HTTPServer((HOST, API_PORT), ApiRequestHandler)
        logger.info(f"API Command Server listening on {HOST}:{API_PORT}")
        api_server_thread = threading.Thread(target=self.api_httpd.serve_forever, daemon=True, name="ApiServerThread")
        api_server_thread.start()

        # Initial health check for backend
        if not self.asp_net_client.check_health():
            logger.warning("Initial ASP.NET API health check failed. Check connectivity and API key.")

    def load_model(self, model_path):
        """ Loads the trained model bundle from a file. """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            model_bundle = joblib.load(model_path)
            # Validate essential keys
            required_keys = ["model", "bin_edges", "timeframe"]
            if not all(key in model_bundle for key in required_keys):
                logger.error(f"Model bundle from '{model_path}' is missing required key(s): {[k for k in required_keys if k not in model_bundle]}")
                return None
            logger.info(f"Successfully loaded and validated model bundle from {model_path}")
            return model_bundle
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
            return None

    def get_trade_info(self, trade_id):
        """ Retrieves info about a trade initiated by this Python instance. """
        with self.active_trades_lock:
            # Find trade by ID (ensure type consistency, using string comparison)
            return next((trade.copy() for trade in self.active_trades if str(trade.get("id")) == str(trade_id)), None)

    def get_device_trades(self, hwid=None):
        """ Retrieves tracked trades associated with a specific HWID (or all). """
        return self.socket_server.get_device_trades(hwid)

    def on_position_closed(self, trade_id, reason="UNKNOWN"):
        """ Handles notification that a position associated with a trade ID has closed. """
        logger.info(f"Position close event received for Trade ID = {trade_id}, Reason: {reason}")
        with self.active_trades_lock:
            initial_len = len(self.active_trades)
            # Remove the trade from the list of active Python-initiated trades
            self.active_trades = [t for t in self.active_trades if str(t.get("id")) != str(trade_id)]
            if len(self.active_trades) < initial_len:
                logger.info(f"Trade ID={trade_id} removed from Python's active_trades list (reason: {reason}).")
            else:
                logger.info(f"Trade ID={trade_id} (closed via {reason}) was not found in Python's active_trades list (might be API trade or already removed by local TP/SL).")


    def check_existing_trade_conflict(self, signal_type, symbol="XAUUSD"):
        """ Checks if a new signal conflicts with current Python-initiated trades. """
        direction = "BUY" if signal_type == 1 else "SELL"
        with self.active_trades_lock:
            for trade in self.active_trades:
                if trade.get("symbol") == symbol and trade.get("type") == direction:
                    logger.info(f"Signal conflict: Existing {direction} trade ID={trade.get('id')} for {symbol}.")
                    return True # Conflict: Already have a trade of the same type for the symbol
        return False # No conflict found

    def check_tp_sl_hits(self, current_price, symbol="XAUUSD"):
        """
        Checks if any Python-initiated trades hit their TP/SL levels.
        If a hit is detected, the trade is removed from Python's active list,
        and a CLOSE signal is sent to the DLL.
        """
        trades_to_signal_close_cmd = [] # Stores tuples (trade_id, reason, trade_type_str, trade_symbol, close_price_detected)
        trade_ids_to_remove_from_active_list = []

        with self.active_trades_lock:
            for trade in self.active_trades:
                # Check only trades for the relevant symbol (of the current bar)
                # Note: If supporting multiple symbols, 'symbol' param to this function is key
                if trade.get("symbol") != symbol:
                    continue

                tp_hit, sl_hit, reason_for_close = False, False, ""
                trade_type = trade.get("type")
                take_profit = trade.get("take_profit")
                stop_loss = trade.get("stop_loss") # This might be None if MQL calculates SL

                if trade_type == "BUY":
                    if take_profit is not None and current_price >= take_profit:
                        tp_hit, reason_for_close = True, "TP_HIT_PY"
                    elif stop_loss is not None and current_price <= stop_loss:
                        sl_hit, reason_for_close = True, "SL_HIT_PY"
                elif trade_type == "SELL":
                    if take_profit is not None and current_price <= take_profit:
                        tp_hit, reason_for_close = True, "TP_HIT_PY"
                    elif stop_loss is not None and current_price >= stop_loss:
                        sl_hit, reason_for_close = True, "SL_HIT_PY"

                if tp_hit or sl_hit:
                    logger.info(f"Python TP/SL Check: Trade ID={trade['id']} ({trade_type} {trade.get('symbol')}) marked for immediate local removal and DLL close signal due to {reason_for_close} at price {current_price:.5f}.")
                    trade_ids_to_remove_from_active_list.append(trade["id"])
                    # Store all necessary info for sending the command later
                    trades_to_signal_close_cmd.append((trade["id"], reason_for_close, trade_type, trade.get("symbol"), current_price))
            
            # Now, remove the identified trades from the active list
            if trade_ids_to_remove_from_active_list:
                initial_len = len(self.active_trades)
                self.active_trades = [t for t in self.active_trades if t["id"] not in trade_ids_to_remove_from_active_list]
                removed_count = initial_len - len(self.active_trades)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} trade(s) from Python's active_trades list based on local TP/SL detection.")

        # Send CLOSE signals outside the lock
        for trade_id, reason, trade_type_str, trade_symbol, close_price_detected in trades_to_signal_close_cmd:
            close_msg = f"CLOSE|ID={trade_id}|REASON={reason}|SYMBOL={trade_symbol}"
            logger.info(f"Python TP/SL: Signaling close for {trade_type_str} ID={trade_id} ({trade_symbol}) due to {reason} at price {close_price_detected:.5f}. Sending: {close_msg}")
            self.send_signal(close_msg)


    def on_new_bar(self, bar_dict):
        """ Main logic triggered by new bar data. """
        self.data_buffer.append(bar_dict)
        # Need enough data for longest lookback period in feature engineering (e.g., SMA 200)
        if len(self.data_buffer.buffer) < 205: # Increased slightly for safety
            # logger.debug(f"Buffer size {len(self.data_buffer.buffer)} < 205, waiting...")
            return

        df_buf = self.data_buffer.to_dataframe()
        if df_buf.empty: return

        try:
            df_feat = engineer_features(df_buf.copy())
            if df_feat.empty: # Handle case where feature eng returns empty (e.g., all NaNs initially)
                # logger.debug("DataFrame empty after feature engineering, likely insufficient data.")
                return
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}", exc_info=True)
            return

        last_row = df_feat.iloc[-1]
        current_price = last_row["close"]
        current_ts_utc = last_row["timestamp"] # This is already UTC and aware

        # --- Log Processed Bar Details ---
        try:
            # Use .get with defaults for robustness if a feature is missing
            log_msg = (
                f"Bar: time={last_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} "
                f"open={last_row.get('open', 0.0):.5f} high={last_row.get('high', 0.0):.5f} "
                f"low={last_row.get('low', 0.0):.5f} close={current_price:.5f} "
                f"vol={last_row.get('tick_volume', 0.0):.1f} " # Use tick_volume if present
                f"rsi_14={last_row.get('rsi_14', float('nan')):.2f} "
                f"until_invalid={int(last_row.get('until_invalid', -1))}"
            )
            logger.info(log_msg)
        except Exception as e:
            logger.warning(f"Could not log bar details: {e}")
        # --- End Logging ---

        # --- Check for TP/SL hits on existing Python trades ---
        self.check_tp_sl_hits(current_price, symbol="XAUUSD") # Assuming XAUUSD for now

        # --- Check if trading is allowed based on news/maintenance ---
        minutes_to_invalid = last_row.get('until_invalid', 9999)
        if minutes_to_invalid == 0:
            # logger.info(f"Bar at {current_ts_utc}: Currently in invalid period (until_invalid=0). No new trades.")
            return # Don't open new trades during invalid periods
        # Add buffer time before event? e.g. 5 minutes
        buffer_minutes_before_event = 5
        if minutes_to_invalid <= buffer_minutes_before_event:
            logger.info(f"Bar at {current_ts_utc}: Approaching invalid period in {minutes_to_invalid}m (buffer {buffer_minutes_before_event}m). No new trades.")
            return

        # --- Prepare features for model prediction ---
        features_for_model = self.features_to_use
        if not features_for_model: # Fallback if not defined in bundle
            features_for_model = [c for c in df_feat.columns if c not in ['timestamp', 'until_invalid', 'open', 'high', 'low', 'volume', 'tick_volume']]

        missing_model_features = [f for f in features_for_model if f not in last_row.index]
        if missing_model_features:
            logger.error(f"Missing features required by model: {missing_model_features}. Available: {list(last_row.index)}. Skipping prediction.")
            return

        # Ensure data types are correct (float)
        X_last = pd.DataFrame([last_row[features_for_model].to_dict()]).astype(float)
        if X_last.isnull().values.any():
            logger.warning(f"NaN values detected in features for prediction: {X_last.columns[X_last.isnull().any()].tolist()}. Skipping.")
            return

        # --- Get Model Prediction ---
        try:
            proba = self.model.predict_proba(X_last)[0] # Get probabilities for the last row
        except Exception as e:
            logger.error(f"Model predict_proba error: {e}", exc_info=True)
            return

        # --- Aggregate Signal and Filter ---
        price_to_sma50 = last_row.get('price_to_sma_50', 0.0) # Get trend context
        model_signal, expected_pos_mag, expected_neg_mag, _ = aggregate_signal(
            proba, current_price, self.bin_edges, self.model.classes_, self.prob_threshold, price_to_sma50
        )

        final_trade_signal = model_signal # Use the filtered signal directly

        if final_trade_signal == 0:
            # logger.debug(f"No trade signal generated after aggregation/filtering at {current_ts_utc}.")
            return # No trade signal

        # --- Check Trade Conflict ---
        if self.check_existing_trade_conflict(final_trade_signal, symbol="XAUUSD"):
            logger.info(f"Signal ({'BUY' if final_trade_signal == 1 else 'SELL'}) conflicts with existing Python trade. No new trade placed.")
            return

        # --- Prepare and Send Trade Signal ---
        trade_id_py = f"PY_{int(time.time()*1000)}" # Unique Python-generated ID
        trade_type_str = "BUY" if final_trade_signal == 1 else "SELL"
        symbol = "XAUUSD" # Assuming XAUUSD

        # Calculate TP based on model expectation. SL is calculated by MQL EA.
        if final_trade_signal == 1: # BUY
            tp_price = current_price + expected_pos_mag
            sl_price = None # MQL calculates SL
        else: # SELL
            tp_price = current_price - expected_neg_mag
            sl_price = None # MQL calculates SL

        # Store trade details initiated by Python
        trade_package = {
            "id": trade_id_py, "api_id": trade_id_py, # Use same ID for API mapping
            "type": trade_type_str, "symbol": symbol,
            "entry_ts_utc": current_ts_utc.isoformat(), "entry_price": round(current_price, 5),
            "take_profit": round(tp_price, 5),
            "stop_loss": sl_price, # Store None as MQL calculates it
            "minutes_until_invalid": int(minutes_to_invalid)
            # "pending_close_signal_sent" was removed
        }
        with self.active_trades_lock:
            self.active_trades.append(trade_package)

        logger.info(
            f"+++ NEW PYTHON TRADE ({current_ts_utc.strftime('%Y-%m-%d %H:%M:%S')}) +++: {trade_type_str} ID={trade_id_py} ({symbol}), "
            f"Entry={current_price:.5f}, TP={tp_price:.5f} (SL calculated by EA), MinsToInvalid={minutes_to_invalid}"
        )

        # Construct message for MQL4 DLL (matching its expected format)
        # OPEN|ID=<py_id>|<TYPE>|<SYMBOL>|TP=<tp_val>|MINUTESUNTILINVALID=<val>|API_ID=<py_id>
        msg_to_dll = (f"OPEN|ID={trade_id_py}|{trade_type_str}|{symbol}"
                      f"|TP={tp_price:.5f}"
                      f"|MINUTESUNTILINVALID={int(minutes_to_invalid)}"
                      f"|API_ID={trade_id_py}") # Include API_ID for confirmation mapping

        self.send_signal(msg_to_dll) # Send encrypted signal via socket server

    def send_signal(self, msg: str):
        """ Sends a command message to the DLL Socket Server to broadcast. """
        # logger.info(f"Broadcasting command to DLLs: {msg}") # Logging done within socket server send now
        self.socket_server.send_signal_to_clients(msg)

    def forward_to_aspnet(self, msg: str, client_ip=None, hwid=None):
        """ Forwards messages received from DLLs (like confirmations) to the ASP.NET API. """
        message_parts = msg.split("|")
        message_type = message_parts[0] if message_parts else ""

        if message_type == "OPEN_CONFIRM":
            logger.info(f"ASP.NET Forward: Received OPEN_CONFIRM from HWID {hwid}.")
            self.asp_net_client.send_open_confirm(message_parts, client_ip, hwid)
        elif message_type == "CLOSED_CONFIRM":
            logger.info(f"ASP.NET Forward: Received CLOSED_CONFIRM from HWID {hwid}.")
            self.asp_net_client.send_closed_confirm(message_parts, client_ip, hwid)
        elif message_type == "EDIT_CONFIRM":
            logger.info(f"ASP.NET Forward: Received EDIT_CONFIRM from HWID {hwid}. (No specific endpoint call defined).")
            # Add call to self.asp_net_client.send_edit_confirm(...) if needed
        # else: logger.debug(f"Msg type '{message_type}' from HWID {hwid} not forwarded to ASP.NET.")

    def shutdown(self):
        """ Initiates shutdown sequence for all components. """
        logger.info("LivePipelineTrader shutdown initiated...")
        # Stop DLL Socket Server first to prevent new connections/messages
        if hasattr(self, 'socket_server') and self.socket_server:
            self.socket_server.stop()

        # Shutdown API HTTP Server
        if hasattr(self, 'api_httpd') and self.api_httpd:
            logger.info("Stopping API Command Server...")
            try:
                # Shutdown must be from a different thread, but serve_forever runs in one.
                # Closing the server socket might be enough for daemon threads.
                threading.Thread(target=self.api_httpd.shutdown, daemon=True).start()
                self.api_httpd.server_close() # Close socket immediately
            except Exception as e:
                logger.error(f"Error shutting down API HTTPD: {e}")
        logger.info("LivePipelineTrader shutdown sequence completed.")


# --- Main Execution ---
def main():
    """ Starts the trading system components. """
    logger.info("Starting LivePipeline Trading System...")
    pipeline_trader = None
    bars_data_httpd = None

    try:
        # Initialize the main trader logic (loads model, starts DLL server, API client)
        pipeline_trader = LivePipelineTrader(MODEL_BUNDLE_PATH)

        # Start the HTTP server for receiving bar data
        BarsRequestHandler.pipeline_trader = pipeline_trader # Link handler to trader instance
        bars_data_httpd = HTTPServer((HOST, HTTP_PORT), BarsRequestHandler)
        logger.info(f"Bars Data HTTP Server listening on {HOST}:{HTTP_PORT}")
        bars_server_thread = threading.Thread(target=bars_data_httpd.serve_forever, daemon=True, name="BarsServerThread")
        bars_server_thread.start()

        logger.info("All core services started. System is live. Press Ctrl+C to shutdown.")
        # Keep main thread alive while daemon threads run
        while True:
            time.sleep(3600) # Wake up occasionally or just wait for interrupt

    except SystemExit as se: # Raised by LivePipelineTrader init on critical failure
        logger.critical(f"System Exit Triggered: {se}. Shutting down.")
    except KeyboardInterrupt:
        logger.info("Ctrl+C received. Initiating graceful shutdown...")
    except FileNotFoundError as fnf_e:
        logger.critical(f"Essential file not found during startup: {fnf_e}. System cannot start.")
    except OSError as os_e:
        # Port binding errors are caught earlier, this might be other OS issues
        logger.critical(f"OS Error during startup: {os_e}. System cannot start.")
    except Exception as e:
        logger.critical(f"Unhandled critical error in main function: {e}", exc_info=True)
    finally:
        # --- Shutdown Sequence ---
        logger.info("System shutdown sequence commencing...")
        # Shutdown trader logic (stops DLL server, tries to stop API server)
        if pipeline_trader:
            pipeline_trader.shutdown()

        # Shutdown the bars data server
        if bars_data_httpd:
            logger.info("Stopping Bars Data HTTP Server...")
            try:
                # Shutdown must be from a different thread for serve_forever
                threading.Thread(target=bars_data_httpd.shutdown, daemon=True).start()
                bars_data_httpd.server_close() # Close socket
            except Exception as e:
                logger.error(f"Error shutting down Bars Data HTTPD: {e}")

        # Check for remaining threads (useful for debugging hangs)
        time.sleep(1) # Give threads a moment to exit
        active_threads = threading.enumerate()
        main_thread = threading.current_thread()
        other_threads = [t.name for t in active_threads if t != main_thread]
        if other_threads:
            logger.info(f"Remaining active threads: {other_threads}")
        else:
            logger.info("All main threads appear to have stopped.")

        logger.info("System shutdown process complete.")

if __name__ == "__main__":
    main()
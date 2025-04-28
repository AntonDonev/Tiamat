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

HOST = "0.0.0.0"
HTTP_PORT = 8020
API_PORT = 8000

SOCKET_PORT = 12345

ASPNET_API_URL = "https://tiamat.kzpmg.com/api/python"
API_KEY = "soPibUUmQmYWfCs3IA9BwrjBEI8qQkSeq7wxQP00q7mkw4UBlSV5zekYr3iTqanmVkSUsaapIfc79wWteD6yoOpSUaryh2pUacToU2BaHjyz9tCDQprJMLAPXqb0Marc"
ALLOWED_ASPNET_IPS = ["178.169.181.27", "127.0.0.1", "localhost"]

MODEL_BUNDLE_PATH = "model.pkl" 

HIGH_IMPACT_NEWS_PATH = "high_impact_news.csv"  

BUFFER_SIZE = 400

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def find_most_recent_swing_high(df, lookback=100): # <-- Increased lookback to 100
    """
    Find the most recent 5-bar swing high within the lookback period.
    A swing high is defined as a bar with a higher high than the two preceding
    and two succeeding bars.
    Returns the index and the bar data for the middle bar (bar 3) of the swing high.
    """
    for i in range(len(df) - 3, max(len(df) - lookback - 1, 2), -1):
         if i >= 2 and i+2 < len(df):
            if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and
                df.iloc[i]['high'] > df.iloc[i-2]['high'] and
                df.iloc[i]['high'] > df.iloc[i+1]['high'] and
                df.iloc[i]['high'] > df.iloc[i+2]['high']):
                return i, df.iloc[i]
    return None, None

def find_most_recent_swing_low(df, lookback=100): # <-- Increased lookback to 100
    """
    Find the most recent 5-bar swing low within the lookback period.
    A swing low is defined as a bar with a lower low than the two preceding
    and two succeeding bars.
    Returns the index and the bar data for the middle bar (bar 3) of the swing low.
    """
    for i in range(len(df) - 3, max(len(df) - lookback - 1, 2), -1):
         if i >= 2 and i+2 < len(df):
            if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and
                df.iloc[i]['low'] < df.iloc[i-2]['low'] and
                df.iloc[i]['low'] < df.iloc[i+1]['low'] and
                df.iloc[i]['low'] < df.iloc[i+2]['low']):
                return i, df.iloc[i]
    return None, None

def load_high_impact_news_csv(file_path):
    try:
        if not os.path.exists(file_path):
            logger.warning(f"News file not found: {file_path}. Continuing without news data.")
            return []
            
        df = pd.read_csv(file_path, delimiter=',', header=None)
        df.columns = [
            'date','time','currency','impact','news','v1','v2','v3','v4','extra'
        ]
        df['timestamp'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y/%m/%d %H:%M',
            utc=True
        )
        df = df[df['impact'] == 'H']
        news_events = df['timestamp'].sort_values().tolist()
        logger.info(f"Loaded {len(news_events)} high-impact news events from CSV (UTC).")
        return news_events
    except Exception as e:
        logger.error(f"Error loading news CSV: {e}")
        return []

def calc_until_invalid(
    bar_ts,
    news_events=None,
    pre_news_buffer=60,
    post_news_buffer=60,
    maintenance_start=22,
    maintenance_end=0,
):
    if news_events is None:
        news_events = []
    
    maint_start = bar_ts.replace(hour=maintenance_start, minute=0, second=0, microsecond=0)
    maint_end = bar_ts.replace(hour=maintenance_end, minute=0, second=0, microsecond=0)
    if maint_end <= maint_start:
        maint_end += timedelta(days=1)
    
    invalid_windows = [(maint_start, maint_end)]
    logger.debug(f"Maintenance window: {maint_start} -> {maint_end}")
    
    for ne in news_events:
        start_ne = ne - timedelta(minutes=pre_news_buffer)
        end_ne   = ne + timedelta(minutes=post_news_buffer)
        invalid_windows.append((start_ne, end_ne))
    
    invalid_windows.sort(key=lambda x: x[0])
    merged = []
    for win in invalid_windows:
        if not merged:
            merged.append(win)
        else:
            last_start, last_end = merged[-1]
            curr_start, curr_end = win
            if curr_start <= last_end:
                merged[-1] = (last_start, max(last_end, curr_end))
            else:
                merged.append(win)
    
    for wstart, wend in merged:
        if wstart <= bar_ts < wend:
            logger.debug("Current time is INSIDE an invalid window")
            return 0
        if bar_ts < wstart:
            delta = wstart - bar_ts
            minutes_until = max(0, int(delta.total_seconds() // 60))
            logger.debug(f"Minutes until invalid window: {minutes_until}")
            return minutes_until
    
    next_day = bar_ts + timedelta(days=1)
    next_maint_start = next_day.replace(hour=maintenance_start, minute=0, second=0, microsecond=0)
    delta = next_maint_start - bar_ts
    minutes_until = max(0, int(delta.total_seconds() // 60))
    logger.debug(f"Minutes until tomorrow's maintenance: {minutes_until}")
    return minutes_until

def engineer_features(df):
    """Engineer features exactly matching the training script to ensure model compatibility."""
    df = df.copy()
    df.sort_values('timestamp', inplace=True)
    
    df['hour'] = df['timestamp'].dt.hour
    
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    
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
    
    psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'], step=0.02, max_step=0.2)
    df['psar'] = psar.psar()
    
    df['psar_distance'] = (df['close'] - df['psar']) / df['close']
    
    ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    
    df['price_to_kijun'] = df['close'] / df['ichimoku_base'] - 1
    df['tenkan_kijun_cross'] = ichimoku.ichimoku_conversion_line() / df['ichimoku_base'] - 1
    
    df['cloud_thickness'] = (df['ichimoku_a'] - df['ichimoku_b']) / df['close']
    
    df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
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
    
    adx5 = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=5)
    df['adx5'] = adx5.adx()
    
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    df['adx_trend_strength'] = df['adx'] * np.sign(df['adx_pos'] - df['adx_neg'])
    df['di_spread'] = (df['adx_pos'] - df['adx_neg']) / (df['adx_pos'] + df['adx_neg'] + 1e-9)
    
    donch = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=20)
    df['donchian_high'] = donch.donchian_channel_hband()
    df['donchian_low'] = donch.donchian_channel_lband()
    df['donchian_mid'] = donch.donchian_channel_mband()
    df['donchian_pos'] = (df['close'] - df['donchian_low']) / (df['donchian_high'] - df['donchian_low'] + 1e-9)
    df['donchian_width'] = (df['donchian_high'] - df['donchian_low']) / df['donchian_mid']
    
    donch55 = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=55)
    high_band55 = donch55.donchian_channel_hband()
    low_band55 = donch55.donchian_channel_lband()
    mid_band55 = donch55.donchian_channel_mband()
    df['donchian_pos_55'] = (df['close'] - low_band55) / (high_band55 - low_band55 + 1e-9)
    df['donchian_width_55'] = (high_band55 - low_band55) / mid_band55
    
    df['volume_price_corr'] = df['close'].rolling(window=20).corr(df['tick_volume'])
    
    df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(window=20).mean()
    
    obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['tick_volume'])
    df['obv_raw'] = obv.on_balance_volume()
    df['obv_ema'] = df['obv_raw'].ewm(span=20, adjust=False).mean()
    df['obv_change'] = df['obv_raw'].pct_change(20)
    df['obv_slope'] = (df['obv_raw'] - df['obv_raw'].shift(5)) / (df['obv_raw'].shift(5) + 1e-9)
    
    for lag in range(1, 6):
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
    
    for window in [10, 20, 100]:
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

def aggregate_signal(proba, price, bin_edges, model_classes, prob_threshold, price_to_sma_50):
    """
    Aggregate signal generation with trend filter similar to trainfinal.py
    
    Args:
        proba: Probability predictions from the model
        price: Current price
        bin_edges: Bin edges for the model
        model_classes: Classes from the model
        prob_threshold: Probability threshold for signal generation
        price_to_sma_50: Current price relative to 50-day SMA (as a percentage)
        
    Returns:
        signal, weighted_pos_move, weighted_neg_move, tp
    """
    bin_label_to_col = {lbl: idx for idx, lbl in enumerate(model_classes)}
    neg_indices, pos_indices = get_neg_pos_bin_indices(bin_edges)
    
    neg_bins = {}
    for i in neg_indices:
        if i in bin_label_to_col:
            neg_bins[i] = abs(bin_edges[i])
    
    pos_bins = {}
    for j in pos_indices:
        if j in bin_label_to_col:
            pos_bins[j] = bin_edges[j+1]
    
    agg_pos = 0.0
    agg_neg = 0.0
    sum_pos_move = 0.0
    sum_neg_move = 0.0

    for p_bin, move_val in pos_bins.items():
        p_idx = bin_label_to_col[p_bin]
        p_prob = proba[p_idx]
        agg_pos += p_prob
        sum_pos_move += p_prob * move_val

    for n_bin, move_val in neg_bins.items():
        n_idx = bin_label_to_col[n_bin]
        n_prob = proba[n_idx]
        agg_neg += n_prob
        sum_neg_move += n_prob * move_val

    weighted_pos_move = sum_pos_move / agg_pos if agg_pos > 0 else 0
    weighted_neg_move = sum_neg_move / agg_neg if agg_neg > 0 else 0
    
    prob_diff = agg_pos - agg_neg
    
    if prob_diff >= prob_threshold:
        signal = 1
        tp = price + weighted_pos_move
    elif -prob_diff >= prob_threshold:
        signal = -1
        tp = price - weighted_neg_move
    else:
        signal = 0
        tp = None
    
    if signal == -1 and price_to_sma_50 > 0:
        if -prob_diff < (prob_threshold * 1.2):
            logger.info("Short signal rejected due to uptrend (higher threshold required)")
            signal = 0
            tp = None
    
    if signal == 1 and price_to_sma_50 < 0:
        if prob_diff < (prob_threshold * 1.2):
            logger.info("Long signal rejected due to downtrend (higher threshold required)")
            signal = 0
            tp = None

    return signal, weighted_pos_move, weighted_neg_move, tp

class LiveDataBuffer:
    def __init__(self, max_size=BUFFER_SIZE, news_events=None):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.news_events = news_events or []

    def append(self, bar):
        bar_ts = pd.to_datetime(bar['timestamp'])
        bar_ts = bar_ts.replace(tzinfo=timezone.utc) - timedelta(hours=3)
        
        until_inv = calc_until_invalid(
            bar_ts,
            news_events=self.news_events,
            pre_news_buffer=1,
            post_news_buffer=30,
            maintenance_start=22,
            maintenance_end=0,
        )

        row = {
            'timestamp': bar_ts,
            'open': float(bar['open']),
            'high': float(bar['high']),
            'low': float(bar['low']),
            'close': float(bar['close']),
            'volume': float(bar['volume']),
            'tick_volume': float(bar['volume']),
            'until_invalid': until_inv
        }
        self.buffer.append(row)
        logger.debug(f"Added bar: close={row['close']}, until_invalid={until_inv}")

    def to_dataframe(self):
        return pd.DataFrame(list(self.buffer))

class DllSocketServer:
    def __init__(self, pipeline_trader, host, port, allowed_devices):
        self.pipeline_trader = pipeline_trader
        self.host = host
        self.port = port
        self.allowed_devices = allowed_devices

        self.server_socket = None
        self.running = True

        self.connections = []
        self.conn_lock = threading.Lock()
        
        self.device_trades = {}

    def start(self):
        t_accept = threading.Thread(target=self._accept_loop, daemon=True)
        t_accept.start()

        t_recv = threading.Thread(target=self._recv_loop, daemon=True)
        t_recv.start()
        
        logger.info(f"DLL Socket Server started on {self.host}:{self.port}")

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("DLL Socket Server stopped")

    def _accept_loop(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        logger.info(f"DLL Socket Server listening on {self.host}:{self.port}")

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                client_ip = addr[0]
                logger.info(f"DLL connection from {client_ip}:{addr[1]}")

                if client_ip in self.allowed_devices or client_ip == "127.0.0.1" or client_ip == "localhost":
                    account_id = self.allowed_devices.get(client_ip, "TEST")
                    with self.conn_lock:
                        self.connections.append((conn, client_ip))
                        if client_ip not in self.device_trades:
                            self.device_trades[client_ip] = {}
                    logger.info(f"Device IP {client_ip} with Account ID {account_id} connected")
                else:
                    logger.warning(f"IP {client_ip} not allowed, closing connection")
                    conn.close()

            except Exception as e:
                if self.running:
                    logger.error(f"Accept error: {e}")
                break

        logger.info("DLL accept loop stopped")

    def _recv_loop(self):
        while self.running:
            with self.conn_lock:
                for conn, client_ip in list(self.connections):
                    try:
                        conn.settimeout(0.001)
                        data = conn.recv(1024)
                        if data:
                            msg = data.decode('utf-8', errors='replace').strip()
                            logger.info(f"Received from {client_ip}: {msg}")

                            if msg.startswith("OPEN_CONFIRM|"):
                                parts = msg.split("|")
                                if len(parts) >= 2:
                                    tid_str = parts[1].replace("ID=", "")
                                    try:
                                        tid = int(tid_str)
                                        trade_info = self.pipeline_trader.get_trade_info(tid)
                                        if trade_info:
                                            self.device_trades[client_ip][tid] = trade_info
                                            logger.info(f"Added trade ID={tid} to device {client_ip}")
                                        else:
                                            self.device_trades[client_ip][tid] = {
                                                "id": tid,
                                                "confirmed_open": True,
                                                "entry_ts": datetime.now(timezone.utc)
                                            }
                                            logger.info(f"Created new trade ID={tid} for device {client_ip}")
                                    except:
                                        logger.warning(f"Could not parse trade ID from OPEN_CONFIRM: {tid_str}")

                            if msg.startswith("CLOSE|"):
                                parts = msg.split("|")
                                if len(parts) >= 2:
                                    tid_str = parts[1].replace("ID=", "")
                                    try:
                                        tid = int(tid_str)
                                        if tid in self.device_trades.get(client_ip, {}):
                                            logger.info(f"Marking trade ID={tid} for closure for device {client_ip}")
                                            self.device_trades[client_ip][tid]["pending_close"] = True
                                        self.pipeline_trader.on_position_closed(tid, "CLOSE")
                                    except:
                                        logger.warning(f"Could not parse trade ID: {tid_str}")

                            if msg.startswith("CLOSED_CONFIRM|"):
                                parts = msg.split("|")
                                if len(parts) >= 2:
                                    tid_str = parts[1].replace("ID=", "")
                                    try:
                                        tid = int(tid_str)
                                        logger.info(f"Received CLOSED_CONFIRM for trade ID: {tid}")
                                        if tid in self.device_trades.get(client_ip, {}):
                                            del self.device_trades[client_ip][tid]
                                            logger.info(f"Removed trade ID={tid} from device {client_ip}")
                                        self.pipeline_trader.on_position_closed(tid, "CLOSED_CONFIRM")
                                    except:
                                        logger.warning(f"Could not parse trade ID from CLOSED_CONFIRM: {tid_str}")

                            if msg.startswith("OPEN_CONFIRM|") or msg.startswith("CLOSED_CONFIRM|") or msg.startswith("EDIT_CONFIRM|"):
                                logger.info(f"Confirmation received: {msg}")
                                
                            self.pipeline_trader.forward_to_aspnet(msg, client_ip)
                    except socket.timeout:
                        pass
                    except Exception as e:
                        logger.error(f"Receive error: {e}")
                        if client_ip in self.device_trades:
                            logger.info(f"Removing all trades for disconnected device {client_ip}")
                            del self.device_trades[client_ip]
                        self.connections.remove((conn, client_ip))
                        conn.close()
            time.sleep(0.05)

    def send_signal_to_clients(self, message):
        with self.conn_lock:
            if not self.connections:
                logger.warning("No connected clients to send message to!")
                return
                
            for conn, client_ip in list(self.connections):
                try:
                    conn.sendall(message.encode('utf-8'))
                    logger.info(f"Sent to {client_ip}: {message}")
                except Exception as e:
                    logger.error(f"Send error: {e}")
                    if client_ip in self.device_trades:
                        logger.info(f"Removing all trades for disconnected device {client_ip}")
                        del self.device_trades[client_ip]
                    self.connections.remove((conn, client_ip))
                    conn.close()
    
    def get_device_trades(self, client_ip=None):
        if client_ip:
            return self.device_trades.get(client_ip, {})
        return self.device_trades

class ApiRequestHandler(BaseHTTPRequestHandler):
    pipeline_trader = None

    def log_message(self, format, *args):
        pass

    def send_json_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def check_api_key(self):
        api_key = self.headers.get('x-api-key')
        if not api_key or api_key != API_KEY:
            self.send_json_response(401, {"error": "Unauthorized"})
            return False
        return True

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-API-Key')
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self.send_json_response(200, {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        else:
            self.send_json_response(404, {"error": "Endpoint not found"})

    def do_POST(self):
        if not self.check_api_key():
            return

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            json_data = json.loads(post_data.decode('utf-8'))
            
            if self.path == '/command':
                command = json_data.get('command')
                
                if command == 'START':
                    account_id = json_data.get('accountId')
                    allowed_ip = json_data.get('ip')
                    
                    if self.pipeline_trader:
                        self.pipeline_trader.allowed_devices[allowed_ip] = account_id
                        logger.info(f"Updated allowed devices: {allowed_ip} -> {account_id}")
                        self.send_json_response(200, {"status": "ok", "message": "Account started"})
                    else:
                        self.send_json_response(500, {"error": "Pipeline trader not initialized"})
                
                elif command == 'EDIT':
                    account_id = json_data.get('accountId')
                    max_risk = json_data.get('maxRisk')
                    untradable_period = json_data.get('untradablePeriod')
                    
                    msg = f"EDIT|{account_id}|{max_risk}|{untradable_period}"
                    logger.info(f"EDIT command: Account={account_id}, MaxRisk={max_risk}, UntradablePeriod={untradable_period}")
                    
                    if self.pipeline_trader:
                        self.pipeline_trader.send_signal(msg)
                        self.send_json_response(200, {"status": "ok", "message": "Edit command sent"})
                    else:
                        self.send_json_response(500, {"error": "Pipeline trader not initialized"})
                
                elif command == 'OPEN':
                    id = json_data.get('id')
                    type = json_data.get('type')
                    symbol = json_data.get('symbol')
                    tp = json_data.get('tp')
                    minutes_until_invalid = json_data.get('minutesUntilInvalid')
                    
                    msg = f"OPEN|ID={id}|{type}|{symbol}|TP={tp}|MinutesUntilInvalid={minutes_until_invalid}"
                    logger.info(f"Trading command received: {msg}")
                    
                    if self.pipeline_trader:
                        self.pipeline_trader.send_signal(msg)
                        self.send_json_response(200, {"status": "ok", "message": "Open command sent"})
                    else:
                        self.send_json_response(500, {"error": "Pipeline trader not initialized"})
                
                elif command == 'CLOSE':
                    id = json_data.get('id')
                    
                    msg = f"CLOSE|ID={id}"
                    logger.info(f"Trading command received: {msg}")
                    
                    if self.pipeline_trader:
                        self.pipeline_trader.send_signal(msg)
                        self.send_json_response(200, {"status": "ok", "message": "Close command sent"})
                    else:
                        self.send_json_response(500, {"error": "Pipeline trader not initialized"})
                
                else:
                    self.send_json_response(400, {"error": f"Unknown command: {command}"})
            
            elif self.path == '/edit':
                account_id = json_data.get('account_id', '')
                max_risk = json_data.get('max_risk', '')
                untradable_period = json_data.get('untradable_period', '')
                
                edit_msg = f"EDIT|{account_id}|{max_risk}|{untradable_period}"
                
                if self.pipeline_trader:
                    logger.info(f"Received EDIT command via HTTP: {edit_msg}")
                    self.pipeline_trader.send_signal(edit_msg)
                    self.send_json_response(200, {"status": "ok", "message": "Edit command processed successfully"})
                else:
                    self.send_json_response(500, {"error": "Pipeline trader not initialized"})
            
            else:
                self.send_json_response(404, {"error": "Endpoint not found"})
                
        except Exception as e:
            logger.error(f"Error in do_POST: {e}")
            self.send_json_response(400, {"error": str(e)})

class BarsRequestHandler(BaseHTTPRequestHandler):
    pipeline_trader = None  

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            json_data = json.loads(post_data.decode('utf-8'))
            bar_data = {
                "timestamp": json_data.get("time", ""),
                "open": json_data.get("open", 0.0),
                "high": json_data.get("high", 0.0),
                "low": json_data.get("low", 0.0),
                "close": json_data.get("close", 0.0),
                "volume": json_data.get("volume", 0.0)
            }
            
            bar_data["timestamp"] = bar_data["timestamp"].replace('.', '-')

            if self.pipeline_trader:
                self.pipeline_trader.on_new_bar(bar_data)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        except Exception as e:
            logger.error(f"Error in do_POST: {e}")
            self.send_response(400)
            self.end_headers()

class AspNetApiClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'PostmanRuntime/7.32.3',
            'Accept': '*/*',
            'Cache-Control': 'no-cache',
            'Postman-Token': f'{self._generate_token()}',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'x-api-key': self.api_key
        }
    
    def _generate_token(self):
        return str(uuid.uuid4())
    
    def send_open_confirm(self, message_parts, client_ip):
        try:
            id_part = message_parts[1] if len(message_parts) > 1 else ""
            internal_id = id_part.replace("ID=", "").strip()
            
            type = message_parts[2].strip() if len(message_parts) > 2 else ""
            symbol = message_parts[3].strip() if len(message_parts) > 3 else ""
            
            size = message_parts[4].strip() if len(message_parts) > 4 else "0"
            risk = message_parts[5].strip() if len(message_parts) > 5 else "0"
            
            opened_at_str = message_parts[6].strip() if len(message_parts) > 6 else ""
            try:
                if opened_at_str and "." in opened_at_str and ":" in opened_at_str:
                    dt = datetime.strptime(opened_at_str, "%Y.%m.%d %H:%M")
                    opened_at = dt.strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    opened_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            except Exception as e:
                logger.warning(f"Could not parse date '{opened_at_str}': {e}")
                opened_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            api_id = message_parts[7].strip().replace('\x00', '') if len(message_parts) > 7 else internal_id
            
            data = {
                "id": api_id,
                "symbol": symbol,
                "type": type,
                "size": float(size),
                "risk": float(risk),
                "openedAt": opened_at,
                "fromIp": client_ip
            }
            
            logger.info(f"Sending OPEN_CONFIRM to ASP.NET API: {data}")
            
            self.headers['Postman-Token'] = self._generate_token()

            url = f"{self.api_url}/open-confirm"
            
            try:
                logger.info(f"Trying Postman-style request: {url}")
                payload = json.dumps(data)
                response = requests.request("POST", url, headers=self.headers, data=payload)
                
                if response.status_code >= 200 and response.status_code < 300:
                    logger.info(f"SUCCESS with Postman-style request! Response: {response.text}")
                    return True
                else:
                    logger.warning(f"Postman-style request failed. Status: {response.status_code}, Response: {response.text}")
            except Exception as e:
                logger.warning(f"Error with Postman-style request: {e}")
            
            try:
                url_with_key = f"{url}?x-api-key={self.api_key}"
                headers_without_key = self.headers.copy()
                if 'x-api-key' in headers_without_key:
                    del headers_without_key['x-api-key']
                    
                logger.info(f"Trying Postman-style with URL param: {url_with_key}")
                payload = json.dumps(data)
                response = requests.request("POST", url_with_key, headers=headers_without_key, data=payload)
                
                if response.status_code >= 200 and response.status_code < 300:
                    logger.info(f"SUCCESS with URL param! Response: {response.text}")
                    return True
                else:
                    logger.warning(f"URL param approach failed. Status: {response.status_code}, Response: {response.text}")
            except Exception as e:
                logger.warning(f"Error with URL param approach: {e}")
            
            for api_key_location in ['header', 'url', 'both']:
                for content_type in ['application/json', 'application/x-www-form-urlencoded']:
                    try:
                        headers = {
                            'Content-Type': content_type,
                            'User-Agent': 'PostmanRuntime/7.32.3',
                            'Accept': '*/*',
                            'Cache-Control': 'no-cache',
                            'Postman-Token': self._generate_token(),
                            'Accept-Encoding': 'gzip, deflate, br',
                            'Connection': 'keep-alive'
                        }
                        
                        url_to_use = url
                        
                        if api_key_location in ['header', 'both']:
                            headers['x-api-key'] = self.api_key
                            
                        if api_key_location in ['url', 'both']:
                            url_to_use = f"{url}?x-api-key={self.api_key}"
                        
                        logger.info(f"Fallback attempt - API key: {api_key_location}, Content-Type: {content_type}")
                        
                        if content_type == 'application/json':
                            payload = json.dumps(data)
                            response = requests.request("POST", url_to_use, headers=headers, data=payload)
                        else:
                            form_data = {k: str(v) for k, v in data.items()}
                            response = requests.request("POST", url_to_use, headers=headers, data=form_data)
                        
                        if response.status_code >= 200 and response.status_code < 300:
                            logger.info(f"SUCCESS with fallback approach! Response: {response.text}")
                            return True
                        else:
                            logger.warning(f"Fallback approach failed. Status: {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Error with fallback approach: {e}")
            
            logger.error("All approaches failed for OPEN_CONFIRM")
            return False
        except Exception as e:
            logger.error(f"Error sending OPEN_CONFIRM to ASP.NET API: {e}")
            return False
    
    def send_closed_confirm(self, message_parts, client_ip):
        try:
            id_part = message_parts[1] if len(message_parts) > 1 else ""
            internal_id = id_part.replace("ID=", "").strip()
            
            profit = message_parts[2].strip() if len(message_parts) > 2 else "0"
            current_capital = message_parts[3].strip() if len(message_parts) > 3 else "0"
            
            closed_at_str = message_parts[4].strip() if len(message_parts) > 4 else ""
            try:
                if closed_at_str and "." in closed_at_str and ":" in closed_at_str:
                    dt = datetime.strptime(closed_at_str, "%Y.%m.%d %H:%M")
                    closed_at = dt.strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    closed_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            except Exception as e:
                logger.warning(f"Could not parse date '{closed_at_str}': {e}")
                closed_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            api_id = message_parts[6].strip().replace('\x00', '') if len(message_parts) > 6 else internal_id
            
            data = {
                "id": api_id,
                "profit": float(profit),
                "currentCapital": float(current_capital),
                "closedAt": closed_at,
                "fromIp": client_ip
            }
            
            logger.info(f"Sending CLOSED_CONFIRM to ASP.NET API: {data}")
            
            self.headers['Postman-Token'] = self._generate_token()
            
            url = f"{self.api_url}/closed-confirm"
            
            try:
                logger.info(f"Trying Postman-style request: {url}")
                payload = json.dumps(data)
                response = requests.request("POST", url, headers=self.headers, data=payload)
                
                if response.status_code >= 200 and response.status_code < 300:
                    logger.info(f"SUCCESS with Postman-style request! Response: {response.text}")
                    return True
                else:
                    logger.warning(f"Postman-style request failed. Status: {response.status_code}, Response: {response.text}")
            except Exception as e:
                logger.warning(f"Error with Postman-style request: {e}")
            
            try:
                url_with_key = f"{url}?x-api-key={self.api_key}"
                headers_without_key = self.headers.copy()
                if 'x-api-key' in headers_without_key:
                    del headers_without_key['x-api-key']
                    
                logger.info(f"Trying Postman-style with URL param: {url_with_key}")
                payload = json.dumps(data)
                response = requests.request("POST", url_with_key, headers=headers_without_key, data=payload)
                
                if response.status_code >= 200 and response.status_code < 300:
                    logger.info(f"SUCCESS with URL param! Response: {response.text}")
                    return True
                else:
                    logger.warning(f"URL param approach failed. Status: {response.status_code}, Response: {response.text}")
            except Exception as e:
                logger.warning(f"Error with URL param approach: {e}")
            
            for api_key_location in ['header', 'url', 'both']:
                for content_type in ['application/json', 'application/x-www-form-urlencoded']:
                    try:
                        headers = {
                            'Content-Type': content_type,
                            'User-Agent': 'PostmanRuntime/7.32.3',
                            'Accept': '*/*',
                            'Cache-Control': 'no-cache',
                            'Postman-Token': self._generate_token(),
                            'Accept-Encoding': 'gzip, deflate, br',
                            'Connection': 'keep-alive'
                        }
                        
                        url_to_use = url
                        
                        if api_key_location in ['header', 'both']:
                            headers['x-api-key'] = self.api_key
                            
                        if api_key_location in ['url', 'both']:
                            url_to_use = f"{url}?x-api-key={self.api_key}"
                        
                        logger.info(f"Fallback attempt - API key: {api_key_location}, Content-Type: {content_type}")
                        
                        if content_type == 'application/json':
                            payload = json.dumps(data)
                            response = requests.request("POST", url_to_use, headers=headers, data=payload)
                        else:
                            form_data = {k: str(v) for k, v in data.items()}
                            response = requests.request("POST", url_to_use, headers=headers, data=form_data)
                        
                        if response.status_code >= 200 and response.status_code < 300:
                            logger.info(f"SUCCESS with fallback approach! Response: {response.text}")
                            return True
                        else:
                            logger.warning(f"Fallback approach failed. Status: {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Error with fallback approach: {e}")
            
            logger.error("All approaches failed for CLOSED_CONFIRM")
            return False
        except Exception as e:
            logger.error(f"Error sending CLOSED_CONFIRM to ASP.NET API: {e}")
            return False
    
    def check_health(self):
        try:
            self.headers['Postman-Token'] = self._generate_token()
            
            url = f"{self.api_url}/health"
            
            logger.info(f"Health check: {url}")
            response = requests.request("GET", url, headers=self.headers)
            
            if response.status_code == 200:
                logger.info(f"Health check successful with Postman headers")
                return True
                
            url_with_key = f"{url}?x-api-key={self.api_key}"
            headers_without_key = self.headers.copy()
            if 'x-api-key' in headers_without_key:
                del headers_without_key['x-api-key']
                
            logger.info(f"Health check with URL param: {url_with_key}")
            response = requests.request("GET", url_with_key, headers=headers_without_key)
            
            if response.status_code == 200:
                logger.info(f"Health check successful with URL param")
                return True
            
            simple_headers = {
                'Accept': '*/*',
                'x-api-key': self.api_key
            }
            
            logger.info(f"Health check with simple headers: {url}")
            response = requests.request("GET", url, headers=simple_headers)
            
            if response.status_code == 200:
                logger.info(f"Health check successful with simple headers")
                return True
                
            logger.error("Health check failed with all approaches")
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

class LivePipelineTrader:
    def __init__(self, model_bundle_path):
        self.model_bundle = self.load_model(model_bundle_path)
        
        self.model = self.model_bundle["model"]
        self.bin_edges = self.model_bundle["bin_edges"]
        self.timeframe = self.model_bundle["timeframe"]
        self.prob_threshold = self.model_bundle.get("prob_threshold", 0.65)
        self.features_to_use = self.model_bundle.get("features", None)
        
        logger.info(f"Model loaded, timeframe={self.timeframe}, prob_threshold={self.prob_threshold}")
        if self.features_to_use:
            logger.info(f"Model uses {len(self.features_to_use)} specific features")
        
        self.news_events = load_high_impact_news_csv(HIGH_IMPACT_NEWS_PATH)
        
        self.data_buffer = LiveDataBuffer(max_size=BUFFER_SIZE, news_events=self.news_events)
        
        self.active_trades = []
        
        self.account_risk_map = {
            125: 5.0,
            200: 2.0,
        }
        
        self.allowed_devices = {}
        
        self.socket_server = DllSocketServer(self, HOST, SOCKET_PORT, self.allowed_devices)
        self.socket_server.start()
        
        self.asp_net_client = AspNetApiClient(ASPNET_API_URL, API_KEY)
        
        ApiRequestHandler.pipeline_trader = self
        api_server_address = (HOST, API_PORT)
        self.api_httpd = HTTPServer(api_server_address, ApiRequestHandler)
        logger.info(f"API Server listening on port {API_PORT}")
        
        api_server_thread = threading.Thread(target=self.api_httpd.serve_forever, daemon=True)
        api_server_thread.start()
        
        self.asp_net_client.check_health()

    def load_model(self, model_path):
        try:
            model_bundle = joblib.load(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            return model_bundle
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_trade_info(self, trade_id):
        for trade in self.active_trades:
            if trade["id"] == trade_id:
                return trade
        return None
    
    def get_device_trades(self, client_ip=None):
        if hasattr(self, 'socket_server') and self.socket_server:
            return self.socket_server.get_device_trades(client_ip)
        return {}
        
    def on_position_closed(self, trade_id, message_type=None):
        logger.info(f"Position closed: Trade ID={trade_id} via {message_type if message_type else 'unknown'} message")
        
        for t in self.active_trades:
            if t["id"] == trade_id:
                self.active_trades.remove(t)
                logger.info(f"Trade ID={trade_id} removed from active_trades")
                return
        
        if message_type != "CLOSED_CONFIRM":
            logger.warning(f"Trade ID={trade_id} not found in active_trades")
        else:
            logger.info(f"Trade ID={trade_id} confirmation received, trade was already removed")

    def check_existing_trade_conflict(self, signal_type, current_price):
        """
        Check if a new trade signal conflicts with existing active trades
        
        Args:
            signal_type: 1 for buy, -1 for sell
            current_price: current market price
            
        Returns:
            bool: True if there's a conflict, False otherwise
        """
        signal_direction = "BUY" if signal_type == 1 else "SELL"
        
        for trade in self.active_trades:
            if (trade["type"] == "BUY" and signal_type == 1) or (trade["type"] == "SELL" and signal_type == -1):
                logger.info(f"New {signal_direction} signal conflicts with existing {trade['type']} trade ID={trade['id']}")
                return True
                
            if trade["type"] == "BUY":
                trade_low = trade["entry_price"] - (trade["take_profit"] - trade["entry_price"]) * 1.5
                if current_price < trade["take_profit"] and current_price > trade_low:
                    logger.info(f"New {signal_direction} signal within price range of existing BUY trade ID={trade['id']}")
                    return True
            else:
                trade_high = trade["entry_price"] + (trade["entry_price"] - trade["take_profit"]) * 1.5
                if current_price > trade["take_profit"] and current_price < trade_high:
                    logger.info(f"New {signal_direction} signal within price range of existing SELL trade ID={trade['id']}")
                    return True
                    
        return False

    def check_tp_sl_hits(self, current_price):
        """
        Check if any active trades have hit TP or SL
        
        Args:
            current_price: current market price
        """
        trades_to_close = []
        
        for trade in self.active_trades:
            tp_hit = False
            sl_hit = False
            reason = ""
            
            if trade["type"] == "BUY":
                sl_price = trade["entry_price"] - (trade["take_profit"] - trade["entry_price"]) * 1.5
                if current_price >= trade["take_profit"]:
                    tp_hit = True
                    reason = "TP"
                elif current_price <= sl_price:
                    sl_hit = True
                    reason = "SL"
            else:
                sl_price = trade["entry_price"] + (trade["entry_price"] - trade["take_profit"]) * 1.5
                if current_price <= trade["take_profit"]:
                    tp_hit = True
                    reason = "TP"
                elif current_price >= sl_price:
                    sl_hit = True
                    reason = "SL"
            
            if tp_hit or sl_hit:
                trades_to_close.append((trade["id"], reason))
        
        for trade_id, reason in trades_to_close:
            msg = f"CLOSE|ID={trade_id}|{reason}"
            logger.info(f"Closing trade ID={trade_id}, reason={reason}")
            self.send_signal(msg)

    def on_new_bar(self, bar_dict):
        logger.debug(f"New bar received: {bar_dict}")
        self.data_buffer.append(bar_dict)
        
        if len(self.data_buffer.buffer) < 200:
            logger.debug(f"Buffer too small ({len(self.data_buffer.buffer)}), waiting for more data")
            return
        
        df_buf = self.data_buffer.to_dataframe()
        df_feat = engineer_features(df_buf)
        logger.debug(f"Data buffer size={len(df_buf)}, after feature engineering shape={df_feat.shape}")
        
        df_feat.dropna(inplace=True)
        if df_feat.empty:
            logger.warning("Empty DataFrame after dropping NaN values")
            return
        
        last_row = df_feat.iloc[-1]
        current_price = last_row["close"]
        
        self.check_tp_sl_hits(current_price)
        
        swing_high_idx, swing_high_bar = find_most_recent_swing_high(df_feat, lookback=75)
        swing_low_idx, swing_low_bar = find_most_recent_swing_low(df_feat, lookback=75)
        
        current_close = last_row["close"]

        sell_condition_met = False
        buy_condition_met = False
        resistance_trigger_level = None
        support_trigger_level = None

        logger.info("==== TRADE ENTRY CONDITIONS ====")

        if swing_high_bar is not None:
            resistance_trigger_level = swing_high_bar['high']
            sell_condition_met = current_close > resistance_trigger_level
            time_diff = last_row["timestamp"] - swing_high_bar["timestamp"]
            minutes_ago = time_diff.total_seconds() / 60

            logger.info(
                f"SELL ENTRY CONDITION: Price must be ABOVE {resistance_trigger_level:.2f} (HIGH of 5-bar swing high from {int(minutes_ago)} mins ago)")
            logger.info(
                f"Current price: {current_close:.2f} | Condition met: {sell_condition_met}")
        else:
            logger.info(
                "SELL ENTRY CONDITION: No recent 5-bar swing high found - cannot generate sell signals")

        if swing_low_bar is not None:
            support_trigger_level = swing_low_bar['low']
            buy_condition_met = current_close < support_trigger_level
            time_diff = last_row["timestamp"] - swing_low_bar["timestamp"]
            minutes_ago = time_diff.total_seconds() / 60

            logger.info(
                f"BUY ENTRY CONDITION: Price must be BELOW {support_trigger_level:.2f} (LOW of 5-bar swing low from {int(minutes_ago)} mins ago)")
            logger.info(f"Current price: {current_close:.2f} | Condition met: {buy_condition_met}")
        else:
            logger.info(
                "BUY ENTRY CONDITION: No recent 5-bar swing low found - cannot generate buy signals")


        logger.info("===============================")
        
        try:
            logger.info(
                "Bar: time=%s open=%.2f high=%.2f low=%.2f close=%.2f vol=%.1f rsi_14=%.2f until_invalid=%d",
                last_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                last_row["open"],
                last_row["high"],
                last_row["low"],
                last_row["close"],
                last_row["tick_volume"],
                last_row["rsi_14"],
                int(last_row["until_invalid"])
            )
            
            if swing_high_bar is not None:
                logger.info(f"Last swing HIGH: time={swing_high_bar['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, "
                          f"O={swing_high_bar['open']:.2f}, H={swing_high_bar['high']:.2f}, "
                          f"L={swing_high_bar['low']:.2f}, C={swing_high_bar['close']:.2f}")
            
            if swing_low_bar is not None:
                logger.info(f"Last swing LOW: time={swing_low_bar['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, "
                          f"O={swing_low_bar['open']:.2f}, H={swing_low_bar['high']:.2f}, "
                          f"L={swing_low_bar['low']:.2f}, C={swing_low_bar['close']:.2f}")
                
        except KeyError as e:
            logger.error(f"Missing key in data row: {e}")
        
        if last_row['until_invalid'] == 0:
            logger.info("Trading during invalid period (news or maintenance), but continuing to generate signals")
        
        if self.features_to_use:
            missing_features = []
            for col in self.features_to_use:
                if col not in df_feat.columns:
                    missing_features.append(col)
            
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                return
            
            X_last = pd.DataFrame(last_row[self.features_to_use]).T.astype(float)
        else:
            features_to_use = [c for c in df_feat.columns if c not in ['timestamp', 'until_invalid']]
            X_last = pd.DataFrame(last_row[features_to_use]).T.astype(float)
        
        if X_last.isnull().any().any():
            logger.warning("NaN values in input features")
            return
            
        proba = self.model.predict_proba(X_last)[0]
        logger.debug(f"Model prediction shape={proba.shape}")
        
        price_to_sma_50 = last_row['price_to_sma_50']
        
        signal, w_pos, w_neg, tp = aggregate_signal(
            proba,
            last_row['close'],
            self.bin_edges,
            self.model.classes_,
            self.prob_threshold,
            price_to_sma_50
        )
        logger.debug(f"Signal={signal}, w_pos={w_pos}, w_neg={w_neg}, tp={tp}, price_to_sma_50={price_to_sma_50}")

        if signal != 0:
            try:
                if signal == -1:
                    if resistance_trigger_level is None:
                        logger.info(f"⚠️ SELL signal invalidated: No recent 5-bar swing high found")
                        signal = 0
                        tp = None
                    elif not sell_condition_met:
                        logger.info(
                            f"⚠️ SELL signal invalidated: Close {current_close:.2f} is NOT ABOVE swing high's HIGH {resistance_trigger_level:.2f}")
                        signal = 0
                        tp = None
                    else:
                        logger.info(
                            f"✅ IDEAL SELL signal confirmed: Close {current_close:.2f} is ABOVE swing high's HIGH {resistance_trigger_level:.2f}")
                elif signal == 1:
                    if support_trigger_level is None:
                        logger.info(f"⚠️ BUY signal invalidated: No recent 5-bar swing low found")
                        signal = 0
                        tp = None
                    elif not buy_condition_met:
                        logger.info(
                            f"⚠️ BUY signal invalidated: Close {current_close:.2f} is NOT BELOW swing low's LOW {support_trigger_level:.2f}")
                        signal = 0
                        tp = None
                    else:
                        logger.info(
                            f"✅ IDEAL BUY signal confirmed: Close {current_close:.2f} is BELOW swing low's LOW {support_trigger_level:.2f}")
            except Exception as e:
                logger.warning(f"Error in swing high/low filter, continuing with original signal: {e}")

        if signal == 0:
            logger.info("No trade signal generated")
            return
        
        if self.check_existing_trade_conflict(signal, current_price):
            logger.info(f"Trade signal invalidated due to conflict with existing trades")
            return
        
        trade_id = len(self.active_trades) + 1
        ttype = "BUY" if signal == 1 else "SELL"
        
        acct_id = next(iter(self.account_risk_map), 125)
        risk_percent = self.account_risk_map.get(acct_id, 1.0)
        
        if signal == 1:
            take_profit = current_price + w_pos
            stop_loss = current_price - (w_pos * 1.5)
        else:
            take_profit = current_price - w_neg
            stop_loss = current_price + (w_neg * 1.5)
        
        trade_info = {
            "id": trade_id,
            "type": ttype,
            "entry_ts": last_row['timestamp'],
            "entry_price": current_price,
            "acct_id": acct_id,
            "risk": risk_percent,
            "weighted_pos_move": w_pos,
            "weighted_neg_move": w_neg,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "until_invalid": last_row['until_invalid']
        }
        self.active_trades.append(trade_info)
        
        logger.info(
            "New trade: %s ID=%d, ACCT=%d, RISK=%.1f%%, Price=%.2f, TP=%.2f, SL=%.2f, until_invalid=%d",
            ttype, trade_id, acct_id, risk_percent, current_price, take_profit, stop_loss, int(last_row['until_invalid'])
        )
        
        msg = f"OPEN|ID={trade_id}|{ttype}|XAUUSD|TP={take_profit:.2f}|MinutesUntilInvalid={int(last_row['until_invalid'])}"
        logger.info(f"Sending signal: {msg}")
        
        self.socket_server.send_signal_to_clients(msg)

    def send_signal(self, msg):
        logger.info(f"Sending signal: {msg}")
        self.socket_server.send_signal_to_clients(msg)
    
    def forward_to_aspnet(self, msg, client_ip=None):
        message_parts = msg.split("|")
        message_type = message_parts[0]
        
        if message_type == "OPEN_CONFIRM":
            self.asp_net_client.send_open_confirm(message_parts, client_ip)
        
        elif message_type == "CLOSED_CONFIRM":
            self.asp_net_client.send_closed_confirm(message_parts, client_ip)
        
        else:
            logger.info(f"Message type {message_type} not forwarded to ASP.NET API")

def main():
    logger.info("Starting LivePipeline trading system...")
    
    try:
        pipeline_trader = LivePipelineTrader(MODEL_BUNDLE_PATH)
        
        BarsRequestHandler.pipeline_trader = pipeline_trader
        bars_server_address = (HOST, HTTP_PORT)
        bars_httpd = HTTPServer(bars_server_address, BarsRequestHandler)
        logger.info(f"Bars HTTP Server listening on port {HTTP_PORT}")
        
        bars_server_thread = threading.Thread(target=bars_httpd.serve_forever, daemon=True)
        bars_server_thread.start()
        
        logger.info("All servers started. Press Ctrl+C to shutdown...")
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
        if hasattr(pipeline_trader, 'api_httpd'):
            pipeline_trader.api_httpd.shutdown()
        bars_httpd.shutdown()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
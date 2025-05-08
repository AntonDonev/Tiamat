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

MODEL_BUNDLE_PATH = "model.pkl"
HIGH_IMPACT_NEWS_PATH = "high_impact_news.csv"
BUFFER_SIZE = 400
DEFAULT_RISK_REWARD_RATIO = 1.0

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

ENCRYPTION_KEY = b"MMgAZWQi788D8238TjqgPgMhx7XYX4CC"

def xor_cipher(data_bytes, key_bytes):
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

def load_high_impact_news_csv(file_path):
    try:
        if not os.path.exists(file_path):
            logger.warning(f"News file not found: {file_path}. Continuing without news data.")
            return []
        df = pd.read_csv(file_path, delimiter=',', header=None)
        df.columns = ['date','time','currency','impact','news','v1','v2','v3','v4','extra']
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M', utc=True)
        df = df[df['impact'] == 'H']
        news_events = df['timestamp'].sort_values().tolist()
        logger.info(f"Loaded {len(news_events)} high-impact news events from CSV (UTC).")
        return news_events
    except Exception as e:
        logger.error(f"Error loading news CSV '{file_path}': {e}", exc_info=True)
        return []

def calc_until_invalid(
    bar_ts, news_events=None,
    maintenance_start=22, maintenance_end=0,
):
    if news_events is None:
        news_events = []
    
    # Ensure bar_ts is UTC
    if bar_ts.tzinfo is None:
        logger.warning(f"calc_until_invalid received naive timestamp: {bar_ts}. Assuming UTC.")
        bar_ts = bar_ts.replace(tzinfo=timezone.utc)
    else:
        bar_ts = bar_ts.astimezone(timezone.utc)

    # Ensure news_events are UTC and sorted (though load_high_impact_news_csv already sorts)
    processed_news_events = []
    for ne_orig in news_events:
        if ne_orig.tzinfo is None:
            ne_utc = ne_orig.replace(tzinfo=timezone.utc)
        else:
            ne_utc = ne_orig.astimezone(timezone.utc)
        processed_news_events.append(ne_utc)
    processed_news_events.sort()

    # 1. Check for exact news match
    if bar_ts in processed_news_events:
        return 0

    # 2. Check for current maintenance period
    # Calculate effective maintenance window for current bar_ts
    maint_start_dt_effective = bar_ts.replace(hour=maintenance_start, minute=0, second=0, microsecond=0)
    maint_end_dt_effective = bar_ts.replace(hour=maintenance_end, minute=0, second=0, microsecond=0)

    if maint_end_dt_effective <= maint_start_dt_effective: # Maintenance spans midnight
        # If bar_ts is on the "start" day but after maint_start, or on the "end" day but before maint_end
        if bar_ts >= maint_start_dt_effective or bar_ts < maint_end_dt_effective:
                # To correctly check, if bar_ts is after midnight but before maint_end, maint_start should be from previous day
            if bar_ts < maint_end_dt_effective and bar_ts.hour < maintenance_start : # e.g. bar_ts 00:30, maint 22-01
                if (maint_start_dt_effective - timedelta(days=1)) <= bar_ts < maint_end_dt_effective:
                    return 0
            elif maint_start_dt_effective <= bar_ts : # e.g bar_ts 23:00, maint 22-01
                return 0

    elif maint_start_dt_effective <= bar_ts < maint_end_dt_effective: # Maintenance does not span midnight
        return 0


    # 3. Calculate news-based distance
    closest_past_news_ts = None
    min_diff_past = timedelta.max
    for ne in processed_news_events:
        if ne < bar_ts:
            diff = bar_ts - ne
            if diff < min_diff_past:
                min_diff_past = diff
                closest_past_news_ts = ne

    closest_future_news_ts = None
    min_diff_future = timedelta.max
    for ne in processed_news_events:
        if ne > bar_ts:
            diff = ne - bar_ts
            if diff < min_diff_future:
                min_diff_future = diff
                closest_future_news_ts = ne

    minutes_from_past = int(min_diff_past.total_seconds() // 60) if closest_past_news_ts else float('inf')
    minutes_to_future = int(min_diff_future.total_seconds() // 60) if closest_future_news_ts else float('inf')

    news_based_distance_minutes = float('inf')
    if minutes_from_past != float('inf') and minutes_to_future != float('inf'):
        if minutes_from_past < minutes_to_future:
            news_based_distance_minutes = minutes_from_past
        elif minutes_to_future < minutes_from_past:
            news_based_distance_minutes = minutes_to_future
        else: # Equidistant
            news_based_distance_minutes = 30 
    elif minutes_from_past != float('inf'):
        news_based_distance_minutes = minutes_from_past
    elif minutes_to_future != float('inf'):
        news_based_distance_minutes = minutes_to_future

    # 4. Calculate minutes to next maintenance start
    maint_today_starts = bar_ts.replace(hour=maintenance_start, minute=0, second=0, microsecond=0)
    maint_tomorrow_starts = (bar_ts + timedelta(days=1)).replace(hour=maintenance_start, minute=0, second=0, microsecond=0)

    actual_next_maint_start_ts = None
    if maint_today_starts > bar_ts: 
        actual_next_maint_start_ts = maint_today_starts
    else: 
        actual_next_maint_start_ts = maint_tomorrow_starts
    
    minutes_to_next_maint = int((actual_next_maint_start_ts - bar_ts).total_seconds() // 60)

    # 5. Final until_invalid value
    if news_based_distance_minutes == float('inf'): 
        final_until_invalid = minutes_to_next_maint
    else:
        final_until_invalid = min(news_based_distance_minutes, minutes_to_next_maint)
        
    return max(0, final_until_invalid)


def engineer_features(df):
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
    df['trend_streak'] = np.minimum(df['trend_streak'], 20)
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
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    df['price_to_kijun'] = df['close'] / (df['ichimoku_base'] + 1e-9) - 1
    df['tenkan_kijun_cross'] = df['ichimoku_conv'] / (df['ichimoku_base'] + 1e-9) - 1
    df['cloud_thickness'] = (df['ichimoku_a'] - df['ichimoku_b']) / (df['close'] + 1e-9)
    df['rsi_7'] = ta.momentum.rsi(df['close'], window=7, fillna=False)
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14, fillna=False)
    df['roc_5'] = ta.momentum.roc(df['close'], window=5, fillna=False)
    df['roc_10'] = ta.momentum.roc(df['close'], window=10, fillna=False)
    df['roc_20'] = ta.momentum.roc(df['close'], window=20, fillna=False)
    rsi_scaled = 2 * (df['rsi_14'].fillna(50) - 50) / 100
    df['fisher_rsi_14'] = 0.5 * np.log((1 + rsi_scaled + 1e-9) / (1 - rsi_scaled + 1e-9))
    boll = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2, fillna=False)
    df['boll_pct_b'] = boll.bollinger_pband()
    df['boll_width'] = boll.bollinger_wband()
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14, fillna=False)
    df['atr_pct'] = atr.average_true_range() / (df['close'] + 1e-9) * 100
    adx5 = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=5, fillna=False)
    df['adx5'] = adx5.adx()
    adx_ind = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=False) 
    df['adx'] = adx_ind.adx()
    df['adx_pos'] = adx_ind.adx_pos()
    df['adx_neg'] = adx_ind.adx_neg()
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
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_neg_pos_bin_indices(bin_edges: np.ndarray):
    neg_indices, pos_indices, num_bins = [], [], len(bin_edges) - 1
    for i in range(num_bins):
        if bin_edges[i+1] <= 0: neg_indices.append(i)
        elif bin_edges[i] >= 0: pos_indices.append(i)
    return neg_indices, pos_indices

def aggregate_signal(proba, price, bin_edges, model_classes, prob_threshold, price_to_sma_50):
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
    if prob_diff >= prob_threshold:
        if price_to_sma_50 < 0 and prob_diff < (prob_threshold * trend_factor): logger.info("BUY rejected: Against trend.")
        else: signal, pred_tp = 1, price + exp_pos_mag
    elif -prob_diff >= prob_threshold:
        if price_to_sma_50 > 0 and (-prob_diff) < (prob_threshold * trend_factor): logger.info("SELL rejected: Against trend.")
        else: signal, pred_tp = -1, price - exp_neg_mag
    return signal, exp_pos_mag, exp_neg_mag, pred_tp


class LiveDataBuffer:
    def __init__(self, max_size=BUFFER_SIZE, news_events=None):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.news_events = news_events if news_events is not None else []
        self.last_timestamp_processed = None

    def append(self, bar):
        try:
            bar_ts_naive = pd.to_datetime(bar['timestamp'].replace('.', '-'))
        except Exception as e:
            logger.error(f"Could not parse timestamp: {bar.get('timestamp')}. Error: {e}. Skipping bar.")
            return
        bar_ts_utc = bar_ts_naive.replace(tzinfo=timezone.utc)

        if bar_ts_utc == self.last_timestamp_processed:
            return

        self.last_timestamp_processed = bar_ts_utc
        until_inv = calc_until_invalid(bar_ts_utc, self.news_events, maintenance_start=22, maintenance_end=0)
        row = {'timestamp': bar_ts_utc, 'open': float(bar['open']), 'high': float(bar['high']),
               'low': float(bar['low']), 'close': float(bar['close']), 'volume': float(bar['volume']),
               'tick_volume': float(bar['volume']),
               'until_invalid': until_inv}
        self.buffer.append(row)

    def to_dataframe(self):
        return pd.DataFrame(list(self.buffer))


class DllSocketServer:
    def __init__(self, pipeline_trader, host, port, allowed_hwids_map_ref):
        self.pipeline_trader = pipeline_trader
        self.host, self.port = host, port
        self.allowed_hwids_map_ref = allowed_hwids_map_ref 
        self.server_socket, self.running = None, True
        self.connections_lock = threading.Lock()
        self.connections = [] 

    def start(self):
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
        self.running = False
        if self.server_socket:
            try: self.server_socket.shutdown(socket.SHUT_RDWR)
            except OSError: pass
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
        while self.running and self.server_socket:
            try:
                conn, addr = self.server_socket.accept()
                logger.info(f"DLL connection attempt from {addr[0]}:{addr[1]}")
                with self.connections_lock:
                    self.connections.append([conn, addr, None, False]) 
            except OSError:
                if self.running: logger.warning("Accept loop error or socket closed.")
                break
            except Exception as e:
                if self.running: logger.error(f"Unexpected accept error: {e}", exc_info=True)
                break
        logger.info("DLL accept loop finished.")

    def _remove_connection(self, conn_to_remove):
        removed = False
        with self.connections_lock:
            idx_to_remove = -1
            for i, conn_details in enumerate(self.connections):
                if conn_details[0] is conn_to_remove:
                    idx_to_remove = i
                    break

            if idx_to_remove != -1:
                conn, addr, hwid, _ = self.connections.pop(idx_to_remove)
                logger.info(f"Removed connection from {addr[0]} (HWID: {hwid}).")
                try: conn.close()
                except OSError: pass
                removed = True
        return removed

    def disconnect_client_by_hwid(self, hwid_to_disconnect):
        logger.info(f"Attempting to disconnect client with HWID: {hwid_to_disconnect}")
        conn_to_close = None
        with self.connections_lock:
            for conn_details in self.connections:
                if conn_details[2] == hwid_to_disconnect and conn_details[3]:
                    conn_to_close = conn_details[0]
                    break
        
        if conn_to_close:
            logger.info(f"Found active connection for HWID {hwid_to_disconnect}. Closing it.")
            try:
                conn_to_close.shutdown(socket.SHUT_RDWR)
            except OSError: pass 
            try:
                conn_to_close.close()
            except OSError: pass
            self._remove_connection(conn_to_close) 
        else:
            logger.info(f"No active authenticated connection found for HWID {hwid_to_disconnect} to disconnect.")


    def _recv_loop(self):
        while self.running:
            connections_to_remove_sockets = []
            with self.connections_lock:
                current_connections_snapshot = list(self.connections)

            for conn_details in current_connections_snapshot:
                conn, addr, current_hwid, is_authenticated = conn_details
                client_ip = addr[0]
                try:
                    conn.settimeout(0.01) 
                    data = conn.recv(1024)
                    if data:
                        msg = data.decode('utf-8', errors='replace').strip()
                        logger.info(f"Socket RECV from {client_ip} (HWID: {current_hwid}, Auth: {is_authenticated}): {msg}")

                        if not is_authenticated:
                            if msg.startswith("AUTH|HWID="):
                                try:
                                    received_hwid = msg.split("=", 1)[1]
                                    if received_hwid in self.pipeline_trader.allowed_devices: 
                                        with self.connections_lock: 
                                            for i, detail in enumerate(self.connections):
                                                if detail[0] is conn: 
                                                    self.connections[i][2] = received_hwid 
                                                    self.connections[i][3] = True      
                                                    current_hwid = received_hwid 
                                                    is_authenticated = True
                                                    break 
                                        acc_id = self.pipeline_trader.allowed_devices[received_hwid]
                                        logger.info(f"HWID {received_hwid} authenticated for AccountID {acc_id} from IP {client_ip}.")
                                    else:
                                        logger.warning(f"Auth FAIL: HWID '{received_hwid}' from {client_ip} not in allowed devices. Closing.")
                                        connections_to_remove_sockets.append(conn)
                                except Exception as e:
                                    logger.error(f"Error processing AUTH msg '{msg}' from {client_ip}: {e}. Closing.", exc_info=True)
                                    connections_to_remove_sockets.append(conn)
                            else:
                                logger.warning(f"First msg from {client_ip} not AUTH: '{msg}'. Closing.")
                                connections_to_remove_sockets.append(conn)
                        else: 
                            if msg.startswith("OPEN_CONFIRM|") or msg.startswith("CLOSED_CONFIRM|") or msg.startswith("EDIT_CONFIRM|"):
                                pass 
                            self.pipeline_trader.forward_to_aspnet(msg, client_ip, current_hwid)

                    elif not data: 
                        logger.info(f"Connection closed by {client_ip} (HWID: {current_hwid}).")
                        connections_to_remove_sockets.append(conn)
                
                except socket.timeout:
                    continue 
                except (ConnectionResetError, BrokenPipeError, OSError) as e:
                    logger.warning(f"Socket ERR for {client_ip} (HWID: {current_hwid}): {e}. Marking for removal.")
                    connections_to_remove_sockets.append(conn)
                except Exception as e:
                    logger.error(f"Unexpected RECV loop ERR for {client_ip} (HWID: {current_hwid}): {e}. Marking for removal.", exc_info=True)
                    connections_to_remove_sockets.append(conn)
            
            if connections_to_remove_sockets:
                for sock_to_remove in list(set(connections_to_remove_sockets)): 
                    self._remove_connection(sock_to_remove)
            
            time.sleep(0.05) 
        logger.info("DLL receive loop finished.")

    def send_signal_to_clients(self, message: str):
        connections_to_remove_sockets = []
        with self.connections_lock:
            if not any(auth for _, _, _, auth in self.connections): 
                return

            encrypted_message_bytes = xor_cipher(message.encode('utf-8'), ENCRYPTION_KEY)
            if not encrypted_message_bytes and message: 
                logger.error(f"XOR Encryption failed for message: {message}. Check key. Aborting send.")
                return
            
            logger.debug(f"Plain message to send: '{message}', Encrypted length: {len(encrypted_message_bytes)}")
            current_connections_snapshot = list(self.connections) 

        for conn_details in current_connections_snapshot:
            conn, addr, hwid, authenticated = conn_details
            if authenticated:
                try:
                    conn.sendall(encrypted_message_bytes)
                    logger.info(f"Sent (encrypted) to {addr[0]} (HWID: {hwid}): {message}")
                except (ConnectionResetError, BrokenPipeError, OSError) as e:
                    logger.error(f"Send error to {addr[0]} (HWID: {hwid}): {e}. Marking for removal.")
                    connections_to_remove_sockets.append(conn)
                except Exception as e:
                    logger.error(f"Unexpected send error to {addr[0]} (HWID: {hwid}): {e}. Marking for removal.", exc_info=True)
                    connections_to_remove_sockets.append(conn)
        
        if connections_to_remove_sockets:
            for sock_to_remove in list(set(connections_to_remove_sockets)): 
                self._remove_connection(sock_to_remove)


class ApiRequestHandler(BaseHTTPRequestHandler):
    pipeline_trader = None 

    def log_message(self, format, *args): pass 

    def send_json_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*') 
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def check_api_key(self):
        api_key = self.headers.get('x-api-key')
        if not api_key or api_key != API_KEY:
            logger.warning(f"Unauthorized API access from {self.client_address[0]}. Path: {self.path}. Key: '{api_key}'")
            self.send_json_response(401, {"error": "Unauthorized"})
            return False
        return True

    def do_OPTIONS(self): 
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-API-Key')
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self.send_json_response(200, {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})
        else:
            self.send_json_response(404, {"error": "Endpoint not found"})

    def do_POST(self):
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
                    new_hwid = json_data.get('hwid')
                    account_id = str(json_data.get('accountId', 'UNKNOWN_ACCOUNT'))
                    
                    if not new_hwid:
                        self.send_json_response(400, {"error": "HWID is required for START."})
                        return
                    if account_id == 'UNKNOWN_ACCOUNT' and not json_data.get('accountId'):
                         logger.warning(f"API START: AccountID not provided, defaulting to 'UNKNOWN_ACCOUNT' for HWID {new_hwid}")
                    
                   
                    old_hwid_for_this_account = None
                    current_allowed_devices_copy = dict(ApiRequestHandler.pipeline_trader.allowed_devices)
                    for hw_iter, acc_iter in current_allowed_devices_copy.items():
                        if acc_iter == account_id: # Found the account
                            if hw_iter != new_hwid: 
                                old_hwid_for_this_account = hw_iter
                            break 

                    if old_hwid_for_this_account:
                        logger.info(f"API START: AccountID '{account_id}' is changing HWID from '{old_hwid_for_this_account}' to '{new_hwid}'.")
                        logger.info(f"Attempting to disconnect client associated with old HWID '{old_hwid_for_this_account}'.")
                        ApiRequestHandler.pipeline_trader.socket_server.disconnect_client_by_hwid(old_hwid_for_this_account)

                        if old_hwid_for_this_account in ApiRequestHandler.pipeline_trader.allowed_devices:
                             del ApiRequestHandler.pipeline_trader.allowed_devices[old_hwid_for_this_account]
                    
                    ApiRequestHandler.pipeline_trader.allowed_devices[new_hwid] = account_id
                    logger.info(f"API: Registered/Updated HWID='{new_hwid}' for AccountID='{account_id}'")
                    self.send_json_response(200, {"status": "ok", "message": f"HWID {new_hwid} registered for Account ID {account_id}"})

                elif command == 'EDIT':
                    account_id_from_api = str(json_data.get('accountId', '0'))
                    max_risk_str = str(json_data.get('maxRisk', '1.5'))
                    untradable_period_str = str(json_data.get('untradablePeriod', '60'))

                    target_hwid_for_account = None
                    for hw_iter, acc_iter in ApiRequestHandler.pipeline_trader.allowed_devices.items():
                        if acc_iter == account_id_from_api:
                            target_hwid_for_account = hw_iter
                            break
                    
                    if not target_hwid_for_account:
                        logger.warning(f"API EDIT: No HWID found registered for AccountID '{account_id_from_api}'. Command not sent.")
                        self.send_json_response(404, {"error": f"AccountID '{account_id_from_api}' is not registered with any HWID."})
                        return

                    device_is_active = False
                    if ApiRequestHandler.pipeline_trader.socket_server:
                        with ApiRequestHandler.pipeline_trader.socket_server.connections_lock:
                            current_connections_snapshot = list(ApiRequestHandler.pipeline_trader.socket_server.connections)
                            for conn_details in current_connections_snapshot:
                                if conn_details[2] == target_hwid_for_account and conn_details[3]: # Check HWID and auth status
                                    device_is_active = True
                                    break
                    
                    if device_is_active:
                        msg_to_dll = f"EDIT|{target_hwid_for_account}|{max_risk_str}|{untradable_period_str}"
                        logger.info(f"API: Sending {command.upper()} command to active client with HWID '{target_hwid_for_account}' (AccountID '{account_id_from_api}'): {msg_to_dll}")
                        ApiRequestHandler.pipeline_trader.send_signal(msg_to_dll) # send_signal broadcasts
                        self.send_json_response(200, {"status": "ok", "message": f"{command.upper()} command sent to active client for HWID '{target_hwid_for_account}'."})
                    else:
                        logger.info(f"API EDIT: HWID '{target_hwid_for_account}' for AccountID '{account_id_from_api}' is registered but not actively connected/authenticated. Command not sent.")
                        self.send_json_response(409, { # 409 Conflict: resource exists but state prevents action
                            "error": f"Device for AccountID '{account_id_from_api}' (HWID: {target_hwid_for_account}) is registered but not currently active. EDIT command not sent."
                        })

                elif command == 'OPEN':
                    py_id = json_data.get('ID', f'API_OPEN_{int(time.time())}')
                    ttype = json_data.get('TYPE', 'BUY')
                    symbol = json_data.get('SYMBOL', 'XAUUSD')
                    tp = json_data.get('TP', '0.0') 
                    mins = json_data.get('MINUTESUNTILINVALID', '0')
                    if tp == '0.0': 
                        logger.error("API OPEN command missing TP value or TP is zero.")
                        self.send_json_response(400, {"error": "TP value (non-zero) required for OPEN command via API"})
                        return
                    msg_to_dll = f"OPEN|ID={py_id}|{ttype}|{symbol}|TP={tp}|MINUTESUNTILINVALID={mins}"
                    logger.info(f"API: Sending {command.upper()} to DLLs: {msg_to_dll}")
                    ApiRequestHandler.pipeline_trader.send_signal(msg_to_dll)
                    self.send_json_response(200, {"status": "ok", "message": f"{command.upper()} command sent to DLLs."})

                elif command == 'CLOSE':
                    params_list = [f"{k.upper()}={v}" for k,v in json_data.items() if k != 'command' and k.lower() != 'id']
                    py_id_val = json_data.get('ID', json_data.get('id')) 
                    if not py_id_val:
                        self.send_json_response(400, {"error": "ID is required for CLOSE command."})
                        return
                    
                    id_param = f"ID={py_id_val}"
                    msg_parts = [command.upper(), id_param] + params_list
                    msg_to_dll = "|".join(msg_parts)
                    
                    logger.info(f"API: Sending {command.upper()} to DLLs: {msg_to_dll}")
                    ApiRequestHandler.pipeline_trader.send_signal(msg_to_dll)
                    self.send_json_response(200, {"status": "ok", "message": f"{command.upper()} command sent to DLLs."})
                else:
                    self.send_json_response(400, {"error": f"Unknown command: {command}"})

            elif self.path == '/edit': 
                logger.warning(f"API: Deprecated /edit endpoint used by {self.client_address[0]}. Use /command.")
                account_id_str = str(json_data.get('account_id', '0'))
                max_risk_str = str(json_data.get('max_risk', '1.5'))
                untradable_period_str = str(json_data.get('untradable_period', '60'))
                
                msg_to_dll = f"EDIT|{account_id_str}|{max_risk_str}|{untradable_period_str}"
                logger.info(f"API /edit: Sending EDIT command to DLLs: {msg_to_dll}")
                ApiRequestHandler.pipeline_trader.send_signal(msg_to_dll)
                self.send_json_response(200, {"status": "ok", "message": "EDIT (legacy) command sent."})
            else:
                self.send_json_response(404, {"error": "API Endpoint not found"})

        except Exception as e:
            logger.error(f"Error processing API command '{json_data.get('command')}' at '{self.path}': {e}", exc_info=True)
            self.send_json_response(500, {"error": f"Internal server error processing command."})


class BarsRequestHandler(BaseHTTPRequestHandler):
    pipeline_trader = None 
    def log_message(self, format, *args): pass 

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"Empty request body"}')
            return
        post_data_bytes = self.rfile.read(content_length)
        try:
            json_data = json.loads(post_data_bytes.decode('utf-8'))
            bar_data = {"timestamp": json_data.get("time", ""), "open": json_data.get("open", 0.0),
                        "high": json_data.get("high", 0.0), "low": json_data.get("low", 0.0),
                        "close": json_data.get("close", 0.0), "volume": json_data.get("volume", 0.0)}

            if not bar_data["timestamp"]: 
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


class AspNetApiClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'PythonTradingSystemClient/1.3', 
            'Accept': '*/*',
            'Cache-Control': 'no-cache', 
            'x-api-key': self.api_key
        })

    def _send_request(self, method, endpoint, data=None, timeout=10):
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
        try:
            py_trade_id, trade_type, symbol, size_str, mql_ticket_str = "", "", "", "0.0", ""
            opened_at_mql_str = ""

            for part in message_parts: 
                if part.startswith("ID="): py_trade_id = part.split("=",1)[1]
            if len(message_parts) > 2: trade_type = message_parts[2]
            if len(message_parts) > 3: symbol = message_parts[3]
            if len(message_parts) > 4: size_str = message_parts[4]
            if len(message_parts) > 6: opened_at_mql_str = message_parts[6] 
            if len(message_parts) > 7: mql_ticket_str = message_parts[7]


            opened_at_iso = datetime.now(timezone.utc).isoformat() 
            if opened_at_mql_str:
                try:
                    opened_at_iso = pd.to_datetime(opened_at_mql_str, format='%Y.%m.%d %H:%M').tz_localize(None).tz_localize('UTC').isoformat()
                except Exception as e:
                    logger.warning(f"Could not parse opened_at '{opened_at_mql_str}'. Using current time. Error: {e}")
            
            payload = {"id": py_trade_id, "symbol": symbol, "type": trade_type, "size": float(size_str), 
                       "openedAt": opened_at_iso, "fromIp": client_ip or "N/A", "fromHwid": hwid or "N/A",
                       "internalDllId": mql_ticket_str } 
            logger.info(f"Forwarding OPEN_CONFIRM to ASP.NET: {payload}")
            success, _ = self._send_request("POST", "open-confirm", data=payload)
            return success
        except Exception as e: logger.error(f"Error formatting/sending OPEN_CONFIRM: {e}", exc_info=True); return False

    def send_closed_confirm(self, message_parts, client_ip, hwid=None):
        try:
            py_trade_id, profit_str, capital_str, closed_at_mql_str, mql_ticket_str = "", "0.0", "0.0", "", ""
            for part in message_parts:
                if part.startswith("ID="): py_trade_id = part.split("=",1)[1]
            if len(message_parts) > 2: profit_str = message_parts[2]
            if len(message_parts) > 3: capital_str = message_parts[3]
            if len(message_parts) > 4: closed_at_mql_str = message_parts[4]
            if len(message_parts) > 6: mql_ticket_str = message_parts[6]


            closed_at_iso = datetime.now(timezone.utc).isoformat()
            if closed_at_mql_str:
                try:
                    closed_at_iso = pd.to_datetime(closed_at_mql_str, format='%Y.%m.%d %H:%M').tz_localize(None).tz_localize('UTC').isoformat()
                except Exception as e:
                    logger.warning(f"Could not parse closed_at '{closed_at_mql_str}'. Using current time. Error: {e}")

            payload = {"id": py_trade_id, "profit": float(profit_str), "currentCapital": float(capital_str),
                       "closedAt": closed_at_iso, "fromIp": client_ip or "N/A", "fromHwid": hwid or "N/A",
                       "internalDllId": mql_ticket_str }
            logger.info(f"Forwarding CLOSED_CONFIRM to ASP.NET: {payload}")
            success, _ = self._send_request("POST", "closed-confirm", data=payload)
            return success
        except Exception as e: logger.error(f"Error formatting/sending CLOSED_CONFIRM: {e}", exc_info=True); return False

    def check_health(self):
        logger.info(f"Checking ASP.NET API health at {self.api_url}/health")
        success, response_data = self._send_request("GET", "health", timeout=5)
        if success: logger.info(f"ASP.NET API Health check OK: {response_data}")
        else: logger.error(f"ASP.NET API Health check FAILED. Details: {response_data}")
        return success


class LivePipelineTrader:
    def __init__(self, model_bundle_path):
        self.model_bundle = self.load_model(model_bundle_path)
        if not self.model_bundle:
            logger.critical(f"Halting: Model bundle '{model_bundle_path}' could not be loaded.")
            raise SystemExit(f"Model bundle load failed: {model_bundle_path}")

        self.model = self.model_bundle["model"]
        self.bin_edges = self.model_bundle["bin_edges"]
        self.timeframe = self.model_bundle["timeframe"]
        self.prob_threshold = self.model_bundle.get("prob_threshold", 0.65) 
        self.features_to_use = self.model_bundle.get("features", None)
        self.risk_reward_ratio = self.model_bundle.get("risk_reward_ratio", DEFAULT_RISK_REWARD_RATIO)
        logger.info(f"Model loaded. TF: {self.timeframe}, ProbThresh: {self.prob_threshold}, R:R: {self.risk_reward_ratio}, Features: {'Specific' if self.features_to_use else 'Auto'}")

        self.news_events = load_high_impact_news_csv(HIGH_IMPACT_NEWS_PATH)
        self.data_buffer = LiveDataBuffer(BUFFER_SIZE, self.news_events)

        self.monitored_trades_lock = threading.Lock()
        self.monitored_trades_for_internal_logic = [] 

        self.allowed_devices = {} 

        self.socket_server = DllSocketServer(self, HOST, SOCKET_PORT, self.allowed_devices)
        self.socket_server.start() 
        self.asp_net_client = AspNetApiClient(ASPNET_API_URL, API_KEY)

        ApiRequestHandler.pipeline_trader = self
        BarsRequestHandler.pipeline_trader = self 

        self.api_httpd = HTTPServer((HOST, API_PORT), ApiRequestHandler)
        logger.info(f"API Command Server listening on {HOST}:{API_PORT}")
        api_server_thread = threading.Thread(target=self.api_httpd.serve_forever, daemon=True, name="ApiServerThread")
        api_server_thread.start()

        if not self.asp_net_client.check_health(): 
            logger.warning("Initial ASP.NET API health check failed. Check connectivity and API key.")

    def load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            model_bundle = joblib.load(model_path)
            required_keys = ["model", "bin_edges", "timeframe"] 
            if not all(key in model_bundle for key in required_keys):
                missing = [k for k in required_keys if k not in model_bundle]
                logger.error(f"Model bundle from '{model_path}' is missing required key(s): {missing}")
                return None
            logger.info(f"Successfully loaded and validated model bundle from {model_path}")
            return model_bundle
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
            return None

    def calculate_sl_for_internal_monitoring(self, trade_type, entry_price, take_profit_price):
        if self.risk_reward_ratio <= 0: 
            logger.warning(f"Invalid Risk/Reward ratio ({self.risk_reward_ratio}) for internal SL calculation.")
            return None
        
        sl_price = None
        if trade_type == "BUY":
            if take_profit_price <= entry_price: 
                logger.warning(f"TP price ({take_profit_price}) not valid for BUY trade at entry {entry_price} for internal SL calc.")
                return None
            reward_amount = take_profit_price - entry_price
            risk_amount = reward_amount / self.risk_reward_ratio
            sl_price = entry_price - risk_amount
        elif trade_type == "SELL":
            if take_profit_price >= entry_price: 
                logger.warning(f"TP price ({take_profit_price}) not valid for SELL trade at entry {entry_price} for internal SL calc.")
                return None
            reward_amount = entry_price - take_profit_price
            risk_amount = reward_amount / self.risk_reward_ratio
            sl_price = entry_price + risk_amount
            
        return round(sl_price, 5) if sl_price is not None else None


    def update_internal_trade_statuses(self, current_price, symbol="XAUUSD"): 
        trades_to_remove_internally = []
        with self.monitored_trades_lock:
            for trade in self.monitored_trades_for_internal_logic:
                if trade.get("symbol") != symbol: 
                    continue

                tp_hit, sl_hit = False, False
                reason_internal = ""
                trade_type = trade.get("type")
                take_profit = trade.get("take_profit") 
                stop_loss = trade.get("stop_loss") 

                if trade_type == "BUY":
                    if take_profit is not None and current_price >= take_profit:
                        tp_hit, reason_internal = True, "INTERNAL_TP_HIT"
                    elif stop_loss is not None and current_price <= stop_loss:
                        sl_hit, reason_internal = True, "INTERNAL_SL_HIT"
                elif trade_type == "SELL":
                    if take_profit is not None and current_price <= take_profit:
                        tp_hit, reason_internal = True, "INTERNAL_TP_HIT"
                    elif stop_loss is not None and current_price >= stop_loss:
                        sl_hit, reason_internal = True, "INTERNAL_SL_HIT"
                
                if tp_hit or sl_hit:
                    logger.info(f"Internal Check: Python model trade ID={trade['id']} ({trade_type} {symbol}) hit internal {reason_internal} at price {current_price:.5f}. Removing from Python's monitoring list.")
                    trades_to_remove_internally.append(trade["id"])
            
            if trades_to_remove_internally:
                self.monitored_trades_for_internal_logic = [
                    t for t in self.monitored_trades_for_internal_logic if t["id"] not in trades_to_remove_internally
                ]
                logger.info(f"Removed {len(trades_to_remove_internally)} trade(s) from Python's internal model monitoring list.")


    def on_new_bar(self, bar_dict):
        self.data_buffer.append(bar_dict)
        if len(self.data_buffer.buffer) < 205: 
            return

        df_buf = self.data_buffer.to_dataframe()
        if df_buf.empty: return

        try:
            df_feat = engineer_features(df_buf.copy()) 
            if df_feat.empty: return
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}", exc_info=True)
            return

        last_row = df_feat.iloc[-1]
        current_price = last_row["close"]
        current_ts_utc = last_row["timestamp"] 

        try: 
            log_msg = (
                f"New bar: O:{last_row.get('open', 0.0):.5f} H:{last_row.get('high', 0.0):.5f} "
                f"L:{last_row.get('low', 0.0):.5f} C:{current_price:.5f} "
                f"Vol:{last_row.get('tick_volume', 0.0):.1f} " 
                f"UntilInvalid:{int(last_row.get('until_invalid', -1))}m "
                f"TS:{last_row['timestamp'].strftime('%H:%M:%S')}"
            )
            logger.info(log_msg)
        except Exception as e: logger.warning(f"Could not log bar details: {e}")
        
        self.update_internal_trade_statuses(current_price, symbol="XAUUSD") 

        with self.monitored_trades_lock:
            if self.monitored_trades_for_internal_logic:
                active_internal_trade_id = self.monitored_trades_for_internal_logic[0]['id'] 
                logger.info(f"Python model is currently monitoring its own trade: {active_internal_trade_id}. No new model signals until its internal TP/SL is hit.")
                return

        minutes_to_invalid = last_row.get('until_invalid', 9999)
        if minutes_to_invalid == 0: 
            return
        buffer_minutes_before_event = 5 
        if minutes_to_invalid <= buffer_minutes_before_event:
            logger.info(f"Bar at {current_ts_utc}: Approaching invalid period in {minutes_to_invalid}m (buffer {buffer_minutes_before_event}m). No new model trades.")
            return

        features_for_model = self.features_to_use
        if not features_for_model: 
            features_for_model = [c for c in df_feat.columns if c not in ['timestamp', 'until_invalid', 'open', 'high', 'low', 'volume', 'tick_volume']]

        missing_model_features = [f for f in features_for_model if f not in last_row.index]
        if missing_model_features:
            logger.error(f"Missing features required by model: {missing_model_features}. Skipping prediction.")
            return

        X_last = pd.DataFrame([last_row[features_for_model].to_dict()]).astype(float) 
        if X_last.isnull().values.any():
            logger.warning(f"NaN values detected in features for prediction: {X_last.columns[X_last.isnull().any()].tolist()}. Skipping.")
            return
        
        try:
            proba = self.model.predict_proba(X_last)[0]
        except Exception as e:
            logger.error(f"Model predict_proba error: {e}", exc_info=True)
            return

        price_to_sma50 = last_row.get('price_to_sma_50', 0.0) 
        model_signal, expected_pos_mag, expected_neg_mag, _ = aggregate_signal(
            proba, current_price, self.bin_edges, self.model.classes_, self.prob_threshold, price_to_sma50
        )

        if model_signal == 0: 
            return

        trade_id_py = f"PY_{int(time.time()*1000)}" 
        trade_type_str = "BUY" if model_signal == 1 else "SELL"
        symbol = "XAUUSD" 

        tp_price = 0.0
        if model_signal == 1:
            tp_price = current_price + expected_pos_mag
        else: 
            tp_price = current_price - expected_neg_mag
        
        tp_price = round(tp_price, 5) 

        sl_price_internal = self.calculate_sl_for_internal_monitoring(trade_type_str, current_price, tp_price)

        if sl_price_internal is None: 
            logger.warning(f"Could not calculate a valid internal SL for {trade_type_str} {symbol} at {current_price} with TP {tp_price}. Signal aborted by Python model.")
            return

        trade_to_monitor = {
            "id": trade_id_py, "symbol": symbol, "type": trade_type_str,
            "entry_price": round(current_price, 5), 
            "take_profit": tp_price, 
            "stop_loss": sl_price_internal, 
            "entry_ts_utc": current_ts_utc.isoformat()
        }
        with self.monitored_trades_lock:
            self.monitored_trades_for_internal_logic.append(trade_to_monitor)

        logger.info(
            f"+++ NEW PYTHON MODEL TRADE SIGNAL +++: {trade_type_str} ID={trade_id_py} ({symbol}), "
            f"Entry@{current_price:.5f}, TP Sent to EA={tp_price:.5f} (EA calculates its own SL), "
            f"Python Internal SL={sl_price_internal:.5f}, MinsToInvalid={minutes_to_invalid}"
        )
        logger.info(f"Trade {trade_id_py} added to Python's internal monitoring list.")

        msg_to_dll = (f"OPEN|ID={trade_id_py}|{trade_type_str}|{symbol}"
                      f"|TP={tp_price:.5f}" 
                      f"|MINUTESUNTILINVALID={int(minutes_to_invalid)}") 

        self.send_signal(msg_to_dll)


    def send_signal(self, msg: str):
        self.socket_server.send_signal_to_clients(msg)

    def forward_to_aspnet(self, msg: str, client_ip=None, hwid=None):
        message_parts = msg.split("|")
        message_type = message_parts[0] if message_parts else ""

        if message_type == "OPEN_CONFIRM":
            self.asp_net_client.send_open_confirm(message_parts, client_ip, hwid)
        elif message_type == "CLOSED_CONFIRM":
            self.asp_net_client.send_closed_confirm(message_parts, client_ip, hwid)
        elif message_type == "EDIT_CONFIRM": 
            logger.info(f"ASP.NET Forward: Received EDIT_CONFIRM from HWID {hwid}. Specific forwarding for this message type to ASP.NET endpoint is not implemented (logged only).")
            pass


    def shutdown(self):
        logger.info("LivePipelineTrader shutdown initiated...")
        if hasattr(self, 'socket_server') and self.socket_server:
            self.socket_server.stop() 

        if hasattr(self, 'api_httpd') and self.api_httpd:
            logger.info("Stopping API Command Server...")
            try:
                threading.Thread(target=self.api_httpd.shutdown, daemon=True).start()
                self.api_httpd.server_close() 
            except Exception as e:
                logger.error(f"Error shutting down API HTTPD: {e}")
        logger.info("LivePipelineTrader shutdown sequence completed.")


def main():
    logger.info("Starting LivePipeline Trading System...")
    pipeline_trader = None
    bars_data_httpd = None 

    try:
        pipeline_trader = LivePipelineTrader(MODEL_BUNDLE_PATH)
        
        BarsRequestHandler.pipeline_trader = pipeline_trader 
        bars_data_httpd = HTTPServer((HOST, HTTP_PORT), BarsRequestHandler)
        logger.info(f"Bars Data HTTP Server listening on {HOST}:{HTTP_PORT}")
        bars_server_thread = threading.Thread(target=bars_data_httpd.serve_forever, daemon=True, name="BarsServerThread")
        bars_server_thread.start()
        
        logger.info("All core services started. System is live. Press Ctrl+C to shutdown.")
        while True: 
            time.sleep(3600) 

    except SystemExit as se: 
        logger.critical(f"System Exit Triggered: {se}. Shutting down.")
    except KeyboardInterrupt:
        logger.info("Ctrl+C received. Initiating graceful shutdown...")
    except FileNotFoundError as fnf_e: 
        logger.critical(f"Essential file not found during startup: {fnf_e}. System cannot start.")
    except OSError as os_e: 
         logger.critical(f"OS Error during startup (e.g., port in use?): {os_e}. System cannot start.")
    except Exception as e:
        logger.critical(f"Unhandled critical error in main function: {e}", exc_info=True)
    finally:
        logger.info("System shutdown sequence commencing...")
        if pipeline_trader: 
            pipeline_trader.shutdown() 

        if bars_data_httpd: 
            logger.info("Stopping Bars Data HTTP Server...")
            try:
                threading.Thread(target=bars_data_httpd.shutdown, daemon=True).start()
                bars_data_httpd.server_close()
            except Exception as e:
                logger.error(f"Error shutting down Bars Data HTTPD: {e}")

        time.sleep(1) 
        active_threads = threading.enumerate()
        main_thread = threading.current_thread()
        other_threads = [t.name for t in active_threads if t != main_thread and t.is_alive()]
        if other_threads:
            logger.info(f"Remaining active threads: {other_threads}")
        else:
            logger.info("All main threads appear to have stopped.")
            
        logger.info("System shutdown process complete.")

if __name__ == "__main__":
    main()
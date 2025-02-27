import socket
import threading
import time
import logging
import json
from collections import deque
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import numpy as np
import ta
import joblib

# ----------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ----------------------------------------------------------------------------
HOST = "0.0.0.0"
HTTP_PORT = 8000

# Original DLL server port for MetaTrader
SOCKET_PORT = 12345
# ALLOWED_DEVICE_IPS removed – now we use a dynamic dictionary mapping IP->AccountID

# New ASP.NET server port and allowed IPs (update with your ASP.NET server's IP)
ASPNET_PORT = 12346
ALLOWED_ASPNET_IPS = ["178.169.181.27"]

MODEL_BUNDLE_PATH   = "lgb_model.pkl"  # The bundle should contain keys: "model", "bin_edges", "timeframe"

# Trade and risk parameters (thresholds now match offline logic)
PROB_THRESHOLD        = 0    # use same threshold as offline pipeline
# MIN_RR parameter removed as requested
MAX_CONCURRENT_TRADES = 10

# Global BIN_SIZE (will be updated from model bundle)
BIN_SIZE = 0.001

# ----------------------------------------------------------------------------
# 1) HELPER: LOAD HIGH-IMPACT NEWS FROM CSV
# ----------------------------------------------------------------------------
def load_high_impact_news_csv(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None)
        df.columns = [
            'date','time','currency','impact','news','v1','v2','v3','v4','extra'
        ]
        df['timestamp'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y/%m/%d %H:%M',
            utc=True
        )
        df = df[df['impact'] == 'H']  # filter only high-impact
        news_events = df['timestamp'].sort_values().tolist()
        print(f"[News] Loaded {len(news_events)} high-impact news from CSV (UTC).")
        return news_events
    except Exception as e:
        print(f"[News] Error loading CSV: {e}")
        return []

# ----------------------------------------------------------------------------
# 2) CALCULATE TIME UNTIL INVALID (MAINTENANCE OR NEWS)
# ----------------------------------------------------------------------------
def calc_until_invalid(
    bar_ts,
    news_events=None,
    pre_news_buffer=1,
    post_news_buffer=1,
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
    print(f"[DEBUG] calc_until_invalid: maintenance window={maint_start} -> {maint_end}")
    
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
    
    for idx, (ws, we) in enumerate(merged):
        print(f"[DEBUG] calc_until_invalid: invalid_window[{idx}]={ws} -> {we}")
    
    for wstart, wend in merged:
        if wstart <= bar_ts < wend:
            print("[DEBUG] calc_until_invalid: bar_ts is INSIDE invalid window.")
            return 0
        if bar_ts < wstart:
            delta = wstart - bar_ts
            minutes_until = max(0, int(delta.total_seconds() // 60))
            print(f"[DEBUG] calc_until_invalid: bar_ts is before invalid window, minutes_until={minutes_until}")
            return minutes_until
    
    next_day = bar_ts + timedelta(days=1)
    next_maint_start = next_day.replace(hour=maintenance_start, minute=0, second=0, microsecond=0)
    delta = next_maint_start - bar_ts
    minutes_until = max(0, int(delta.total_seconds() // 60))
    print(f"[DEBUG] calc_until_invalid: no upcoming invalid window, returning {minutes_until} until tomorrow's maintenance start")
    return minutes_until

# ----------------------------------------------------------------------------
# 3) FEATURE ENGINEERING (same as training/inference code)
# ----------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    df.sort_values('timestamp', inplace=True)
    
    if 'tick_volume' not in df.columns and 'volume' in df.columns:
        df['tick_volume'] = df['volume']
    
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

# ----------------------------------------------------------------------------
# 4) SOCKET SERVER FOR DLL/METATRADER (Port 12345)
# ----------------------------------------------------------------------------
class DllSocketServer:
    def __init__(self, pipeline_trader, host, port, allowed_devices):
        self.pipeline_trader = pipeline_trader
        self.host = host
        self.port = port
        self.allowed_devices = allowed_devices  # now a dictionary {ip: account_id}

        self.server_socket = None
        self.running = True

        # Now we store tuples: (conn, client_ip)
        self.connections = []
        self.conn_lock = threading.Lock()

    def start(self):
        t_accept = threading.Thread(target=self._accept_loop, daemon=True)
        t_accept.start()

        t_recv = threading.Thread(target=self._recv_loop, daemon=True)
        t_recv.start()

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _accept_loop(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        print(f"[SocketServer] Listening on {self.host}:{self.port}")

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                client_ip = addr[0]
                print(f"[SocketServer] Connection from {client_ip}:{addr[1]}")

                if client_ip in self.allowed_devices:
                    account_id = self.allowed_devices[client_ip]
                    with self.conn_lock:
                        # Store both connection and client IP
                        self.connections.append((conn, client_ip))
                    print(f"[SocketServer] Allowed device IP {client_ip} with Account ID {account_id} connected.")
                else:
                    print(f"[SocketServer] IP {client_ip} not allowed, closing.")
                    conn.close()

            except Exception as e:
                if self.running:
                    print(f"[SocketServer] Accept error: {e}")
                break

        print("[SocketServer] Accept loop stopped.")

    def _recv_loop(self):
        while self.running:
            with self.conn_lock:
                # iterate over a copy of the connections list
                for conn, client_ip in list(self.connections):
                    try:
                        conn.settimeout(0.001)
                        data = conn.recv(1024)
                        if data:
                            msg = data.decode('utf-8', errors='replace').strip()
                            print(f"[SocketServer] Received from {client_ip}: {msg}")

                            # Example: Process a "CLOSE|" message as before
                            if msg.startswith("CLOSE|"):
                                parts = msg.split("|")
                                if len(parts) >= 2:
                                    tid_str = parts[1].replace("ID=", "")
                                    try:
                                        tid = int(tid_str)
                                        self.pipeline_trader.on_position_closed(tid)
                                    except:
                                        pass

                            # Append the sender's IP to the message before forwarding
                            msg_with_ip = f"{msg}| FROM_IP={client_ip}"
                            self.pipeline_trader.forward_to_aspnet(msg_with_ip)
                    except socket.timeout:
                        pass
                    except Exception as e:
                        print(f"[SocketServer] Recv error: {e}")
                        self.connections.remove((conn, client_ip))
                        conn.close()
            time.sleep(0.05)

    def send_signal_to_clients(self, message):
        with self.conn_lock:
            for conn, client_ip in list(self.connections):
                try:
                    conn.sendall(message.encode('utf-8'))
                except Exception as e:
                    print(f"[SocketServer] Send error: {e}")
                    self.connections.remove((conn, client_ip))
                    conn.close()


# ----------------------------------------------------------------------------
# 4b) UPDATED: SOCKET SERVER FOR ASP.NET (Port 12346)
# ----------------------------------------------------------------------------
class AspNetSocketServer:
    def __init__(self, host, port, allowed_ips, pipeline_trader):
        self.host = host
        self.port = port
        self.allowed_ips = allowed_ips  # For ASP.NET connections, still using a static list
        self.pipeline_trader = pipeline_trader

        self.server_socket = None
        self.running = True

        self.connections = []
        self.conn_lock = threading.Lock()

    def start(self):
        t_accept = threading.Thread(target=self._accept_loop, daemon=True)
        t_accept.start()

        t_recv = threading.Thread(target=self._recv_loop, daemon=True)
        t_recv.start()

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _accept_loop(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"[AspNetSocketServer] Listening on {self.host}:{self.port}")

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                client_ip = addr[0]
                print(f"[AspNetSocketServer] Connection from {client_ip}:{addr[1]}")
                if client_ip in self.allowed_ips:
                    with self.conn_lock:
                        self.connections.append(conn)
                    print(f"[AspNetSocketServer] Allowed ASP.NET IP {client_ip} connected.")
                else:
                    print(f"[AspNetSocketServer] IP {client_ip} not allowed, closing.")
                    conn.close()
            except Exception as e:
                if self.running:
                    print(f"[AspNetSocketServer] Accept error: {e}")
                break

        print("[AspNetSocketServer] Accept loop stopped.")

    def _recv_loop(self):
        while self.running:
            with self.conn_lock:
                for conn in list(self.connections):
                    try:
                        conn.settimeout(0.001)
                        data = conn.recv(1024)
                        if data:
                            msg = data.decode('utf-8', errors='replace').strip()
                            print(f"[AspNetSocketServer] Received: {msg}")
                            # NEW: Process messages to update allowed devices
                            if msg.startswith("START"):
                                parts = msg.split()
                                if len(parts) >= 3:
                                    account_id = parts[1]
                                    allowed_ip = parts[2]
                                    # Update the allowed devices dictionary in pipeline_trader
                                    self.pipeline_trader.allowed_devices[allowed_ip] = account_id
                                    print(f"[AspNetSocketServer] Updated allowed_devices: {allowed_ip} -> {account_id}")
                    except socket.timeout:
                        pass
                    except Exception as e:
                        print(f"[AspNetSocketServer] Recv error: {e}")
                        self.connections.remove(conn)
                        conn.close()
            time.sleep(0.05)

    def send_signal_to_clients(self, message):
        with self.conn_lock:
            for conn in list(self.connections):
                try:
                    conn.sendall(message.encode('utf-8'))
                except Exception as e:
                    print(f"[AspNetSocketServer] Send error: {e}")
                    self.connections.remove(conn)
                    conn.close()

# ----------------------------------------------------------------------------
# 5) DATA BUFFER
# ----------------------------------------------------------------------------
class LiveDataBuffer:
    def __init__(self, max_size=300, news_events=None):
        self.max_size = max_size
        self.buffer   = deque()
        self.news_events = news_events or []

    def append(self, bar):
        bar_ts = pd.to_datetime(bar['timestamp'])
        bar_ts = bar_ts - timedelta(hours=2)  # Bulgarian time (UTC+2) -> UTC
        bar_ts = bar_ts.replace(tzinfo=timezone.utc)
        print(f"[DEBUG] LiveDataBuffer.append() new_bar: {bar}, parsed_ts={bar_ts}")

        until_inv = calc_until_invalid(
            bar_ts,
            news_events=self.news_events,
            pre_news_buffer=60,
            post_news_buffer=60,
            maintenance_start=22,
            maintenance_end=0,
        )

        row = {
            'timestamp': bar_ts,
            'open': float(bar['open']),
            'high': float(bar['high']),
            'low':  float(bar['low']),
            'close': float(bar['close']),
            'volume': float(bar['volume']),
            'tick_volume': float(bar['volume']),
            'until_invalid': until_inv
        }
        self.buffer.append(row)
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()

    def to_dataframe(self):
        return pd.DataFrame(list(self.buffer))

# ----------------------------------------------------------------------------
# 6) MODIFIED SIGNAL AGGREGATION (removed MIN_RR dependency)
# ----------------------------------------------------------------------------
def get_neg_pos_bin_indices(bin_edges: np.ndarray):
    neg_indices = []
    pos_indices = []
    num_bins = len(bin_edges) - 1
    for i in range(num_bins):
        left_edge = bin_edges[i]
        right_edge = bin_edges[i+1]
        if right_edge <= 0:
            neg_indices.append(i)
        elif left_edge >= 0:
            pos_indices.append(i)
    return neg_indices, pos_indices

def aggregate_signal(proba, price, bin_edges, model_classes, prob_threshold):
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

    # Modified to remove MIN_RR dependency for signal generation
    if (agg_pos - agg_neg) >= prob_threshold:
        signal = 1
    elif (agg_neg - agg_pos) >= prob_threshold:
        signal = -1
    else:
        signal = 0

    # Only calculate TP, not SL
    if signal == 1:
        tp = price * (1 + weighted_pos_move)
    elif signal == -1:
        tp = price * (1 - weighted_neg_move)
    else:
        tp = None

    return signal, weighted_pos_move, weighted_neg_move, tp

# ----------------------------------------------------------------------------
# 7) PIPELINE TRADER (modified to remove SL/TP calculations)
# ----------------------------------------------------------------------------
class LivePipelineTrader:
    def __init__(self, model_bundle, news_events=None):
        self.logger = logging.getLogger(__name__)
        self.news_events = news_events if news_events is not None else []
        self.data_buffer = LiveDataBuffer(max_size=300, news_events=self.news_events)

        self.model = model_bundle["model"]
        self.bin_edges = model_bundle["bin_edges"]
        self.timeframe = model_bundle["timeframe"]
        self.num_bins = len(self.model.classes_)

        self.feature_cols = [
            'open', 'high', 'low', 'close', 'tick_volume', 'rsi',
            'boll_upper', 'boll_middle', 'boll_lower', 'atr',
            'adx', 'adx_pos', 'adx_neg', 'obv', 'obv_ema',
            'donchian_high', 'donchian_low', 'donchian_mid',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5',
            'rolling_mean_10', 'rolling_std_10'
        ]

        self.active_trades = []
        self.max_concurrent_trades = MAX_CONCURRENT_TRADES

        self.account_risk_map = {
            125: 5.0,
            200: 2.0,
        }

        # NEW: Initialize allowed_devices as an empty dictionary {ip: account_id}
        self.allowed_devices = {}

        # Start the DLL/MetaTrader server (port 12345) using the dynamic allowed_devices dict
        self.socket_server = DllSocketServer(self, HOST, SOCKET_PORT, self.allowed_devices)
        self.socket_server.start()

        # Start the ASP.NET server (port 12346) and pass self so it can update allowed_devices
        self.asp_net_server = AspNetSocketServer(HOST, ASPNET_PORT, ALLOWED_ASPNET_IPS, self)
        self.asp_net_server.start()

    def on_new_bar(self, bar_dict):
        print(f"[DEBUG] on_new_bar called with: {bar_dict}")
        self.data_buffer.append(bar_dict)

        if len(self.data_buffer.buffer) < 50:
            print(f"[DEBUG] on_new_bar -> skip (buffer < 50: len={len(self.data_buffer.buffer)})")
            return

        df_buf = self.data_buffer.to_dataframe()
        df_feat = engineer_features(df_buf)
        print(f"[DEBUG] Data buffer size={len(df_buf)}, after feature eng shape={df_feat.shape}")

        df_feat.dropna(inplace=True)
        if df_feat.empty:
            print("[DEBUG] on_new_bar -> skip (df_feat is empty after dropna)")
            return

        last_row = df_feat.iloc[-1]
        print("[DEBUG] last_row from df_feat:")
        print(last_row.to_dict())

        try:
            self.logger.info(
                "[Bar] time=%s open=%.2f high=%.2f low=%.2f close=%.2f vol=%.1f rsi=%.2f atr=%.2f until_invalid=%d",
                last_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                last_row["open"],
                last_row["high"],
                last_row["low"],
                last_row["close"],
                last_row["tick_volume"],
                last_row["rsi"],
                last_row["atr"],
                int(last_row["until_invalid"])
            )
        except KeyError:
            pass

        if last_row['until_invalid'] == 0:
            print("[DEBUG] on_new_bar -> skip (until_invalid == 0)")
            return

        if len(self.active_trades) >= self.max_concurrent_trades:
            print(f"[DEBUG] on_new_bar -> skip (max trades open: {self.max_concurrent_trades})")
            return

        for col in self.feature_cols:
            if col not in last_row:
                print(f"[DEBUG] on_new_bar -> skip (missing feature col: {col})")
                return

        if last_row[self.feature_cols].isnull().any():
            print("[DEBUG] on_new_bar -> skip (some feature_cols are NaN)")
            return

        X_last = pd.DataFrame(last_row[self.feature_cols].values.reshape(1, -1), columns=self.feature_cols).astype(float)
        print(f"[DEBUG] X_last.shape={X_last.shape}, features={self.feature_cols}")

        proba = self.model.predict_proba(X_last)[0]
        print(f"[DEBUG] model.predict_proba -> shape={proba.shape}, proba={proba}")

        # Modified to include TP but not SL
        signal, w_pos, w_neg, tp = aggregate_signal(
            proba,
            last_row['close'],
            self.bin_edges,
            self.model.classes_,
            PROB_THRESHOLD
        )
        print(f"[DEBUG] signal = {signal}, weighted_pos_move = {w_pos}, weighted_neg_move = {w_neg}, tp = {tp}")
        if signal == 0:
            print("[DEBUG] on_new_bar -> No trade signal.")
            return

        trade_id = len(self.active_trades) + 1
        ttype = "BUY" if signal == 1 else "SELL"
        acct_id = 125  # example account id; adjust as needed
        risk_percent = self.account_risk_map.get(acct_id, 1.0)

        trade_info = {
            "id": trade_id,
            "type": ttype,
            "entry_ts": last_row['timestamp'],
            "entry_price": last_row['close'],
            "acct_id": acct_id,
            "risk": risk_percent,
            "weighted_pos_move": w_pos,
            "weighted_neg_move": w_neg,
            "take_profit": tp
        }
        self.active_trades.append(trade_info)
        self.logger.info(
            "[Trade] %s ID=%d, ACCT=%d, RISK=%.1f%%, Price=%.2f, TP=%.2f, w_pos=%.4f, w_neg=%.4f",
            ttype, trade_id, acct_id, risk_percent, last_row['close'], tp, w_pos, w_neg
        )

        # Modified signal message to remove ACCT, RISK, W_POS, and W_NEG
        msg = f"OPEN|ID={trade_id}|{ttype}|XAUUSD|TP={tp}"
        print(f"[DEBUG] SENDING SIGNAL: {msg}")
        self.socket_server.send_signal_to_clients(msg)
        # Forward the trade-open message to ASP.NET:
        self.forward_to_aspnet(msg)

    def on_position_closed(self, trade_id):
        print(f"[DEBUG] on_position_closed called with trade_id={trade_id}")
        for t in self.active_trades:
            if t["id"] == trade_id:
                self.active_trades.remove(t)
                self.logger.info(f"[Trade] ID={trade_id} closed.")
                print(f"[DEBUG] Trade ID={trade_id} closed and removed from active_trades.")
                return
        self.logger.warning(f"[Trade] on_position_closed: ID={trade_id} not found.")
        print(f"[DEBUG] on_position_closed -> trade_id={trade_id} NOT FOUND in active_trades.")

    def forward_to_aspnet(self, msg):
        # This method forwards any message received on the DLL server to the ASP.NET server.
        if self.asp_net_server:
            print(f"[DEBUG] Forwarding message to ASP.NET server: {msg}")
            self.asp_net_server.send_signal_to_clients(msg)

# ----------------------------------------------------------------------------
# 8) HTTP REQUEST HANDLER
# ----------------------------------------------------------------------------
class RequestHandler(BaseHTTPRequestHandler):
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
                "open":     json_data.get("open", 0.0),
                "high":     json_data.get("high", 0.0),
                "low":      json_data.get("low", 0.0),
                "close":    json_data.get("close", 0.0),
                "volume":   json_data.get("volume", 0.0)
            }
            bar_data["timestamp"] = bar_data["timestamp"].replace('.', '-')

            if self.pipeline_trader:
                self.pipeline_trader.on_new_bar(bar_data)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        except Exception as e:
            logging.error(f"[RequestHandler] Error in do_POST: {e}")
            self.send_response(400)
            self.end_headers()

# ----------------------------------------------------------------------------
# 9) RUN SERVER
# ----------------------------------------------------------------------------
def run_server(port=HTTP_PORT):
    global BIN_SIZE

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    try:
        model_bundle = joblib.load(MODEL_BUNDLE_PATH)
        logger.info("[Main] Model bundle loaded.")
    except Exception as ex:
        logger.error(f"[Main] Failed to load model bundle: {ex}")
        return

    print("[DEBUG] Data from model_bundle:")
    for k, v in model_bundle.items():
        print(f"  - {k}: {v}")

    if "bin_edges" in model_bundle:
        edges = model_bundle["bin_edges"]
        if len(edges) >= 2:
            new_bin_size = edges[1] - edges[0]
            print(f"[DEBUG] Updating BIN_SIZE from model_bundle: {new_bin_size}")
            BIN_SIZE = new_bin_size
        else:
            print("[DEBUG] bin_edges found but not enough values to compute bin_size.")
    else:
        print("[DEBUG] No bin_edges found in model_bundle, using default BIN_SIZE.")

    print(f"[DEBUG] BIN_SIZE after update: {BIN_SIZE}")

    news_events = load_high_impact_news_csv("high_impact_news.csv")
    pipeline_trader = LivePipelineTrader(model_bundle, news_events=news_events)
    RequestHandler.pipeline_trader = pipeline_trader

    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"[HTTPServer] Listening on port {port}...")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[HTTPServer] Shutting down...")

    pipeline_trader.socket_server.stop()
    pipeline_trader.asp_net_server.stop()
    httpd.server_close()

if __name__ == '__main__':
    run_server(port=HTTP_PORT)
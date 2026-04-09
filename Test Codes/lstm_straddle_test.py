#!/usr/bin/env python3
"""
STAD70 — LSTM Intraday Straddle Strategy: Test Script
=====================================================
Loads a pre-trained StraddleLSTM model and runs it on held-out test data.
Outputs:
  1. cashflow_table.csv  (one row per option-leg trade)
  2. Cumulative P&L summary and plot (cum_pnl.png)

Usage:
  python lstm_straddle_test.py
  (Expects spy_etf_test.csv, spy_opt_test.csv, and model .pth in the same dir)
"""

import math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ═══════════════════════════════════════════════════════════════════════════════
#  FIXED PARAMETERS  (identical to training — do NOT modify)
# ═══════════════════════════════════════════════════════════════════════════════
BAR_MINUTES       = 15
BARS_PER_DAY      = int(6.5 * 60 / BAR_MINUTES)          # 26
ANNUALISATION     = math.sqrt(252 * BARS_PER_DAY)

FEAT1_K           = [1, 4, 8, 13, 26]
MACD_SHORT        = [2, 4, 8]
MACD_LONG         = [8, 16, 32]
FEAT3_N           = [1, 5, 10, 20]

TRAJECTORY_LEN    = 10        # τ (from Optuna best trial)
SIGMA_TGT         = 0.15
VOL_EWM_SPAN      = BARS_PER_DAY

BID_ASK_HALF      = 0.02
TC_PER_UNIT       = BID_ASK_HALF * 2 * 100   # $4 per straddle unit (2 legs × $0.02 × 100 multiplier)

# Best hyperparameters from Optuna
HIDDEN_DIM        = 40
NUM_LAYERS        = 1
DROPOUT           = 0.5
INIT_THRESHOLD    = 0.35

# Model file (change this if your .pth has a different name)
MODEL_PATH        = "final_improved_model_Copy.pth"

FEATURE_COLS = (
    ["sigma_t"]
    + [f"norm_ret_{k}"  for k in FEAT1_K]
    + [f"macd_{s}_{l}" for s, l in zip(MACD_SHORT, MACD_LONG)]
    + [f"mom_{n}"       for n in FEAT3_N]
    + ["log_moneyness_call", "log_moneyness_put", "dte_frac"]
    + [f"spy_vol_{w}" for w in [5, 10, 20]]
)

TARGET_COLS = [
    "straddle_log_ret", "spy_log_ret", "sigma_t",
    "straddle_open",    "straddle_close",
    "call_open",        "call_close",
    "put_open",         "put_close",
    "spy_close",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION  (must match training exactly)
# ═══════════════════════════════════════════════════════════════════════════════
class StraddleLSTM(nn.Module):
    def __init__(self,
                 input_dim:      int   = len(FEATURE_COLS),
                 hidden_dim:     int   = HIDDEN_DIM,
                 num_layers:     int   = NUM_LAYERS,
                 dropout:        float = DROPOUT,
                 init_threshold: float = INIT_THRESHOLD):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln       = nn.LayerNorm(hidden_dim)
        self.fc_res1  = nn.Linear(hidden_dim, hidden_dim)
        self.fc_res2  = nn.Linear(hidden_dim, hidden_dim)
        self.dropout  = nn.Dropout(dropout)
        self.fc_out   = nn.Linear(hidden_dim, 1)
        self.log_threshold = nn.Parameter(
            torch.tensor(math.log(init_threshold), dtype=torch.float32)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:   nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.0)
                n = param.size(0)
                param.data[n//4 : n//2].fill_(1.0)
        for m in [self.fc_res1, self.fc_res2, self.fc_out]:
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: nn.init.zeros_(m.bias.data)

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = self.ln(lstm_out)
        res = torch.tanh(self.fc_res1(lstm_out))
        res = self.dropout(res)
        res = torch.tanh(self.fc_res2(res))
        lstm_out = torch.sigmoid(lstm_out) + res
        raw_sig = torch.tanh(self.fc_out(lstm_out))
        mask = (raw_sig.detach().abs() >= self.threshold).float()
        sig  = raw_sig * mask
        return sig, hidden, self.threshold

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class StraddleDataset(Dataset):
    def __init__(self, feat_df: pd.DataFrame, tau: int = TRAJECTORY_LEN):
        self.tau = tau
        self.X   = feat_df[FEATURE_COLS].values.astype(np.float32)
        self.y   = feat_df[TARGET_COLS].values.astype(np.float32)
        self.n   = len(feat_df) - tau + 1
        assert self.n > 0, f"Too few bars ({len(feat_df)}) for τ={tau}"
    def __len__(self):  return self.n
    def __getitem__(self, idx):
        sl = slice(idx, idx + self.tau)
        return torch.from_numpy(self.X[sl]), torch.from_numpy(self.y[sl]), idx

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & RESAMPLING
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_resample(spy_path: str, opt_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spy = pd.read_csv(spy_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    opt = pd.read_csv(opt_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()

    rule = f"{BAR_MINUTES}min"

    spy_bars = spy.resample(rule, label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),     close=("close", "last"),
        volume=("volume", "sum"), vwap=("vwap", "mean"),
    ).dropna(subset=["open", "close"])

    # Options: include vwap (volume-weighted mean approximated by mean)
    def _vwap_agg(g):
        if g["volume"].sum() > 0:
            return np.average(g["vwap"], weights=g["volume"])
        return g["vwap"].mean()

    opt_bars = (opt.groupby(["type", "strike"])
                   .resample(rule, label="left", closed="left")
                   .agg(open=("open", "first"), close=("close", "last"),
                        volume=("volume", "sum"), vwap=("vwap", "mean"))
                   .reset_index()
                   .dropna(subset=["open", "close"]))

    # Regular trading hours only
    spy_bars = spy_bars.between_time("09:30", "15:45")
    opt_bars = opt_bars[opt_bars["timestamp"].dt.time.between(
        pd.Timestamp("09:30").time(), pd.Timestamp("15:45").time())]

    return spy_bars, opt_bars

# ═══════════════════════════════════════════════════════════════════════════════
#  STRADDLE SERIES CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════
def build_straddle_series(spy_bars: pd.DataFrame,
                          opt_bars: pd.DataFrame) -> pd.DataFrame:
    records = []
    opt_by_ts = opt_bars.set_index("timestamp")

    for ts, spy_row in spy_bars.iterrows():
        spy_close = spy_row["close"]
        spy_open  = spy_row["open"]

        try:
            opt_t = opt_by_ts.loc[ts]
            if isinstance(opt_t, pd.Series):
                opt_t = opt_t.to_frame().T
            else:
                opt_t = opt_t.reset_index()
        except KeyError:
            continue

        has_both = (
            opt_t.groupby("strike")["type"]
                 .apply(lambda s: {"C", "P"}.issubset(set(s)))
        )
        valid_strikes = has_both[has_both].index.tolist()
        if not valid_strikes:
            continue

        atm_strike = min(valid_strikes, key=lambda k: abs(k - spy_open))
        if not (0.95 <= spy_open / atm_strike <= 1.05):
            continue

        call_row = opt_t[(opt_t["type"] == "C") & (opt_t["strike"] == atm_strike)].iloc[0]
        put_row  = opt_t[(opt_t["type"] == "P") & (opt_t["strike"] == atm_strike)].iloc[0]

        if any(pd.isna(v) or v <= 0 for v in [
                call_row["open"], call_row["close"],
                put_row["open"],  put_row["close"]]):
            continue

        market_close = ts.normalize() + pd.Timedelta(hours=15, minutes=59)
        minutes_left = max((market_close - ts).total_seconds() / 60, 1)

        # VWAP for options — use from resampled bar, fallback to (open+close)/2
        call_vwap = call_row.get("vwap", (call_row["open"] + call_row["close"]) / 2)
        put_vwap  = put_row.get("vwap",  (put_row["open"]  + put_row["close"]) / 2)
        if pd.isna(call_vwap) or call_vwap <= 0:
            call_vwap = (call_row["open"] + call_row["close"]) / 2
        if pd.isna(put_vwap) or put_vwap <= 0:
            put_vwap = (put_row["open"] + put_row["close"]) / 2

        records.append({
            "timestamp":          ts,
            "date":               ts.date(),
            "strike":             atm_strike,
            "call_open":          call_row["open"],
            "call_close":         call_row["close"],
            "put_open":           put_row["open"],
            "put_close":          put_row["close"],
            "call_vwap":          call_vwap,
            "put_vwap":           put_vwap,
            "straddle_open":      call_row["open"]  + put_row["open"],
            "straddle_close":     call_row["close"] + put_row["close"],
            "straddle_vwap":      call_vwap + put_vwap,
            "log_moneyness_call": math.log(spy_open / atm_strike),
            "log_moneyness_put":  math.log(atm_strike / spy_open),
            "dte_frac":           minutes_left / (252 * 390),
            "spy_close":          spy_close,
            "spy_open":           spy_open,
            "spy_log_ret":        math.log(spy_close / spy_open) if spy_open > 0 else 0.0,
        })

    df = pd.DataFrame(records).set_index("timestamp").sort_index()

    # Close-to-close returns within each day
    cc_ret = df.groupby("date")["straddle_close"].transform(lambda g: np.log(g / g.shift(1)))
    co_ret = np.log(df["straddle_close"] / df["straddle_open"])
    prev_strike = df.groupby("date")["strike"].shift(1)
    strike_changed = (prev_strike.notna()) & (df["strike"] != prev_strike)
    df["straddle_log_ret"] = np.where(strike_changed, co_ret, cc_ret)

    return df.dropna(subset=["straddle_log_ret"])

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
def _ewmsd(series, span):  return series.ewm(span=span, adjust=False).std()
def _ewma(series, halflife): return series.ewm(halflife=halflife, adjust=False).mean()
def _phi(y):  return y * np.exp(-y**2 / 4) / 0.89

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    ret  = df["straddle_log_ret"]

    sigma_t_raw = (
        df.groupby("date")["straddle_log_ret"]
          .transform(lambda g: g.ewm(span=VOL_EWM_SPAN, adjust=False).std())
          .shift(1)
    )
    sigma_t_filled = sigma_t_raw.fillna(sigma_t_raw.expanding().median())
    sigma_t = sigma_t_filled.where(sigma_t_filled.notna(), other=1e-4).clip(lower=1e-6)
    feat["sigma_t"] = sigma_t

    for k in FEAT1_K:
        rolling_ret = ret.shift(1).rolling(k, min_periods=1).sum()
        feat[f"norm_ret_{k}"] = (rolling_ret / (sigma_t * math.sqrt(k))).clip(-10, 10).fillna(0.0)

    price = df["straddle_close"]
    price_std5 = price.rolling(5, min_periods=2).std().shift(1)
    price_std5 = price_std5.fillna(price.expanding(min_periods=2).std()).clip(lower=1e-6)
    for s, l in zip(MACD_SHORT, MACD_LONG):
        hl_s = math.log(0.5) / math.log(1 - 1/s)
        hl_l = math.log(0.5) / math.log(1 - 1/l)
        ma_s = _ewma(price, hl_s).shift(1)
        ma_l = _ewma(price, hl_l).shift(1)
        macd_raw  = (ma_s - ma_l) / price_std5
        macd_std  = macd_raw.rolling(20, min_periods=5).std()
        macd_std  = macd_std.fillna(macd_raw.expanding(min_periods=5).std()).clip(lower=1e-6)
        macd_norm = (macd_raw / macd_std).clip(-3, 3).fillna(0.0)
        feat[f"macd_{s}_{l}"] = _phi(macd_norm.values)

    for n in FEAT3_N:
        feat[f"mom_{n}"] = ret.shift(1).rolling(n, min_periods=1).mean().fillna(0.0)

    feat["log_moneyness_call"] = df["log_moneyness_call"].fillna(0.0)
    feat["log_moneyness_put"]  = df["log_moneyness_put"].fillna(0.0)
    feat["dte_frac"]           = df["dte_frac"].fillna(0.0)

    for w in [5, 10, 20]:
        feat[f"spy_vol_{w}"] = df["spy_log_ret"].shift(1).rolling(w, min_periods=1).std().fillna(0.0)

    # Auxiliary columns
    feat["straddle_log_ret"] = df["straddle_log_ret"]
    feat["spy_log_ret"]      = df["spy_log_ret"].fillna(0.0)
    feat["spy_close"]        = df["spy_close"]
    feat["call_open"]        = df["call_open"]
    feat["call_close"]       = df["call_close"]
    feat["put_open"]         = df["put_open"]
    feat["put_close"]        = df["put_close"]
    feat["call_vwap"]        = df["call_vwap"]
    feat["put_vwap"]         = df["put_vwap"]
    feat["straddle_open"]    = df["straddle_open"]
    feat["straddle_close"]   = df["straddle_close"]
    feat["straddle_vwap"]    = df["straddle_vwap"]
    feat["date"]             = df["date"]
    feat["strike"]           = df["strike"]

    feat[FEATURE_COLS] = feat[FEATURE_COLS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feat.dropna(subset=["straddle_log_ret", "straddle_close", "spy_close"])

# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE  (run model → per-bar signals)
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(model: StraddleLSTM,
                  feat_df: pd.DataFrame,
                  device: torch.device) -> pd.DataFrame:
    """
    Produce per-bar signals and build a results DataFrame with all columns
    needed to generate the cashflow table.  No vol-scaling on P&L.
    """
    model.eval()
    feat_reset = feat_df.reset_index(drop=False)  # brings 'timestamp' back as column
    ds = StraddleDataset(feat_df, tau=TRAJECTORY_LEN)

    all_idx, all_sig = [], []
    for X, y, idx in DataLoader(ds, batch_size=128, shuffle=False):
        sig, _, _ = model(X.to(device))
        all_sig.append(sig[:, -1, 0].cpu().numpy())
        all_idx.append(idx.numpy() + TRAJECTORY_LEN - 1)

    all_sig = np.concatenate(all_sig)
    all_idx = np.concatenate(all_idx)

    needed = ["timestamp", "sigma_t", "straddle_close", "straddle_open",
              "straddle_vwap", "call_vwap", "put_vwap",
              "spy_close", "date", "strike"]
    res = feat_reset.iloc[all_idx][needed].copy().reset_index(drop=True)
    res["signal"] = all_sig

    # Force flat at first/last bar of each day
    for grp_idx in res.groupby("date", sort=False).groups.values():
        idx_list = list(grp_idx)
        res.loc[idx_list[0],  "signal"] = 0.0
        res.loc[idx_list[-1], "signal"] = 0.0

    # Position held during bar t was decided at bar t-1
    res["held_pos"]    = res["signal"].shift(1).fillna(0.0)
    res["prev_strike"] = res["strike"].shift(1)
    res["strike_changed"] = (
        res["prev_strike"].notna()
        & (res["prev_strike"].astype(str) != res["strike"].astype(str))
    )

    return res

# ═══════════════════════════════════════════════════════════════════════════════
#  CASHFLOW TABLE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def build_cashflow_table(res: pd.DataFrame) -> pd.DataFrame:
    """
    Build the required cashflow_table with one row per option-leg trade.

    Raw P&L (no vol-scaling). TC is embedded by adjusting the execution price:
      - Buy  at VWAP + BID_ASK_HALF
      - Sell at VWAP − BID_ASK_HALF

    Columns: timestamp, asset, position, vwap_price, cashflow, cum_pnl
    """
    rows = []

    for i in range(len(res)):
        ts       = res.loc[i, "timestamp"]
        strike   = res.loc[i, "strike"]
        signal   = res.loc[i, "signal"]
        held_pos = res.loc[i, "held_pos"]
        sc       = res.loc[i, "strike_changed"]

        call_vwap = res.loc[i, "call_vwap"]
        put_vwap  = res.loc[i, "put_vwap"]

        ts_str = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")

        if sc and abs(held_pos) > 1e-9:
            # Strike changed — close old position at previous bar's VWAP
            # Use current bar's VWAP as proxy (conservative; old strike closed)
            close_amt = -held_pos  # opposite sign to close
            sign_close = 1 if close_amt > 0 else -1  # +1=buy, -1=sell
            # Closing the old position: we sell if held_pos>0, buy if held_pos<0
            prev_strike = res.loc[i, "prev_strike"]
            # Use current VWAP as best available price for the close
            c_exec = call_vwap - BID_ASK_HALF if held_pos > 0 else call_vwap + BID_ASK_HALF
            p_exec = put_vwap  - BID_ASK_HALF if held_pos > 0 else put_vwap  + BID_ASK_HALF
            c_exec = max(c_exec, 0)
            p_exec = max(p_exec, 0)

            rows.append({
                "timestamp":  ts_str,
                "asset":      f"Call K={prev_strike:.0f}",
                "position":   round(close_amt, 6),
                "vwap_price": round(call_vwap, 4),
                "cashflow":   round(close_amt * c_exec * (-100), 4),
            })
            rows.append({
                "timestamp":  ts_str,
                "asset":      f"Put K={prev_strike:.0f}",
                "position":   round(close_amt, 6),
                "vwap_price": round(put_vwap, 4),
                "cashflow":   round(close_amt * p_exec * (-100), 4),
            })
            # After closing, the effective held_pos going into the new-strike open is 0
            held_pos = 0.0

        # Position change for this bar
        delta = signal - held_pos
        if abs(delta) < 1e-9:
            continue

        # Determine execution price with spread
        if delta > 0:  # buying
            c_exec = call_vwap + BID_ASK_HALF
            p_exec = put_vwap  + BID_ASK_HALF
        else:          # selling
            c_exec = max(call_vwap - BID_ASK_HALF, 0)
            p_exec = max(put_vwap  - BID_ASK_HALF, 0)

        # cashflow: buying is cash out (negative), selling is cash in (positive)
        # delta > 0 → buying → cashflow = -delta * exec * 100
        # delta < 0 → selling → cashflow = -delta * exec * 100  (positive since delta<0)
        rows.append({
            "timestamp":  ts_str,
            "asset":      f"Call K={strike:.0f}",
            "position":   round(delta, 6),
            "vwap_price": round(call_vwap, 4),
            "cashflow":   round(-delta * c_exec * 100, 4),
        })
        rows.append({
            "timestamp":  ts_str,
            "asset":      f"Put K={strike:.0f}",
            "position":   round(delta, 6),
            "vwap_price": round(put_vwap, 4),
            "cashflow":   round(-delta * p_exec * 100, 4),
        })

    cf = pd.DataFrame(rows)
    if len(cf) == 0:
        cf = pd.DataFrame(columns=["timestamp", "asset", "position",
                                    "vwap_price", "cashflow", "cum_pnl"])
    else:
        cf["cum_pnl"] = cf["cashflow"].cumsum().round(4)
    return cf

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# dir_spy_path = "spy_etf_train.csv"
# dir_opt_path = "spy_opt_train.csv"

dir_spy_path = "spy_etf_test.csv"
dir_opt_path = "spy_opt_test.csv"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    spy_path = os.path.join(script_dir, dir_spy_path)
    opt_path = os.path.join(script_dir, dir_opt_path)
    model_path = os.path.join(script_dir, MODEL_PATH)

    for p in [spy_path, opt_path, model_path]:
        if not os.path.isfile(p):
            sys.exit(f"ERROR: file not found — {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Bar length: {BAR_MINUTES} min | Bars/day: {BARS_PER_DAY} | τ: {TRAJECTORY_LEN}")

    # ── 1. Load & resample ────────────────────────────────────────────────────
    print("Loading and resampling data …")
    spy_bars, opt_bars = load_and_resample(spy_path, opt_path)
    print(f"  SPY bars: {len(spy_bars)}  |  OPT bars: {len(opt_bars)}")

    # ── 2. Build straddle series ──────────────────────────────────────────────
    straddle_df = build_straddle_series(spy_bars, opt_bars)
    print(f"  Straddle bars: {len(straddle_df)}")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    feat_df = build_features(straddle_df)
    print(f"  Feature bars: {len(feat_df)}  |  Feature dim d = {len(FEATURE_COLS)}")

    # ── 4. Load model ─────────────────────────────────────────────────────────
    print(f"Loading model from {model_path} …")
    model = StraddleLSTM(
        input_dim=len(FEATURE_COLS),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        init_threshold=INIT_THRESHOLD,
    ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Threshold θ = {model.threshold.item():.4f}")

    # ── 5. Run inference ──────────────────────────────────────────────────────
    print("Running inference …")
    res = run_inference(model, feat_df, device)
    print(f"  Bars with signal: {(res['signal'].abs() > 1e-9).sum()} / {len(res)}")

    # ── 6. Build & save cashflow table ────────────────────────────────────────
    cf = build_cashflow_table(res)
    out_csv = os.path.join(script_dir, "cashflow_table_LSTM.csv")
    cf.to_csv(out_csv, index=False)
    print(f"\nSaved {len(cf)} trade rows → {out_csv}")

    if len(cf) > 0:
        final_pnl = cf["cum_pnl"].iloc[-1]
        print(f"\n{'='*50}")
        print(f"  CUMULATIVE P&L : ${final_pnl:.2f}")
        print(f"  Total trades   : {len(cf)}")
        print(f"{'='*50}")

        # ── 7. Cumulative P&L plot ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(cf)), cf["cum_pnl"], color="steelblue", lw=1.2)
        ax.axhline(0, color="red", ls="--", lw=0.8)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title("LSTM Straddle Strategy — Cumulative P&L (Test Set)")
        fig.tight_layout()
        plot_path = os.path.join(script_dir, "cum_pnl_LSTM.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot → {plot_path}")
        plt.close(fig)
    else:
        print("WARNING: No trades generated.")


if __name__ == "__main__":
    main()

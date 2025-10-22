#!/usr/bin/env python3
"""
crypto_ai_advanced.py - Revisado
Pipeline ML para analisar criptomoedas com LightGBM, indicadores técnicos, backtest e sinais reais.
"""

import argparse
import ccxt #type:ignore
import pandas as pd
import numpy as np
import ta
import joblib
import datetime as dt
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb #type:ignore
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Try import optuna if available
try:
    import optuna #type:ignore
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# ---------- Config ----------
DEFAULT_EXCHANGE = "binance"
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LIMIT = 2000
MODEL_FILE_DEFAULT = "crypto_lgbm_model.pkl"
RANDOM_STATE = 42

# ---------- Fetch OHLCV ----------
def fetch_ohlcv_binance(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME, limit=DEFAULT_LIMIT, exchange_name=DEFAULT_EXCHANGE):
    ex = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df["timestamp"] = df["timestamp"].dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')
    df.set_index('timestamp', inplace=True)
    return df

# ---------- Awesome Oscillator ----------
def add_awesome_oscillator(df):
    median_price = (df['high'] + df['low']) / 2
    sma5 = median_price.rolling(5).mean()
    sma34 = median_price.rolling(34).mean()
    df['mom'] = sma5 - sma34
    return df

# ---------- Technical Indicators ----------
def add_technical_indicators(df):
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']

    # Moving averages
    df['ma7'] = close.rolling(7).mean()
    df['ma21'] = close.rolling(21).mean()
    df['ema12'] = close.ewm(span=12, adjust=False).mean()
    df['ema26'] = close.ewm(span=26, adjust=False).mean()
    df['wma50'] = close.rolling(50).apply(lambda x: np.arange(1,len(x)+1).dot(x)/np.arange(1,len(x)+1).sum(), raw=True)

    # MACD
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    df['rsi14'] = ta.momentum.rsi(close, window=14, fillna=False)

    # Stochastic RSI
    df['stoch_rsi_k'] = ta.momentum.stochrsi(close, window=14, smooth1=3, smooth2=3)
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_h'] = bb.bollinger_hband()
    df['bb_m'] = bb.bollinger_mavg()
    df['bb_l'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_h'] - df['bb_l']) / (df['bb_m'] + 1e-9)

    # ATR, ADX
    df['atr14'] = ta.volatility.average_true_range(high, low, close, window=14)
    df['adx14'] = ta.trend.adx(high, low, close, window=14)

    # OBV
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()

    # CCI
    df['cci'] = ta.trend.cci(high, low, close, window=20)

    # ROC / momentum
    df['roc'] = ta.momentum.roc(close, window=12)

    df = add_awesome_oscillator(df)

    # Volume aggregates
    df['vol_rolling_mean_20'] = vol.rolling(20).mean()
    df['vol_rolling_mean_50'] = vol.rolling(50).mean()

    # Price returns
    df['return_1'] = close.pct_change(1)
    df['return_3'] = close.pct_change(3)
    df['return_7'] = close.pct_change(7)

    # EMA/MA spreads
    df['ema_diff'] = df['ema12'] - df['ema26']
    df['ma_ratio'] = df['ma7'] / (df['ma21'] + 1e-9)

    df = df.replace([np.inf,-np.inf], np.nan).dropna()
    return df

# ---------- Feature matrix ----------
def make_feature_matrix(df, n_lags=3):
    df = df.copy()
    features = []
    for c in df.columns:
        if c in ['open','high','low','close','volume']:
            continue
        features.append(c)
        for lag in range(1,n_lags+1):
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
            features.append(f"{c}_lag{lag}")
    df = df.dropna()
    return df, features

# ---------- Label creation ----------
def create_labels(df, forward_horizon=12, threshold_up=0.02, threshold_down=-0.02):
    df = df.copy()
    df['future_max'] = df['close'].shift(-1).rolling(window=forward_horizon,min_periods=1).max()
    df['future_min'] = df['close'].shift(-1).rolling(window=forward_horizon,min_periods=1).min()
    df['future_max_ret'] = (df['future_max'] - df['close']) / df['close']
    df['future_min_ret'] = (df['future_min'] - df['close']) / df['close']

    def row_label(r):
        if r['future_max_ret'] >= threshold_up:
            return 1
        elif r['future_min_ret'] <= threshold_down:
            return -1
        else:
            return 0

    df['label'] = df.apply(row_label, axis=1)
    df = df.dropna()
    return df

# ---------- Train LightGBM ----------
def train_lightgbm(df, feature_cols, n_splits=5, tune=False):
    X = df[feature_cols]
    y = df['label'].astype(int)

    # Remap labels: -1 -> 0, 0 -> 1, 1 -> 2
    label_map = {-1:0, 0:1, 1:2}
    y = y.map(label_map)

    tss = TimeSeriesSplit(n_splits=n_splits)
    best_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': -1
    }

    if tune and OPTUNA_AVAILABLE:
        def objective(trial):
            param = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq',1,7),
                'lambda_l1': trial.suggest_float('lambda_l1',0,5),
                'lambda_l2': trial.suggest_float('lambda_l2',0,5),
                'objective': 'multiclass',
                'num_class':3,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbosity': -1
            }
            scores = []
            for train_idx, test_idx in tss.split(X):
                Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
                ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
                dtrain = lgb.Dataset(Xtr, label=ytr)
                dval = lgb.Dataset(Xte, label=yte, reference=dtrain)
                bst = lgb.train(param, dtrain)
                preds = np.argmax(bst.predict(Xte), axis=1)
                acc = (preds == yte).mean()
                scores.append(acc)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        best_params.update(study.best_params)
        print("Optuna melhor resultado:", study.best_params)

    # Treinar modelo final
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(best_params, dtrain)
    return model, label_map

# ---------- Forecast Future ----------
def forecast_future_signals(model, df, feature_cols, label_map, n_future=12, timeframe="1h"):
    df_future = df.copy()
    last_time = df_future.index[-1]
    inv_label_map = {v:k for k,v in label_map.items()}
    future_data = []

    unit = timeframe[-1]  # h, m, d
    step = int(timeframe[:-1])

    for i in range(n_future):
        X_last = df_future[feature_cols].iloc[-1:].values
        preds_proba = model.predict(X_last)
        pred_idx = preds_proba.argmax(axis=1)[0]
        signal = inv_label_map[pred_idx]

        next_time = last_time + pd.to_timedelta(step, unit=unit)
        new_row = df_future.iloc[-1:].copy()
        new_row.index = [next_time]
        new_row['sinal_previsto'] = signal
        df_future = pd.concat([df_future, new_row])
        future_data.append((next_time, signal))
        last_time = next_time

    future_df = pd.DataFrame(future_data, columns=['timestamp','sinal_previsto']).set_index('timestamp')
    return future_df

# ---------- Main pipeline ----------
def build_and_run(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME, limit=DEFAULT_LIMIT,
    tune=False, forward_horizon=12, threshold_up=0.02, threshold_down=-0.02,
    model_file=MODEL_FILE_DEFAULT, no_backtest=False, predict_future=False):

    df = fetch_ohlcv_binance(symbol, timeframe, limit)
    df = add_technical_indicators(df)
    df = create_labels(df, forward_horizon, threshold_up, threshold_down)
    df, feature_cols = make_feature_matrix(df)
    model, label_map = train_lightgbm(df, feature_cols, tune=tune)
    joblib.dump(model, model_file)

    preds, _ = predict_lgbm(model, df, feature_cols, label_map)
    df['sinal_previsto'] = preds

    if not no_backtest:
        acc = (df['sinal_previsto'] == df['label']).mean()
        print(f"Backtest accuracy: {acc*100:.2f}%")

    future_df = None
    if predict_future:
        future_df = forecast_future_signals(model, df, feature_cols, label_map, n_future=forward_horizon, timeframe=timeframe)

    return {
        "df": df,
        "model": model,
        "label_map": label_map,
        "feature_cols": feature_cols,
        "future_df": future_df
    }


# ---------- Predict ----------
def predict_lgbm(model, df, feature_cols, label_map):
    X = df[feature_cols]
    preds_proba = model.predict(X)
    preds_idx = np.argmax(preds_proba, axis=1)
    inv_label_map = {v:k for k,v in label_map.items()}
    preds = np.array([inv_label_map[i] for i in preds_idx])
    return preds, preds_proba


# ---------- Run script ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--limit", default=DEFAULT_LIMIT, type=int)
    parser.add_argument("--tune", default=False, action='store_true')
    parser.add_argument("--predict_future", default=False, action='store_true')
    args = parser.parse_args()

    build_and_run(symbol=args.symbol, timeframe=args.timeframe, limit=args.limit,
    tune=args.tune, predict_future=args.predict_future)
import ccxt
import pandas as pd
import numpy as np
import time
import json
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
import statistics
from enum import Enum
from numba import njit

def detect_real_time_pivots(df, window=5):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    pivots = []
    for i in range(window, len(df) - window):
        left = highs[i - window:i]
        right = highs[i + 1:i + window + 1]
        if highs[i] > max(left) and highs[i] > max(right):
            pivots.append({'index': i, 'pivot_confirmed': 'high'})
            continue
        left = lows[i - window:i]
        right = lows[i + 1:i + window + 1]
        if lows[i] < min(left) and lows[i] < min(right):
            pivots.append({'index': i, 'pivot_confirmed': 'low'})
    return pd.DataFrame(pivots)

def second_derivative(arr):
    return np.gradient(np.gradient(arr))

def curvature_based_peaks(df, sensitivity=1.0):
    close = df['close'].values
    deriv2 = second_derivative(close)
    peaks = []
    for i in range(2, len(deriv2) - 2):
        if deriv2[i-1] > 0 and deriv2[i] < 0 and close[i] > close[i-1] and close[i] > close[i+1]:
            fe = compute_fractal_energy(close[i-14:i+1])
            if fe > 0.6:
                peaks.append({'index': i, 'pivot_confirmed': 'high'})
        elif deriv2[i-1] < 0 and deriv2[i] > 0 and close[i] < close[i-1] and close[i] < close[i+1]:
            fe = compute_fractal_energy(close[i-14:i+1])
            if fe > 0.6:
                peaks.append({'index': i, 'pivot_confirmed': 'low'})
    return pd.DataFrame(peaks)

def compute_fractal_energy(close, length=14):
    import math
    if len(close) < length:
        return 0
    log_hl = np.log(np.max(close[-length:]) / np.min(close[-length:]))
    sum_log_diff = np.sum(np.abs(np.diff(np.log(close[-length:]))))
    if sum_log_diff == 0:
        return 0
    return log_hl / sum_log_diff

@njit
def detect_curvature_fractal_peaks(prices, window=5):
    hma = adaptive_hma(prices, length=9)
    curvature = np.gradient(np.gradient(hma))
    peaks = []
    for i in range(window, len(prices)-window):
        if curvature[i-1] > 0 and curvature[i] < 0:
            peaks.append(i)
    return np.array(peaks)

@njit
def classify_wave_structure(pivot_indices):
    structure = []
    for i in range(0, len(pivot_indices)-4, 2):
        wave = pivot_indices[i:i+5]
        if len(wave) == 5:  
            structure.append(i)
        elif len(wave) == 3:  
            structure.append(i)
    return np.array(structure)

@njit
def detect_nested_waves(wave_segment, level=1, max_depth=3, min_wave_length=5):
    if level > max_depth or len(wave_segment) < min_wave_length:
        return np.array([])
    internal_pivots = detect_curvature_fractal_peaks(wave_segment)
    return internal_pivots

def generate_dynamic_rr(entry_price, stop_loss, target_price):
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    rr_ratio = round(reward / risk, 3) if risk != 0 else None
    return {
        "entry": round(entry_price, 2),
        "sl": round(stop_loss, 2),
        "tp": round(target_price, 2),
        "rr_ratio": rr_ratio
    }

@njit
def compute_curvature(series, window=5):
    smoothed = np.zeros(len(series))
    for i in range(len(series)):
        start = max(0, i - window // 2)
        end = min(len(series), i + window // 2 + 1)
        smoothed[i] = np.mean(series[start:end])
    first_derivative = np.gradient(smoothed)
    second_derivative = np.gradient(first_derivative)
    curvature = np.zeros(len(second_derivative))
    for i in range(len(second_derivative)):
        start = max(0, i - window // 2)
        end = min(len(second_derivative), i + window // 2 + 1)
        curvature[i] = np.mean(second_derivative[start:end])
    return curvature

@njit
def is_real_time_fractal_peak(i, curvature, threshold):
    if i < 2 or i > len(curvature) - 3:
        return False
    left = curvature[i - 2:i]
    right = curvature[i + 1:i + 3]
    center = curvature[i]
    return center > threshold and center > np.max(left) and center > np.max(right)

@njit
def is_real_time_fractal_valley(i, curvature, threshold):
    if i < 2 or i > len(curvature) - 3:
        return False
    left = curvature[i - 2:i]
    right = curvature[i + 1:i + 3]
    center = curvature[i]
    return center < -threshold and center < np.min(left) and center < np.min(right)

@njit
def fibonacci_extension(low_point, high_point, retrace_point, ratio=1.618):
    return retrace_point + (high_point - low_point) * ratio

def is_bullish_engulfing(curr, prev):
    return prev['close'] < prev['open'] and \
           curr['close'] > curr['open'] and \
           curr['close'] > prev['open'] and \
           curr['open'] < prev['close']

def is_bearish_engulfing(curr, prev):
    return prev['close'] > prev['open'] and \
           curr['close'] < curr['open'] and \
           curr['close'] < prev['open'] and \
           curr['open'] > prev['close']

def is_pin_bar(candle):
    body = abs(candle['close'] - candle['open'])
    upper_shadow = candle['high'] - max(candle['close'], candle['open'])
    lower_shadow = min(candle['close'], candle['open']) - candle['low']
    return (upper_shadow > body * 2) or (lower_shadow > body * 2)

def is_doji(candle):
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    return body < total_range * 0.1

def is_hammer(candle):
    body = abs(candle['close'] - candle['open'])
    lower_shadow = min(candle['open'], candle['close']) - candle['low']
    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
    ratio = lower_shadow / (body + 1e-5)
    return ratio > 2.5 and upper_shadow < body * 0.3

def is_tweezer_top(curr, prev):
    body1 = abs(prev['close'] - prev['open'])
    body2 = abs(curr['close'] - curr['open'])
    return (
        prev['high'] == curr['high'] and
        prev['close'] > prev['open'] and
        curr['close'] < curr['open'] and
        body1 > 0 and body2 > 0
    )

def is_tweezer_bottom(curr, prev):
    body1 = abs(prev['close'] - prev['open'])
    body2 = abs(curr['close'] - curr['open'])
    return (
        prev['low'] == curr['low'] and
        prev['close'] < prev['open'] and
        curr['close'] > curr['open'] and
        body1 > 0 and body2 > 0
    )

def is_inside_bar(curr, prev):
    return (
        curr['high'] < prev['high'] and
        curr['low'] > prev['low']
    )

def is_morning_star(c1, c2, c3):
    body1 = abs(c1['close'] - c1['open'])
    body2 = abs(c2['close'] - c2['open'])
    body3 = abs(c3['close'] - c3['open'])
    return (
        c1['close'] < c1['open'] and
        body2 < body1 * 0.5 and
        c3['close'] > c3['open'] and
        c3['close'] > (c1['open'] + c1['close']) / 2
    )

@njit
def detect_real_time_pivot(price_series, curvature_threshold=0.01, window=5):
    curvature = np.gradient(np.gradient(price_series))
    pivot_highs = np.zeros(len(price_series), dtype=np.bool_)
    pivot_lows = np.zeros(len(price_series), dtype=np.bool_)
    for i in range(window, len(price_series) - window):
        if curvature[i] < -curvature_threshold:
            pivot_highs[i] = True
        if curvature[i] > curvature_threshold:
            america = True
    return pivot_highs, pivot_lows

@njit
def ultra_adaptive_ema(prices, length=21, sensitivity=1.0):
    ema = np.zeros(len(prices))
    curvature = compute_curvature(prices)
    atr_like = np.std(prices[-length:])
    for i in range(len(prices)):
        weight = min(max(abs(curvature[i]) / (atr_like + 1e-6), 0.01), 1.0)
        alpha = (2 / (length + 1)) * weight * sensitivity
        if i == 0:
            ema[i] = prices[i]
        else:
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

@njit
def ultra_adaptive_hma(prices, length=21, sensitivity=1.0):
    curvature = compute_curvature(prices)
    weight = np.clip(np.abs(curvature) / (np.std(prices[-length:]) + 1e-6), 0.1, 1.0)
    half_len = int(length / 2)
    sqrt_len = int(np.sqrt(length))
    wma1 = np.convolve(prices * weight, np.ones(half_len) / half_len, mode='same')
    wma2 = np.convolve(prices * weight, np.ones(length) / length, mode='same')
    raw_hma = 2 * wma1 - wma2
    hma = np.convolve(raw_hma, np.ones(sqrt_len) / sqrt_len, mode='same')
    return hma

@njit
def adaptive_hma(prices, length=9):
    wma_short = np.convolve(prices, np.ones(length) / length, mode='valid')
    wma_long = np.convolve(prices, np.ones(length * 2) / (length * 2), mode='valid')
    raw_hma = 2 * wma_short - wma_long[:len(wma_short)]
    sqrt_len = int(np.sqrt(length))
    padding = np.zeros(len(prices) - len(raw_hma))
    padded_raw_hma = np.concatenate([raw_hma, padding])
    hma = np.convolve(padded_raw_hma, np.ones(sqrt_len) / sqrt_len, mode='same')
    diff_1 = np.diff(prices)
    diff_2 = np.diff(diff_1)
    curvature_strength = np.pad(np.abs(diff_2), (2,0), 'constant', curvature_strength=0)
    curvature_threshold = np.percentile(curvature_strength, 70)
    hma[curvature_strength > curvature_threshold] *= 0.9
    return hma

@njit
def realtime_peak_valley_v2(price_series, curvature_series, threshold=0.002, window=2):
    pivot_labels = np.array([0] * len(price_series))  
    for i in range(window, len(price_series) - window):
        left = price_series[i - window:i]
        right = price_series[i + 1:i + 1 + window]
        center = price_series[i]
        if center > np.max(left) and center > np.max(right) and curvature_series[i] < -threshold:
            pivot_labels[i] = 1
        elif center < np.min(left) and center < np.min(right) and curvature_series[i] > threshold:
            pivot_labels[i] = -1
    return pivot_labels

@njit
def adaptive_fractal_filter(price_series, window=3, deviation_threshold=0.5):
    pivots = []
    for i in range(window, len(price_series) - window):
        local_range = np.max(price_series[i - window:i + window + 1]) - np.min(price_series[i - window:i + window + 1])
        local_std = np.std(price_series[i - window:i + window + 1])
        if local_range > 0 and local_std / local_range < deviation_threshold:
            pivots.append(i)
    return np.array(pivots)

@njit
def calculate_wave_structure(prices):
    pivots = detect_swing_pivots(prices)
    return classify_wave_structure(pivots)

@njit
def detect_swing_pivots(prices, left=3, right=3):
    pivots = []
    for i in range(left, len(prices) - right):
        if prices[i] == np.max(prices[i-left:i+right+1]) or prices[i] == np.min(prices[i-left:i+right+1]):
            pivots.append(i)
    return np.array(pivots)

@njit
def ema_adaptive(prices, length=14):
    ema = np.zeros(len(prices))
    alpha = 2 / (length + 1)
    for i in range(len(prices)):
        if i == 0:
            ema[i] = prices[i]
        else:
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

@njit
def detect_fibonacci_zone(prices):
    pivots = detect_swing_pivots(prices)
    zones = []
    for i in range(1, len(pivots)):
        low = prices[pivots[i-1]]
        high = prices[pivots[i]]
        diff = high - low
        zones.append(low + diff * 0.618)
    return np.array(zones)

@njit
def wave_pattern_detector(prices):
    pivots = detect_swing_pivots(prices)
    return classify_wave_structure(pivots)

def validate_wave3_strength(df, wave1, wave3):
    rsi = df['rsi']
    adx = df['adx']
    plus_di = df['plus_di']
    minus_di = df['minus_di']
    start1, end1 = wave1['start'], wave1['end']
    start3, end3 = wave3['start'], wave3['end']
    avg_rsi_1 = np.mean(rsi.iloc[start1:end1+1].values)
    avg_rsi_3 = np.mean(rsi.iloc[start3:end3+1].values)
    avg_adx_1 = np.mean(adx.iloc[start1:end1+1].values)
    avg_adx_3 = np.mean(adx.iloc[start3:end3+1].values)
    strong_trend = all(plus_di.iloc[i] > minus_di.iloc[i] for i in range(start3, end3))
    rsi_trending_up = rsi[end3] > rsi[start3]
    adx_trending_up = adx[end3] > adx[start3]
    return (
        avg_rsi_3 > avg_rsi_1 and
        avg_adx_3 > avg_adx_1 and
        strong_trend and
        rsi_trending_up and
        adx_trending_up
    )

def is_not_impulsive_advanced(w1, w2, w3, w4, w5, rsi, adx):
    w3_vs_w1 = (w3['length'] / w1['length']) if w1['length'] != 0 else 0
    w5_vs_w1 = (w5['length'] / w1['length']) if w1['length'] != 0 else 0
    fib_ok = 1.382 <= w3_vs_w1 <= 2.618 and 0.618 <= w5_vs_w1 <= 1.618
    overlap_ok = w4['low'] > w1['high']
    momentum_ok = adx[w3['end']] > 25
    divergence_ok = rsi[w5['end']] < rsi[w3['end']]
    structure_ok = w1['subwaves'] == 5 and w3['subwaves'] == 5 and w5['subwaves'] == 5
    is_impulse = fib_ok and overlap_ok and momentum_ok and structure_ok
    return not is_impulse

def detect_wxy_wxyz_structure(pivot_prices, pivot_indexes):
    structures = []
    for i in range(len(pivot_prices) - 6):
        p0, p1, p2, p3, p4, p5 = pivot_prices[i:i + 6]
        i0, i1, i2, i3, i4, i5 = pivot_indexes[i:i + 6]
        is_w = 0.3 < retrace_pct(p0, p1, p2) < 0.7
        is_x1 = abs(p3 - p2) < abs(p2) * 0.382
        is_y = 0.3 < retrace_pct(p3, p4, p5) < 0.7
        if is_w and is_x1 and is_y:
            structures.append({
                'type': 'WXY',
                'points': [i0, i1, i2, i3, i4, i5],
                'prices': [p0, p1, p2, p3, p4, p5]
            })
        if i + 8 < len(pivot_prices):
            p6, p7 = pivot_prices[i+6:i+8]
            i6, i7 = pivot_indexes[i+6:i+8]
            is_x2 = abs(p6 - p5) < abs(p5) * 0.382
            is_z = 0.3 < retrace_pct(p6, p7, p7) < 0.7
            if is_w and is_x1 and is_y and is_x2 and is_z:
                structures.append({
                    'type': 'WXYXZ',
                    'points': [i0, i1, i2, i3, i4, i5, i6, i7],
                    'prices': [p0, p1, p2, p3, p4, p5, p6, p7]
                })
    return structures

def retrace_pct(p0, p1, p2):
    move = abs(p1 - p0)
    retrace = abs(p2 - p1)
    return retrace / move if move != 0 else 0

def advanced_price_action_validation(df, i, signal_type="high"):
    if i < 1 or i >= len(df):
        return False
    current = df.iloc[i]
    prev = df.iloc[i - 1]
    bullish_engulfing = prev['close'] < prev['open'] and current['close'] > current['open'] and current['close'] > prev['open'] and current['open'] < prev['close']
    bearish_engulfing = prev['close'] > prev['open'] and current['close'] < current['open'] and current['close'] < prev['open'] and current['open'] > prev['close']
    upper_shadow = current['high'] - max(current['close'], current['open'])
    lower_shadow = min(current['close'], current['open']) - current['low']
    body = abs(current['close'] - current['open'])
    bullish_pinbar = lower_shadow > body * 2 and current['close'] > current['open']
    bearish_pinbar = upper_shadow > body * 2 and current['close'] < current['open']
    wick_rejection = (upper_shadow > 1.5 * body and signal_type == "high") or (lower_shadow > 1.5 * body and signal_type == "low")
    ema_trend_up = df['adaptive_ema'].iloc[i] > df['adaptive_ema'].iloc[i - 1]
    ema_trend_down = df['adaptive_ema'].iloc[i] < df['adaptive_ema'].iloc[i - 1]
    hma_trend_up = df['adaptive_hma'].iloc[i] > df['adaptive_hma'].iloc[i - 1]
    hma_trend_down = df['adaptive_hma'].iloc[i] < df['adaptive_hma'].iloc[i - 1]
    confluence_up = ema_trend_up and hma_trend_up
    confluence_down = ema_trend_down and hma_trend_down
    if signal_type == "low":
        return bullish_engulfing or bullish_pinbar or wick_rejection or confluence_up
    else:
        return bearish_engulfing or bearish_pinbar or wick_rejection or confluence_down

def get_higher_tf_data(df, interval="15min"):
    df_resampled = df.resample(interval, on='timestamp').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_resampled.reset_index(inplace=True)
    return df_resampled

def detect_elliott_waves(df, df_15m=None):
    df = df.copy()
    price_array = df['close'].values.astype(np.float64)
    df['adaptive_ema'] = ultra_adaptive_ema(price_array, length=21, sensitivity=1.0)
    df['adaptive_hma'] = ultra_adaptive_hma(price_array, length=9, sensitivity=1.0)
    df['curvature'] = compute_curvature(price_array)
    df['pivot'] = realtime_peak_valley_v2(price_array, df['curvature'].values, threshold=0.0025, window=2)
    df['pivot_confirmed'] = None
    for i in range(len(df)):
        if df['pivot'].iloc[i] == 1 and advanced_price_action_validation(df, i, "high"):
            df.at[i, 'pivot_confirmed'] = 'high'
        elif df['pivot'].iloc[i] == -1 and advanced_price_action_validation(df, i, "low"):
            df.at[i, 'pivot_confirmed'] = 'low'
    pivot_points = df[df['pivot_confirmed'].notna()].reset_index()
    pivot_prices = pivot_points.apply(
        lambda row: df['high'][row['index']] if row['pivot_confirmed'] == 'high' else df['low'][row['index']],
        axis=1).tolist()
    pivot_indexes = pivot_points['index'].tolist()
    waves = []
    for i in range(len(pivot_prices) - 6):
        p0, p1, p2, p3, p4, p5 = pivot_prices[i:i+6]
        i0, i1, i2, i3, i4, i5 = pivot_indexes[i:i+6]
        r12 = retrace_pct(p0, p1, p2)
        r34 = retrace_pct(p2, p3, p4)
        if df_15m is not None:
            if not is_wave_valid_in_context(i0, i5, df_15m):
                continue
        if (0.3 < r12 < 0.7 and 0.3 < r34 < 0.7):
            wave = {
                'start_index': i0,
                'end_index': i5,
                'timestamp': df['timestamp'].iloc[i5],
                'wave': '1-5'
            }
            waves.append(wave)
    return pd.DataFrame(waves)

def is_wave_valid_in_context(wave_start_idx_5m, wave_end_idx_5m, df_15m, tolerance=0.382):
    time_start = df_5m.index[wave_start_idx_5m]
    time_end = df_5m.index[wave_end_idx_5m]
    df_context = df_15m[(df_15m.index >= time_start) & (df_15m.index <= time_end)]
    if len(df_context) < 3:
        return True
    trend_direction = df_context['close'].iloc[-1] - df_context['close'].iloc[0]
    trend_5m = df_5m['close'].iloc[wave_end_idx_5m] - df_5m['close'].iloc[wave_start_idx_5m]
    return (trend_direction * trend_5m) > 0

def map_htf_wave_to_ltf(df, htf_waves):
    htf_wave_at_time = []
    for ts in df['timestamp']:
        valid_waves = htf_waves[htf_waves['timestamp'] <= ts]
        if not valid_waves.empty:
            last_wave = valid_waves.iloc[-1]
            htf_wave_at_time.append(last_wave['wave'])
        else:
            htf_wave_at_time.append(None)
    df['htf_wave'] = htf_wave_at_time
    return df

def is_wave_confirmed(df, wave_column='wave', htf_column='htf_wave'):
    confirmed = []
    for i in range(len(df)):
        local_wave = df[wave_column].iloc[i] if wave_column in df.columns else None
        higher_wave = df[htf_column].iloc[i]
        if higher_wave and local_wave and local_wave == higher_wave:
            confirmed.append(True)
        else:
            confirmed.append(False)
    df['confirmed_wave'] = confirmed
    return df

def validate_nested_structure(df, wave_num):
    if wave_num == 3:
        subwaves = detect_elliott_waves(df)
        return len(subwaves) >= 5
    elif wave_num in (2, 4):
        subwaves = detect_elliott_waves(df)
        return len(subwaves) >= 3
    return False

class Wave:
    def __init__(self, label, start_index, end_index, wave_type, fib_ratio=None, time_ratio=None, subwaves=None):
        self.label = label
        self.start_index = start_index
        self.end_index = end_index
        self.wave_type = wave_type
        self.fib_ratio = fib_ratio
        self.time_ratio = time_ratio
        self.subwaves = subwaves if subwaves else []

def detect_wxy_wxyz(price_data, start, end):
    segment = price_data[start:end+1]
    if len(segment) < 9:
        return None
    pivots = detect_real_time_pivot(segment)[0]  
    piv_locs = np.where(pivots)[0]
    if len(piv_locs) < 6:
        return None
    w, x, y = piv_locs[0], piv_locs[1], piv_locs[2]
    if detect_correction_type(segment[w:x+1]) == "Zigzag" and \
       detect_correction_type(segment[x:y+1]) in ["Flat", "Zigzag"]:
        return Wave(
            label="WXY",
            start_index=start + w,
            end_index=start + y,
            wave_type="WXY",
            subwaves=[]
        )
    if len(piv_locs) >= 6:
        w, x, y, x2, z = piv_locs[:5]
        if detect_correction_type(segment[w:x+1]) == "Zigzag" and \
           detect_correction_type(segment[x:y+1]) == "Flat" and \
           detect_correction_type(segment[x2:z+1]) in ["Triangle", "Zigzag"]:
            return Wave(
                label="WXYXZ",
                start_index=start + w,
                end_index=start + z,
                wave_type="WXYXZ",
                subwaves=[]
            )
    return None

def detect_nested_elliott_waves(price_data, start, end, level=0):
    waves = []
    if end - start < 10:
        return waves
    for i in range(start, end - 4):
        for j in range(i + 2, end - 2):
            for k in range(j + 1, end):
                combo_wave = detect_wxy_wxyz(price_data, i, k)
                if combo_wave:
                    waves.append(combo_wave)
                    continue
                wave1 = price_data[i]
                wave3 = price_data[j]
                wave5 = price_data[k]
                fib1 = abs(wave3 - wave1) / abs(wave5 - wave3) if abs(wave5 - wave3) != 0 else 0
                if 0.5 < fib1 < 2.0:
                    time1 = j - i
                    time2 = k - j
                    time_ratio = time2 / time1 if time1 != 0 else 0
                    wave = Wave(
                        label=f"Wave_Level{level}_{len(waves)}",
                        start_index=i,
                        end_index=k,
                        wave_type="impulse",
                        fib_ratio=fib1,
                        time_ratio=time_ratio,
                        subwaves=detect_nested_elliott_waves(price_data, i, k, level + 1)
                    )
                    waves.append(wave)
    return waves

def wave_to_json(wave):
    return {
        "label": wave.label,
        "start": wave.start_index,
        "end": wave.end_index,
        "type": "DoubleThree" if wave.wave_type == "WXY" else "TripleThree" if wave.wave_type == "WXYXZ" else wave.wave_type,
        "fib_ratio": wave.fib_ratio,
        "time_ratio": wave.time_ratio,
        "subwaves": [wave_to_json(sw) for sw in wave.subwaves]
    }

def check_rsi_divergence(wave, rsi_data):
    start_rsi = rsi_data[wave.start_index]
    end_rsi = rsi_data[wave.end_index]
    if wave.wave_type == "impulse":
        return end_rsi < start_rsi
    else:
        return end_rsi > start_rsi

def check_adx_strength(wave, adx_data):
    adx_values = adx_data[wave.start_index:wave.end_index+1]
    avg_adx = np.mean(adx_values)
    return avg_adx > 25

def calculate_wave_complexity(wave_data):
    if len(wave_data) < 3:
        return 0
    price_range = np.max(wave_data) - np.max(wave_data)
    variation = sum(abs(wave_data[i] - wave_data[i-1]) for i in range(1, len(wave_data)))
    return variation / price_range

def detect_correction_type(wave_data):
    if len(wave_data) < 5:
        return "Unknown"
    peak1 = wave_data[0]
    trough = np.min(wave_data[1:-1])
    peak2 = wave_data[-1]
    if abs(peak2 - peak1) < 0.15 * abs(peak1 - trough):
        return "Flat"
    elif wave_data[1] < wave_data[2] > wave_data[3] < wave_data[4]:
        return "Triangle"
    else:
        return "Zigzag"

def check_alternation(wave2_data, wave4_data):
    type2 = detect_correction_type(wave2_data)
    type4 = detect_correction_type(wave4_data)
    time2 = len(wave2_data)
    time4 = len(wave4_data)
    complexity2 = calculate_wave_complexity(wave2_data)
    complexity4 = calculate_wave_complexity(wave4_data)
    alternation_passed = (
        type2 != type4 or
        abs(time4 - time2) > 3 or
        complexity4 > complexity2 * 1.5
    )
    return {
        "type2": type2,
        "type4": type4,
        "time2": time2,
        "time4": time4,
        "complexity2": round(complexity2, 2),
        "complexity4": round(complexity4, 2),
        "alternation_passed": alternation_passed
    }

def get_slope(wave):
    return (wave['end_price'] - wave['start_price']) / (wave['end_time'] - wave['start_time'])

def check_overlap(w1, w4):
    return w4['start_price'] <= w1['end_price']

def is_leading_diagonal(waves):
    w1, w2, w3, w4, w5 = waves
    return (
        w1['end_price'] < w3['end_price'] > w5['end_price'] and
        w2['end_price'] > w4['end_price'] and
        abs(w1['end_price'] - w4['end_price']) < 0.382 * abs(w1['end_price'] - w3['end_price']) and
        abs(w3['end_price'] - w5['end_price']) < 0.618 * abs(w1['end_price'] - w3['end_price']) and
        w1['end_time'] < w2['end_time'] < w3['end_time'] < w4['end_time'] < w5['end_time'] and
        check_overlap(w1, w4)
    )

def is_ending_diagonal(waves):
    w1, w2, w3, w4, w5 = waves
    return (
        w3['end_price'] > w1['end_price'] and
        w5['end_price'] < w3['end_price'] and
        w4['end_price'] > w2['end_price'] and
        abs(w1['end_price'] - w4['end_price']) < 0.382 * abs(w1['end_price'] - w3['end_price']) and
        abs(w3['end_price'] - w5['end_price']) < 0.618 * abs(w1['end_price'] - w3['end_price']) and
        w1['end_time'] < w2['end_time'] < w3['end_time'] < w4['end_time'] < w5['end_time'] and
        check_overlap(w1, w4)
    )

def is_bearish_divergence(prices, rsi_values):
    return prices[-1] > prices[-2] and rsi_values[-1] < rsi_values[-2]

def is_multi_tf_confirmed(major_wave, df_1m, df_5m):
    t_start = df_5m['timestamp'].iloc[major_wave['points'][0]]
    t_end = df_5m['timestamp'].iloc[major_wave['points'][-1]]
    df_sub = df_1m[(df_1m['timestamp'] >= t_start) & (df_1m['timestamp'] <= t_end)]
    if len(df_sub) < 10:
        return False
    nested_waves = detect_nested_elliott_waves(df_sub['close'].values, 0, len(df_sub)-1)
    return len(nested_waves) > 0

def get_wave_direction(waves):
    if len(waves) < 2:
        return None
    return "up" if waves[-1]["end_price"] > waves[-2]["end_price"] else "down"

def confirm_contextual_alignment(dir_1m, dir_5m, dir_15m):
    return dir_1m == dir_5m == dir_15m

def is_impulse_wave(waves):
    if len(waves) < 5:
        return False
    return (
        waves[2]["length"] > waves[0]["length"] and
        waves[3]["end_price"] > waves[1]["end_price"] and
        waves[4]["end_price"] > waves[2]["end_price"]
    )

def detect_combined_divergence(df, indexes, wave_type='peak'):
    divergences = []
    for i in range(1, len(indexes)):
        idx1 = indexes[i-1]
        idx2 = indexes[i]
        price1 = df['close'][idx1]
        price2 = df['close'][idx2]
        rsi1 = df['rsi'][idx1]
        rsi2 = df['rsi'][idx2]
        adx1 = df['adx'][idx1]
        adx2 = df['adx'][idx2]
        if wave_type == 'peak' and price2 > price1 and rsi2 < rsi1 and adx2 < adx1:
            divergences.append(idx2)
        if wave_type == 'valley' and price2 < price1 and rsi2 > rsi1 and adx2 < adx1:
            divergences.append(idx2)
    return divergences

class WavePatternType(Enum):
    IMPULSE = "Impulse"
    ZIGZAG = "Zigzag"
    FLAT = "Flat"
    TRIANGLE = "Triangle"
    DOUBLE_THREE = "Double Three"
    TRIPLE_THREE = "Triple Three"

class CompositeCorrection:
    def __init__(self, waves, type):
        self.waves = waves
        self.type = type

def detect_double_three(waves):
    if len(waves) < 5:
        return None
    for i in range(1, len(waves) - 3):
        w = waves[i-1]
        x = waves[i]
        y = waves[i+1]
        if w['wave'] == '1-5' and x['wave'] == '1-5' and y['wave'] == '1-5':  
            return CompositeCorrection([w, x, y], WavePatternType.DOUBLE_THREE)
    return None

def detect_triple_three(waves):
    if len(waves) < 7:
        return None
    for i in range(1, len(waves) - 5):
        w = waves[i-1]
        x = waves[i]
        y = waves[i+1]
        z1 = waves[i+2]
        z2 = waves[i+3]
        if all(w['wave'] == '1-5' for w in [w, x, y, z1, z2]):  
            return CompositeCorrection([w, x, y, z1, z2], WavePatternType.TRIPLE_THREE)
    return None

def detect_higher_tf_wave_structure(df_15m):
    pivot_indexes = df_15m[df_15m['pivot_confirmed'].notna()]['index'].tolist()
    pivot_prices = df_15m.apply(
        lambda row: row['high'] if row['pivot_confirmed'] == 'high' else row['low'], axis=1
    ).tolist()
    if len(pivot_indexes) < 6:
        return None
    waves = []
    for i in range(len(pivot_indexes) - 6):
        i0, i1, i2, i3, i4, i5 = pivot_indexes[i:i+6]
        p0, p1, p2, p3, p4, p5 = pivot_prices[i:i+6]
        r12 = retrace_pct(p0, p1, p2)
        r34 = retrace_pct(p2, p3, p4)
        if 0.3 < r12 < 0.7 and 0.3 < r34 < 0.7:
            wave_structure = {
                'wave': '1-5',
                'indexes': [i0, i1, i2, i3, i4, i5],
                'prices': [p0, p1, p2, p3, p4, p5],
                'direction': 'up' if p5 > p0 else 'down'
            }
            waves.append(wave_structure)
    return waves[-1] if waves else None

def infer_context_from_higher_tf(higher_tf_wave, current_index):
    i0, i1, i2, i3, i4, i5 = higher_tf_wave['indexes']
    if i2 <= current_index <= i3:
        return 'wave_3_high_tf'
    elif i4 <= current_index <= i5:
        return 'wave_5_high_tf'
    elif i1 <= current_index <= i2:
        return 'wave_2_correction'
    elif i3 <= current_index <= i4:
        return 'wave_4_correction'
    else:
        return 'unknown'

def fetch_ohlcv(symbol, timeframe, limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_wave_duration(wave):
    return wave["end_index"] - wave["start_index"]

def is_strong_wave3(df_5m, i2, i3):
    return True

def project_fibonacci_zone_advanced(p0, p1, p3):
    fib_ratios = [0.382, 0.5, 0.618, 1.0, 1.272, 1.618, 2.0, 2.618, 4.236]
    wave1_length = p1['price'] - p0['price']
    wave3_length = p3['price'] - p1['price']
    fib_projection = {
        'wave1_proj': {r: p1['price'] + wave1_length * r for r in fib_ratios},
        'wave3_ext': {r: p1['price'] + wave3_length * r for r in fib_ratios},
        'wave3_retrace': {r: p3['price'] - wave3_length * r for r in fib_ratios}
    }
    return fib_projection

def project_fibonacci_time(t0, t1, t2):
    time_ratios = [0.382, 0.5, 0.618, 1.0, 1.618, 2.0]
    duration_1 = t1 - t0
    duration_2 = t2 - t1
    fib_times = {r: int(t2 + r * duration_2) for r in time_ratios}
    return fib_times

def compute_PRZ_zone(price_projections, time_projections):
    prz = {
        'price_min': min([
            min(price_projections['wave1_proj'].values()),
            min(price_projections['wave3_ext'].values())
        ]),
        'price_max': max([
            max(price_projections['wave1_proj'].values()),
            max(price_projections['wave3_ext'].values())
        ]),
        'time_start': min(time_projections.values()),
        'time_end': max(time_projections.values())
    }
    return prz

def check_fib_reaction(df_5m, levels, i5):
    current_price = df_5m['close'].iloc[i5]
    tolerance = current_price * 0.005
    confirmed = [lvl for lvl in levels if abs(current_price - lvl) <= tolerance]
    return confirmed

def define_trade_zone(p4, fib_level):
    return {
        'entry_zone': (min(p4, fib_level), max(p4, fib_level)),
        'target': fib_level
    }

def define_prz(pivot_lows, pivot_highs, ema_series, price_series, wave_labels):
    prz_zones = []
    for i in range(len(price_series)):
        if wave_labels[i] == '4_end' and abs(price_series[i] - ema_series[i]) < 0.1:
            prz_zones.append((i, price_series[i]))
    return prz_zones

def calculate_entry_sl_tp(prz_point, price_series, rr_ratio=2.0):
    entry_price = price_series[prz_point]
    sl_price = entry_price - (entry_price * 0.005)
    tp_price = entry_price + (entry_price - sl_price) * rr_ratio
    return entry_price, sl_price, tp_price

def is_rsi_divergence(price_series, rsi_series, pivot_idx):
    if pivot_idx < 14:
        return False
    return (price_series[pivot_idx] > price_series[pivot_idx-14]) and (rsi_series[pivot_idx] < rsi_series[pivot_idx-14])

def is_diagonal_structure(waves):
    return False

def calculate_rsi(price_data):
    rsi_indicator = RSIIndicator(close=pd.Series(price_data), window=14)
    return rsi_indicator.rsi()

def detect_impulse_waves(df_5m, pivot_prices_5m, pivot_indexes_5m):
    waves = []
    for i in range(len(pivot_prices_5m) - 6):
        p0, p1, p2, p3, p4, p5 = pivot_prices_5m[i:i + 6]
        i0, i1, i2, i3, i4, i5 = pivot_indexes_5m[i:i + 6]
        wave1 = {'length': abs(p1 - p0), 'high': p1, 'start': i0, 'end': i1}
        wave2 = {'start': i1, 'end': i2}
        wave3 = {'length': abs(p3 - p2), 'start': i2, 'end': i3}
        wave4 = {'low': p4, 'start': i3, 'end': i4}
        wave5 = {'length': abs(p5 - p4), 'start': i4, 'end': i5}
        if wave3['length'] <= wave1['length'] or wave3['length'] <= wave5['length']:
            continue
        if is_diagonal_structure([wave1, wave2, wave3, wave4, wave5]):
            pass
        else:
            if wave4['low'] <= wave1['high']:
                continue
        rsi = calculate_rsi(df_5m['close'])
        if rsi[wave3['end']] < rsi[wave1['end']]:
            continue
        duration_3 = wave3['end'] - wave3['start']
        duration_1 = wave1['end'] - wave1['start']
        if duration_3 > duration_1 and wave3['length'] / duration_3 < wave1['length'] / duration_1:
            continue
        waves.append({
            'wave1': wave1,
            'wave2': wave2,
            'wave3': wave3,
            'wave4': wave4,
            'wave5': wave5
        })
    return waves

def calculate_indicators(df):
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    length_ema_fast = 9
    volatility = ta.atr(high, low, close, 14)
    ema_fast = ta.ema(close + volatility, length_ema_fast)
    hma = adaptive_hma(close, length=9)
    df['ema_fast'] = ema_fast
    df['hma'] = hma
    return df

def calculate_hma(close):
    hma = adaptive_hma(close, length=9)
    return hma

def generate_trade_signals(df):
    wave4_end_price = None
    pivots = detect_swing_pivots(df['close'].values)
    if len(pivots) >= 5:
        wave4_end_price = df['close'].iloc[pivots[-2]]
    signal = None
    threshold = df['close'].iloc[-1] * 0.005
    if wave4_end_price is not None:
        if abs(df['ema_fast'].iloc[-1] - wave4_end_price) < threshold and df['hma'].iloc[-1] < df['ema_fast'].iloc[-1]:
            signal = 'potential_reversal_buy'
    return signal, wave4_end_price

def export_signals(signal, filename="signals.json"):
    with open(filename, "w") as f:
        json.dump(signal, f, indent=4)

def realtime_wave_detection(symbol='BTC/USDT'):
    def compute_fractal_energy(price_series, window=14):
        high = pd.Series(price_series).rolling(window=window).max()
        low = pd.Series(price_series).rolling(window=window).min()
        range_ = high - low
        energy = range_ / (pd.Series(price_series).rolling(window=window).std() + 1e-8)
        return energy.fillna(0).values

    while True:
        df_1m = fetch_ohlcv(symbol, '1m', limit=1000)
        df_5m = fetch_ohlcv(symbol, '5m', limit=1000)
        df_15m = fetch_ohlcv(symbol, '15m', limit=500)
        if df_1m.empty or df_5m.empty or df_15m.empty:
            print("Error fetching data, retrying...")
            time.sleep(15)
            continue
        price_array_1m = df_1m['close'].values.astype(np.float64)
        price_array_5m = df_5m['close'].values.astype(np.float64)
        price_array_15m = df_15m['close'].values.astype(np.float64)
        df_5m = calculate_indicators(df_5m)
        waves_1m = detect_elliott_waves(df_1m)
        waves_5m = detect_elliott_waves(df_5m, df_15m)
        waves_15m = detect_elliott_waves(df_15m)
        dir_1m = get_wave_direction(waves_1m)
        dir_5m = get_wave_direction(waves_5m)
        dir_15m = get_wave_direction(waves_15m)
        df_15m_from_5m = get_higher_tf_data(df_5m, interval="15min")
        htf_waves = detect_elliott_waves(df_15m_from_5m)
        df_5m = map_htf_wave_to_ltf(df_5m, htf_waves)
        df_5m = is_wave_confirmed(df_5m)
        curvature = compute_curvature(price_array_5m, window=5)
        abs_curv = np.abs(curvature)
        if len(price_array_5m) > 4:
            valid_abs_curv = abs_curv[2:-2]
            threshold_value = np.percentile(valid_abs_curv, 75) if len(valid_abs_curv) > 0 else 0.001
        else:
            threshold_value = 0.001
        df_5m['pivot_confirmed'] = None
        fractal_energy = compute_fractal_energy(price_array_5m)
        for i in range(2, len(df_5m)-2):
            adaptive_thresh = threshold_value * (1 + df_5m['adx'].iloc[i] / 100)
            if is_real_time_fractal_peak(i, curvature, adaptive_thresh) and fractal_energy[i] > 1.0:
                df_5m.at[i, 'pivot_confirmed'] = 'high'
            elif is_real_time_fractal_valley(i, curvature, adaptive_thresh) and fractal_energy[i] > 1.0:
                df_5m.at[i, 'pivot_confirmed'] = 'low'
        df_5m['rsi'] = RSIIndicator(close=df_5m['close'], window=14).rsi()
        adx_i = ADXIndicator(df_5m['high'], df_5m['low'], df_5m['close'], window=14)
        df_5m['adx'] = adx_i.adx()
        df_5m['plus_di'] = adx_i.plus_di()
        df_5m['minus_di'] = adx_i.minus_di()
        pivot_highs, pivot_lows = detect_real_time_pivot(price_array_5m)
        wave_labels = ['none'] * len(df_5m)
        if pivot_lows[-1]:
            wave_labels[-1] = '4_end'
        prz_zone = define_prz(pivot_lows, pivot_highs, df_5m['ema_fast'], df_5m['close'], wave_labels)
        signal, wave4_end_price = generate_trade_signals(df_5m)
        atr_value = ta.atr(df_5m['high'], df_5m['low'], df_5m['close'], 14).iloc[-1]
        current_price = df_5m['close'].iloc[-1]
        if prz_zone and signal == 'potential_reversal_buy':
            entry, sl, tp = calculate_entry_sl_tp(prz_zone[-1][0], df_5m['close'])
            rr = (tp - entry) / (entry - sl)
            if rr >= 1.5:
                signal_json = {
                    "entry": round(entry, 3),
                    "sl": round(sl, 3),
                    "tp": round(tp, 3),
                    "rr": round(rr, 2),
                    "type": "BUY",
                    "timestamp": str(df_5m['timestamp'].iloc[-1])
                }
                with open("trade_signal.json", "w") as outfile:
                    json.dump(signal_json, outfile)
        pivot_points_5m = curvature_based_peaks(df_5m)
        valley_indexes = [row['index'] for _, row in pivot_points_5m.iterrows() if row['pivot_confirmed'] == 'low']
        peak_indexes = [row['index'] for _, row in pivot_points_5m.iterrows() if row['pivot_confirmed'] == 'high']
        wave3_divs = detect_combined_divergence(df_5m, valley_indexes, wave_type='valley')
        wave5_divs = detect_combined_divergence(df_5m, peak_indexes, wave_type='peak')
        confirmed_wave3 = [idx for idx in valley_indexes if idx in wave3_divs]
        confirmed_wave5 = [idx for idx in peak_indexes if idx in wave5_divs]
        waves = []
        confirmed_trades = []
        pivot_prices_5m = pivot_points_5m.apply(
            lambda row: df_5m['high'][row['index']] if row['pivot_confirmed'] == 'high' else df_5m['low'][row['index']],
            axis=1).tolist()
        pivot_indexes_5m = pivot_points_5m['index'].tolist()
        impulse_waves = detect_impulse_waves(df_5m, pivot_prices_5m, pivot_indexes_5m)
        for i in range(len(pivot_prices_5m) - 6):
            p0 = {'price': pivot_prices_5m[i], 'time': pivot_indexes_5m[i]}
            p1 = {'price': pivot_prices_5m[i + 1], 'time': pivot_indexes_5m[i + 1]}
            p2 = pivot_prices_5m[i + 2]
            p3 = {'price': pivot_prices_5m[i + 3], 'time': pivot_indexes_5m[i + 3]}
            p4 = pivot_prices_5m[i + 4]
            p5 = pivot_prices_5m[i + 5]
            i0, i1, i2, i3, i4, i5 = pivot_indexes_5m[i:i + 6]
            r12 = retrace_pct(p0['price'], p1['price'], p2)
            r34 = retrace_pct(p2, p3['price'], p4)
            wave2_data = df_5m['close'].iloc[i1:i2+1].values
            wave4_data = df_5m['close'].iloc[i3:i4+1].values
            alternation_result = check_alternation(wave2_data, wave4_data)
            alternation_passed = alternation_result["alternation_passed"]
            wave1 = {"start_index": i0, "end_index": i1}
            wave3 = {"start_index": i2, "end_index": i3}
            wave5 = {"start_index": i4, "end_index": i5}
            time_wave1 = calculate_wave_duration(wave1)
            time_wave3 = calculate_wave_duration(wave3)
            time_wave5 = calculate_wave_duration(wave5)
            total_impulse_time = i5 - i0
            time_ratios = {}
            if time_wave1 and time_wave3:
                time_ratios["wave3_vs_wave1"] = time_wave3 / time_wave1
            if time_wave1 and time_wave5:
                time_ratios["wave5_vs_total"] = time_wave5 / total_impulse_time if total_impulse_time else None
            valid_time = True
            if time_ratios.get("wave3_vs_wave1") is not None:
                ratio3 = time_ratios["wave3_vs_wave1"]
                if not (0.618 <= ratio3 <= 2.0):
                    valid_time = False
            if time_ratios.get("wave5_vs_total") is not None:
                ratio5 = time_ratios["wave5_vs_total"]
                if not (0.3 <= ratio5 <= 0.9):
                    valid_time = False
            wave1_diag = {'start_price': p0['price'], 'end_price': p1['price'], 'start_time': i0, 'end_time': i1}
            wave2_diag = {'start_price': p1['price'], 'end_price': p2, 'start_time': i1, 'end_time': i2}
            wave3_diag = {'start_price': p2, 'end_price': p3['price'], 'start_time': i2, 'end_time': i3}
            wave4_diag = {'start_price': p3['price'], 'end_price': p4, 'start_time': i3, 'end_time': i4}
            wave5_diag = {'start_price': p4, 'end_price': p5, 'start_time': i4, 'end_time': i5}
            waves_for_diagonal = [wave1_diag, wave2_diag, wave3_diag, wave4_diag, wave5_diag]
            if is_leading_diagonal(waves_for_diagonal):
                diagonal_type = "Leading Diagonal"
            elif is_ending_diagonal(waves_for_diagonal):
                diagonal_type = "Ending Diagonal"
            else:
                diagonal_type = None
            major_wave = {
                'wave': '1-5',
                'points': [i0, i1, i2, i3, i4, i5],
                'prices': [p0['price'], p1['price'], p2, p3['price'], p4, p5],
                'retraces': {'1-2': round(r12, 2), '3-4': round(r34, 2)},
                'alternation': alternation_passed,
                'alternation_details': alternation_result,
                'time_ratios': time_ratios,
                'durations': {
                    'wave1': time_wave1,
                    'wave3': time_wave3,
                    'wave5': time_wave5,
                    'total_impulse': total_impulse_time
                },
                'valid_time_structure': valid_time,
                'diagonal': diagonal_type
            }
            if diagonal_type is not None:
                wave5_prices = df_5m['close'].iloc[wave5_diag['start_time']:wave5_diag['end_time']+1].values
                wave5_rsi = df_5m['rsi'].iloc[wave5_diag['start_time']:wave5_diag['end_time']+1].values
                divergence = False
                if len(wave5_prices) >= 2 and is_bearish_divergence(wave5_prices, wave5_rsi):
                    divergence = True
                major_wave['diagonal_signal'] = {
                    'type': diagonal_type,
                    'entry_price': wave5_diag['end_price'],
                    'time': wave5_diag['end_time'],
                    'sl': wave4_diag['end_price'],
                    'tp': wave5_diag['end_price'] + (wave5_diag['end_price'] - wave3_diag['end_price']) * 1.618,
                    'divergence': divergence
                }
                if i5 == len(df_5m) - 1:
                    print(f"🔺 Detected {diagonal_type} at candle {i5} — Entry Zone Possible")
            wave1_dict = {'start': i0, 'end': i1}
            wave3_dict = {'start': i2, 'end': i3}
            wave3_strength_valid = validate_wave3_strength(df_5m, wave1_dict, wave3_dict)
            contextual_alignment = confirm_contextual_alignment(dir_1m, dir_5m, dir_15m)
            impulse_valid = is_impulse_wave(waves_15m)
            higher_tf_wave = detect_higher_tf_wave_structure(df_15m)
            if higher_tf_wave:
                context = infer_context_from_higher_tf(higher_tf_wave, i5)
                major_wave['context_15m'] = context
                if context == 'wave_3_high_tf' or context == 'wave_5_high_tf':
                    contextually_valid = True
                else:
                    contextually_valid = False
            else:
                contextually_valid = False
                major_wave['context_15m'] = None
            major_wave['contextual_validation'] = contextually_valid
            major_wave['higher_tf_context'] = context if higher_tf_wave else None
            pivots = detect_swing_pivots(price_array_5m, 3, 3)
            w1_dict = {'length': abs(p1['price'] - p0['price']), 'high': p1['price'], 'subwaves': 5}
            w2_dict = {'subwaves': 3}
            w3_dict = {'length': abs(p3['price'] - p2), 'subwaves': 5, 'end': i3}
            w4_dict = {'low': p4, 'subwaves': 3}
            w5_dict = {'length': abs(p5 - p4), 'subwaves': 5, 'end': i5}
            rsi_list = df_5m['rsi'].tolist()
            adx_list = df_5m['adx'].tolist()
            is_entire_impulse_not_valid = is_not_impulsive_advanced(
                w1_dict, w2_dict, w3_dict, w4_dict, w5_dict, rsi_list, adx_list)
            if (0.3 < r12 < 0.7 and
                0.3 < r34 < 0.7 and
                alternation_passed and
                wave3_strength_valid and
                i2 in confirmed_wave3 and
                i5 in confirmed_wave5 and
                contextual_alignment and
                impulse_valid and
                contextually_valid and
                not is_entire_impulse_not_valid):
                multi_tf_confirmed = is_multi_tf_confirmed(major_wave, df_1m, df_5m)
                major_wave['multi_tf_confirmed'] = multi_tf_confirmed
                mtf_confirmed = df_5m['confirmed_wave'].iloc[i5]
                nested_valid = validate_nested_structure(df_5m.iloc[i0:i5], 3)
                if multi_tf_confirmed and mtf_confirmed and nested_valid:
                    fib_projections = project_fibonacci_zone_advanced(p0, p1, p3)
                    fib_times = project_fibonacci_time(p0['time'], p1['time'], p3['time'])
                    prz_zone = compute_PRZ_zone(fib_projections, fib_times)
                    major_wave['fib_times'] = fib_times
                    major_wave['prz_zone'] = prz_zone
                    all_fib_levels = list(fib_projections['wave1_proj'].values()) + list(fib_projections['wave3_ext'].values())
                    confirmed_levels = check_fib_reaction(df_5m, all_fib_levels, i5)
                    div_wave3 = i2 in wave3_divs
                    div_wave5 = i5 in wave5_divs
                    if confirmed_levels:
                        trade_zone = define_trade_zone(p4, confirmed_levels[0])
                    else:
                        trade_zone = None
                    major_wave['fib_levels'] = {
                        'projected': [round(lvl, 2) for lvl in fib_projections['wave1_proj'].values()],
                        'extended': [round(lvl, 2) for lvl in fib_projections['wave3_ext'].values()],
                        'confirmed': confirmed_levels
                    }
                    major_wave['divergence'] = {'wave3': div_wave3, 'wave5': div_wave5}
                    major_wave['trade_zone'] = trade_zone
                    segment_data = price_array_5m[i0:i5+1]
                    major_wave['nested_waves'] = detect_nested_waves(segment_data)
                    if i5 == len(df_5m) - 1:
                        print(f"🚀 Real-Time Signal: Entry at {df_5m['close'].iloc[i5]}")
                    wave1_len = abs(p1['price'] - p0['price'])
                    wave3_len = abs(p3['price'] - p2)
                    wave4_end_price = p4
                    wave5_target_0618 = p3['price'] + wave1_len * 0.618
                    wave5_target_1 = p3['price'] + wave1_len * 1.0
                    wave5_target_1618 = p3['price'] + wave1_len * 1.618
                    entry_zone_bottom = wave4_end_price
                    entry_zone_top = wave5_target_0618
                    expected_reversal_price = fibonacci_extension(
                        low_point=p0['price'], 
                        high_point=p3['price'], 
                        retrace_point=p4, 
                        ratio=1.618
                    )
                    ema_val = df_5m['ema_fast'].values
                    hma_val = df_5m['hma'].values
                    hma_cross = (hma_val[-2] < ema_val[-2] and hma_val[-1] > ema_val[-1]) or \
                                (hma_val[-2] > ema_val[-2] and hma_val[-1] < ema_val[-1])
                    current_candle = df_5m.iloc[-1]
                    prev_candle = df_5m.iloc[-2]
                    c1 = df_5m.iloc[-3] if len(df_5m) >= 3 else df_5m.iloc[-1]
                    c2 = df_5m.iloc[-2]
                    c3 = df_5m.iloc[-1]
                    bullish_pattern = (
                        is_bullish_engulfing(c3, c2) or
                        is_pin_bar(c3) or
                        is_doji(c3) or
                        is_hammer(c3) or
                        is_tweezer_bottom(c3, c2) or
                        is_inside_bar(c3, c2) or
                        is_morning_star(c1, c2, c3)
                    )
                    bearish_pattern = (
                        is_bearish_engulfing(c3, c2) or
                        is_pin_bar(c3) or
                        is_doji(c3) or
                        is_tweezer_top(c3, c2) or
                        is_inside_bar(c3, c2)
                    )
                    tolerance = expected_reversal_price * 0.005
                    current_price = df_5m['close'].iloc[-1]
                    if (current_price >= expected_reversal_price - tolerance and
                        current_price <= expected_reversal_price + tolerance and
                        hma_cross and 
                        bullish_pattern):
                        entry_price = current_price
                        stop_loss = wave4_end_price - (wave1_len * 0.382)
                        take_profit = entry_price + (entry_price - stop_loss) * 1.618
                        signal = {
                            "timestamp": current_candle["timestamp"].to_pydatetime().timestamp(),
                            "price": current_candle["close"],
                            "entry": current_candle["close"],
                            "sl": stop_loss,
                            "tp": take_profit,
                            "rr": round((take_profit - current_candle["close"]) / (current_candle["close"] - stop_loss), 2),
                            "wave_id": "wave5",
                            "wave_type": "Impulse",
                            "confirmed_by": {
                                "candlestick": "Engulfing",
                                "divergence": "RSI Bullish" if i4 in wave3_divs else "None",
                                "multi_tf": multi_tf_confirmed
                            },
                            "timeframe": "5m"
                        }
                        export_signals(signal)
                        print(f"✅ سیگنال خرید در پایان موج ۵ صادر شد! Entry: {entry_price:.2f}, TP: {take_profit:.2f}, SL: {stop_loss:.2f}, R:R = 1.618")
                        major_wave['wave5_entry'] = signal
                    elif (current_price >= expected_reversal_price - tolerance and
                          current_price <= expected_reversal_price + tolerance and
                          hma_cross and 
                          bearish_pattern):
                        entry_price = current_price
                        stop_loss = wave4_end_price + (wave1_len * 0.382)
                        take_profit = entry_price - (stop_loss - entry_price) * 1.618
                        signal = {
                            "timestamp": current_candle["timestamp"].to_pydatetime().timestamp(),
                            "price": current_candle["close"],
                            "entry": current_candle["close"],
                            "sl": stop_loss,
                            "tp": take_profit,
                            "rr": round((current_candle["close"] - take_profit) / (stop_loss - current_candle["close"]), 2),
                            "wave_id": "wave5",
                            "wave_type": "Impulse",
                            "confirmed_by": {
                                "candlestick": "Engulfing",
                                "divergence": "RSI Bearish" if i5 in wave5_divs else "None",
                                "multi_tf": multi_tf_confirmed
                            },
                            "timeframe": "5m"
                        }
                        export_signals(signal)
                        print(f"✅ سیگنال فروش در پایان موج ۵ صادر شد! Entry: {entry_price:.2f}, TP: {take_profit:.2f}, SL: {stop_loss:.2f}, R:R = 1.618")
                        major_wave['wave5_entry'] = signal
                    else:
                        major_wave['wave5_entry'] = None
                    major_wave['wave5_forecast'] = {
                        'target_0618': wave5_target_0618,
                        'target_1': wave5_target_1,
                        'target_1618': wave5_target_1618,
                        'entry_zone': (entry_zone_bottom, entry_zone_top),
                        'expected_reversal_price': expected_reversal_price,
                        'tolerance': tolerance,
                        'hma_cross': hma_cross,
                        'bullish_pattern': bullish_pattern,
                        'bearish_pattern': bearish_pattern
                    }
                    wave4_end_index = i4
                    wave4_end_price = p4
                    early_entry_ready = True
                    wave1_len = abs(p1['price'] - p0['price'])
                    is_uptrend = p1['price'] > p0['price']
                    if is_uptrend:
                        wave5_proj_price = wave4_end_price + wave1_len * 0.618
                        stop_loss = min(p2, p4)
                    else:
                        wave5_proj_price = wave4_end_price - wave1_len * 0.618
                        stop_loss = max(p2, p4)
                    entry_signal = {
                        'entry_price': wave4_end_price,
                        'tp': wave5_proj_price,
                        'sl': stop_loss,
                        'rr_ratio': abs(wave5_proj_price - wave4_end_price) / abs(wave4_end_price - stop_loss),
                        'wave': 'early_wave5'
                    }
                    major_wave['early_entry'] = entry_signal
                    idx = i4
                    is_engulfing_val = is_bullish_engulfing(df_5m.iloc[idx], df_5m.iloc[idx-1]) if idx >= 1 else False
                    is_pinbar_val = is_pin_bar(df_5m.iloc[idx])
                    is_price_action = is_engulfing_val or is_pinbar_val
                    hma_val = df_5m['hma'].iloc[idx]
                    ema_val = df_5m['ema_fast'].iloc[idx]
                    if idx >= 1:
                        hma_cross_up = (hma_val > ema_val) and (df_5m['hma'].iloc[idx-1] <= df_5m['ema_fast'].iloc[idx-1])
                    else:
                        hma_cross_up = False
                    wave3_len = abs(p3['price'] - p2)
                    wave1_len = abs(p1['price'] - p0['price'])
                    wave3_momentum = df_5m['adx'].iloc[i3]
                    wave1_momentum = df_5m['adx'].iloc[i1]
                    strong_wave3 = abs(wave3_len) > abs(wave1_len) and wave3_momentum > wave1_momentum
                    entry_signal = is_price_action and hma_cross_up and strong_wave3
                    if entry_signal:
                        entry_price = df_5m['close'].iloc[idx]
                        stop_loss = p4
                        take_profit = entry_price + 1.618 * (p3['price'] - p0['price'])
                        trade_signal = {
                            "entry_idx": int(idx),
                            "entry_price": float(entry_price),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "type": "buy"
                        }
                        confirmed_trades.append(trade_signal)
                    if wave3_strength_valid:
                        if p5 > p0:
                            wave3_entry_price = p2
                            wave3_sl = min(p1['price'], p2) - abs(p3['price'] - p2) * 0.382
                            wave3_tp = p2 + abs(p3['price'] - p2) * 1.618
                        else:
                            wave3_entry_price = p2
                            wave3_sl = max(p1['price'], p2) + abs(p3['price'] - p2) * 0.382
                            wave3_tp = p2 - abs(p3['price'] - p2) * 1.618
                        major_wave['wave3_rr'] = generate_dynamic_rr(wave3_entry_price, wave3_sl, wave3_tp)
                    if contextually_valid and confirmed_levels:
                        if p5 > p0:
                            wave5_entry_price = current_price
                            wave5_sl = wave4_end_price - (wave1_len * 0.382)
                            wave5_tp = wave5_entry_price + (wave5_entry_price - wave5_sl) * 1.618
                        else:
                            wave5_entry_price = current_price
                            wave5_sl = wave4_end_price + (wave1_len * 0.382)
                            wave5_tp = wave5_entry_price - (wave5_sl - wave5_entry_price) * 1.618
                        major_wave['wave5_rr'] = generate_dynamic_rr(wave5_entry_price, wave5_sl, wave5_tp)
                    major_wave['rr_structures'] = {
                        'wave1': major_wave.get('wave1_rr', None),
                        'wave3': major_wave.get('wave3_rr', None),
                        'wave5': major_wave.get('wave5_rr', None)
                    }
                    waves.append(major_wave)
        wxy_structures = detect_wxy_wxyz_structure(pivot_prices_5m, pivot_indexes_5m)
        for struct in wxy_structures:
            if struct['type'] == 'WXY':
                print(f"🔁 ساختار اصلاحی WXY شناسایی شد در نقاط {struct['points']}")
            elif struct['type'] == 'WXYXZ':
                if not is_wave_valid_in_context(struct['points'][0], struct['points'][-1], df_15m):
                    continue
                print(f"🔁🔁 ساختار اصلاحی WXYXZ شناسایی شد در نقاط {struct['points']}")
                z_end_index = struct['points'][-1]
                price_z = struct['prices'][-1]
                if len(df_5m) - 1 == z_end_index:
                    entry_price = df_5m['close'].iloc[z_end_index]
                    stop = entry_price - (entry_price * 0.005)
                    tp = entry_price + (entry_price - stop) * 1.618
                    signal = {
                        "entry": entry_price,
                        "sl": stop,
                        "tp": tp,
                        "type": "BUY",
                        "structure": "WXYXZ"
                    }
                    confirmed_trades.append(signal)
        output = {
            'waves': [
                {
                    **wave,
                    'nested_structure': wave.get('nested_waves', []),
                    'wave5_forecast': wave.get('wave5_forecast', None),
                    'wave5_entry': wave.get('wave5_entry', None),
                    'early_entry': wave.get('early_entry', None),
                    'contextual_validation': wave.get('contextual_validation', None),
                    'higher_tf_context': wave.get('higher_tf_context', None),
                    'rr_structures': wave.get('rr_structures', None)
                } 
                for wave in waves
            ],
            'confirmed_trades': confirmed_trades,
            'peaks': peak_indexes,
            'valleys': valley_indexes,
            'complex_corrections': wxy_structures
        }
        with open('wave_professional_price_action_multitimeframe.json', 'w') as f:
            f.write(json.dumps(output, indent=2))
        print(json.dumps(output, indent=2))
        time.sleep(15)

if __name__ == "__main__":
    realtime_wave_detection()

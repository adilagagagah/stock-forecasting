import pandas as pd
import numpy as np
from scipy.stats import linregress

def direction_label(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """ Menghitung slope tren pergerakan harga Close dari T+0 hingga T+4 """
    df = df.copy()
    x = np.arange(1, window + 1)
    open_t = df['Open']
    
    def get_slope(y):
        if len(y) < window or np.isnan(y).any():
            return np.nan
        slope, _, _, _, _ = linregress(x, y)
        return slope

    # .shift(1-window) menarik jendela rolling ke depan dimulai dari hari T+0
    norm_close = df['Close'].copy()
    df['trend_slope'] = norm_close.rolling(window=window).apply(get_slope, raw=True).shift(1-window)
    # Bagi dengan open_t untuk mendapatkan skala persentase pertumbuhan per hari
    df['trend_slope'] = df['trend_slope'] / open_t

    return df

def return_label(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """ Menghitung potensi return maksimum dari High (T+0 s.d T+4) dibanding Open T+0 """
    df = df.copy()
    open_t = df['Open'] # Harga entry di pagi hari T+0
    
    future_high_max = df['High'].rolling(window=window).max().shift(1-window)
    df['return'] = (future_high_max - open_t) / open_t

    return df

def risk_label(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """ Menghitung potensi risiko maksimum dari Low (T+0 s.d T+4) dibanding Open T+0 """
    df = df.copy()
    open_t = df['Open'] # Harga entry di pagi hari T+0
    
    future_low_min = df['Low'].rolling(window=window).min().shift(1-window)
    df['risk'] = (future_low_min - open_t) / open_t
    
    return df

def additional_information(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """ Menambahkan informasi bantuan untuk memastikan keabsahan hitungan """
    df = df.copy()
    open_t = df['Open']
    future_high_max = df['High'].rolling(window=window).max().shift(1-window)
    future_low_min = df['Low'].rolling(window=window).min().shift(1-window)
    
    def get_argmax(y):
        if len(y) < window or np.isnan(y).any():
            return np.nan
        return np.argmax(y) + 1

    def get_argmin(y):
        if len(y) < window or np.isnan(y).any():
            return np.nan
        return np.argmin(y) + 1
        
    df['days_to_max'] = df['High'].rolling(window=window).apply(get_argmax, raw=True).shift(1-window)
    df['days_to_min'] = df['Low'].rolling(window=window).apply(get_argmin, raw=True).shift(1-window)
    
    # Kolom tracking data mentah untuk keperluan validasi analisis pasar
    df['entry'] = open_t
    df['max'] = future_high_max
    df['min'] = future_low_min
    
    return df

def buy_the_dip_label(df: pd.DataFrame, window: int = 5, min_return: float = 0.03, max_drawdown: float = -0.05) -> pd.DataFrame:
    """
    Membuat label klasifikasi biner untuk mendeteksi momen 'Buy the Dip'.
    Label = 1 jika pada T+0 s.d T+4:
      - Harga tertinggi (High) naik >= min_return dari Open T+0 (ada potensi cuan)
      - Harga terendah (Low) tidak turun lebih dalam dari max_drawdown dari Open T+0 (risiko terkendali)
    
    Parameters:
    - min_return (float): Minimum return yang diharapkan (default: 0.03 = 3%)
    - max_drawdown (float): Batas drawdown maksimum yang ditoleransi (default: -0.05 = -5%)
    """
    df = df.copy()
    open_t = df['Open']
    
    future_high_max = df['High'].rolling(window=window).max().shift(1 - window)
    future_low_min = df['Low'].rolling(window=window).min().shift(1 - window)
    
    future_return = (future_high_max - open_t) / open_t
    future_risk = (future_low_min - open_t) / open_t
    
    df['is_dip'] = ((future_return >= min_return) & (future_risk >= max_drawdown)).astype(int)
    
    return df


def tp_before_sl_label(df: pd.DataFrame, window: int = 5, tp_pct: float = 0.08, sl_pct: float = -0.05) -> pd.DataFrame:
    """
    Label biner: apakah harga menyentuh Take Profit SEBELUM Stop Loss dalam horizon T+0 s/d T+(window-1).
    
    Logika per hari T:
    - TP level = Open_T * (1 + tp_pct)
    - SL level = Open_T * (1 + sl_pct)   (sl_pct negatif, misal -0.05)
    - Iterasi hari T+0 s/d T+(window-1):
        - Jika High >= TP level → label = 1 (TP tercapai duluan)
        - Jika Low <= SL level → label = 0 (SL tercapai duluan)
        - Jika keduanya tersentuh pada hari yang sama → asumsi konservatif SL duluan (label = 0)
    - Jika tidak ada yang tercapai → label = 0 (timeout, tidak ada edge)
    
    Parameters:
    - tp_pct (float): Target profit percentage (default: 0.08 = +8%)
    - sl_pct (float): Stop loss percentage, negatif (default: -0.05 = -5%)
    """
    df = df.copy()
    open_vals = df['Open'].values
    high_vals = df['High'].values
    low_vals = df['Low'].values
    n = len(df)
    
    labels = np.full(n, np.nan)
    
    for i in range(n):
        if np.isnan(open_vals[i]) or open_vals[i] <= 0:
            continue
        
        tp_level = open_vals[i] * (1 + tp_pct)
        sl_level = open_vals[i] * (1 + sl_pct)
        
        # Cek apakah horizon cukup
        end_idx = i + window
        if end_idx > n:
            continue
        
        result = 0  # Default: timeout / SL first
        for j in range(i, end_idx):
            if np.isnan(high_vals[j]) or np.isnan(low_vals[j]):
                continue
            
            hit_sl = low_vals[j] <= sl_level
            hit_tp = high_vals[j] >= tp_level
            
            if hit_sl and hit_tp:
                # Same candle: asumsi konservatif SL duluan
                result = 0
                break
            elif hit_sl:
                result = 0
                break
            elif hit_tp:
                result = 1
                break
        
        labels[i] = result
    
    df['tp_before_sl'] = labels
    return df


def swing_reversal_label(df: pd.DataFrame, window: int = 5, min_bounce: float = 0.05) -> pd.DataFrame:
    """
    Label biner untuk mendeteksi hari swing reversal: harga berada di bawah MA20
    (kondisi dip/oversold) DAN dalam horizon T+0 s/d T+(window-1) harga naik minimal min_bounce dari Open.
    
    Kriteria label = 1:
    - Close kemarin (T-1) di bawah MA20 (T-1) → sinyal harga sedang tertekan
    - High tertinggi dalam T+0 s/d T+(window-1) naik >= min_bounce dari Open T+0
    
    Ini lebih permisif daripada is_dip karena tidak mensyaratkan drawdown terkendali,
    cocok untuk menangkap bounce cepat di bear market.
    
    Parameters:
    - min_bounce (float): Minimum bounce percentage (default: 0.05 = 5%)
    """
    df = df.copy()
    open_t = df['Open']
    
    # MA20 dari Close, di-shift 1 agar menggunakan data sampai kemarin (T-1)
    ma20 = df['Close'].rolling(window=20).mean()
    close_below_ma20 = (df['Close'].shift(1) < ma20.shift(1))
    
    # MFE dalam horizon
    future_high_max = df['High'].rolling(window=window).max().shift(1 - window)
    future_return = (future_high_max - open_t) / open_t
    
    df['is_swing_reversal'] = (
        close_below_ma20 & 
        (future_return >= min_bounce)
    ).astype(int)
    
    # NaN-kan baris warm-up
    df.loc[df.index[:20], 'is_swing_reversal'] = np.nan
    
    return df


def create_labels(df: pd.DataFrame, window: int = 5, min_ret: float = 0.15, max_dd: float = -0.05) -> pd.DataFrame:
    """ Menggabungkan seluruh fungsi labeling masa depan pada baris indeks T+0 """
    df = direction_label(df, window)
    df = return_label(df, window)
    df = risk_label(df, window)
    df = additional_information(df, window)
    df = buy_the_dip_label(df, window, min_return=min_ret, max_drawdown=max_dd)
    df = tp_before_sl_label(df, window, tp_pct=0.08, sl_pct=-0.05)
    df = swing_reversal_label(df, window, min_bounce=0.05)
    return df
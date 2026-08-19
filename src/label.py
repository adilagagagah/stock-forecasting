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
    
    df['is_buy_dip'] = ((future_return >= min_return) & (future_risk >= max_drawdown)).astype(int)
    
    return df


def create_labels(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """ Menggabungkan seluruh fungsi labeling masa depan pada baris indeks T+0 """
    df = direction_label(df, window)
    df = return_label(df, window)
    df = risk_label(df, window)
    df = additional_information(df, window)
    df = buy_the_dip_label(df, window, min_return=0.15, max_drawdown=-0.05)
    return df
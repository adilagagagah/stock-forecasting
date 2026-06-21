import pandas as pd
import numpy as np
from scipy.stats import linregress

def direction_label(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """ Menghitung slope tren pergerakan harga Close dari T+0 hingga T+4 """
    df = df.copy()
    x = np.arange(1, window + 1)
    
    def get_slope(y):
        if len(y) < window or np.isnan(y).any():
            return np.nan
        slope, _, _, _, _ = linregress(x, y)
        return slope

    # .shift(-(window - 1)) menarik jendela rolling ke depan dimulai dari hari T+0
    df['trend_slope'] = df['Close'].rolling(window=window).apply(get_slope, raw=True).shift(1-window)
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

def create_labels(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """ Menggabungkan seluruh fungsi labeling masa depan pada baris indeks T+0 """
    df = direction_label(df, window)
    df = return_label(df, window)
    df = risk_label(df, window)
    df = additional_information(df, window)
    return df
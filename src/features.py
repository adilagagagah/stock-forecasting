import pandas as pd

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Menghitung Average True Range (ATR) untuk mengukur volatilitas asli pasar.
    Menghilangkan bias emosional dengan angka volatilitas matematis.
    """
    # Validasi input agar tidak merusak data asli
    df = df.copy()
    
    # 1. Hitung 3 komponen True Range (TR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    
    # 2. Ambil nilai maksimum dari ketiga komponen
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # 3. Hitung ATR menggunakan Exponential Moving Average (EMA)
    # Periode standar industri adalah 14 hari
    df['ATR'] = true_range.ewm(span=period, adjust=False).mean()
    
    return df
import pandas as pd
import numpy as np

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Menghitung Average True Range (ATR) untuk mengukur volatilitas pasar.
    Digeser (.shift(1)) agar fitur pada hari T+0 hanya menggunakan nilai volatilitas 
    dari hari T-1 ke belakang untuk mencegah data leakage.
    """
    df = df.copy()
    
    high_low = df['High'] - df['Low']                               
    high_close = (df['High'] - df['Close'].shift(1)).abs()          
    low_close = (df['Low'] - df['Close'].shift(1)).abs()            
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1) 
    
    # 1. Hitung ATR harian berdasarkan data penutupan pasar
    atr_raw = true_range.ewm(span=period, adjust=False).mean() 
    
    # 2. Geser 1 hari agar siap digunakan sebagai input pada pagi hari berikutnya (T+0)
    df['ATR'] = atr_raw.shift(1)
    return df

def calculate_technical_indicators(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ 
    Menghitung indikator momentum (RSI), Volume Ratio, dan Log Return.
    Seluruh output digeser (.shift(1)) sehingga fitur pada hari T+0 
    hanya mengetahui data dari hari T-1 ke belakang.
    """
    df = df.copy()
    
    # 1. Momentum (RSI)
    delta = df['Close'].diff()                                
    gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()  
    loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean() 
    rs = gain / (loss + 1e-9)
    rsi_raw = 100 - (100 / (1 + rs))
    
    # 2. Volume Ratio (Mendeteksi akumulasi volume terhadap MA20)
    volume_ma20 = df['Volume'].rolling(window=20).mean()
    volume_ratio_raw = df['Volume'] / (volume_ma20 + 1e-9)
    
    # 3. Log Return harian
    log_return_raw = np.log(df['Close'] / df['Close'].shift(1))
    
    # 4. Geser seluruh fitur sebanyak 1 hari untuk mengamankan dari data leakage
    df['RSI'] = rsi_raw.shift(1)
    df['Volume_Ratio'] = volume_ratio_raw.shift(1)
    df['Log_Return'] = log_return_raw.shift(1)
    
    return df

def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Mendeteksi Zona Support & Resistance Otomatis yang Adaptif terhadap Perubahan Peran (Role Reversal).
    Menggunakan ATR sebagai buffer/toleransi area imajiner agar tidak kaku pada satu angka.
    """
    df = df.copy()
    
    # Pastikan indikator ATR sudah dihitung sebelumnya untuk dasar area toleransi
    if 'ATR' not in df.columns:
        df["ATR_temp"] = calculate_atr(df)
    else:
        df['ATR_temp'] = df['ATR']

    # 1. Hitung batas absolut Donchian Channel dari data T-1 ke belakang (Mencegah Data Leakage)
    # Kita gunakan window lebih panjang (misal 20 hari / 1 bulan bursa) untuk mencerminkan swing wave yang kuat
    df['Raw_Support'] = df['Low'].shift(1).rolling(window=window).min()
    df['Raw_Resistance'] = df['High'].shift(1).rolling(window=window).max()
    
    # 2. Definisikan Zona/Area Imajiner menggunakan Buffer ATR (Standard Deviasi Volatilitas)
    # Area Support = Batas terendah s.d Batas terendah + (0.5 * ATR)
    # Area Resistance = Batas tertinggi s.d Batas tertinggi - (0.5 * ATR)
    df['Support_Zone_Low'] = df['Raw_Support']
    df['Support_Zone_High'] = df['Raw_Support'] + (0.5 * df['ATR_temp'])
    
    df['Resistance_Zone_Low'] = df['Raw_Resistance'] - (0.5 * df['ATR_temp'])
    df['Resistance_Zone_High'] = df['Raw_Resistance']

    # 3. Logika Role Reversal (S/R Flip)
    # Kita buat kolom dinamis untuk mendeteksi status harga saat ini terhadap level historis
    df['Support'] = df['Raw_Support']
    df['Resistance'] = df['Raw_Resistance']
    
    # Looping logis untuk mendeteksi persilangan peran (S/R Flip)
    # Jika harga close kemarin tembus di bawah support historis, maka support lama menjadi resistance baru
    for i in range(1, len(df)):
        close_yesterday = df['Close'].iloc[i-1]
        support_yesterday = df['Raw_Support'].iloc[i-1]
        resistance_yesterday = df['Raw_Resistance'].iloc[i-1]
        
        # S/R Flip: Breakout Downward (Support Jebol -> Jadi Resistance Baru)
        if close_yesterday < support_yesterday:
            df.loc[df.index[i], 'Resistance'] = support_yesterday
            # Cari support baru dari minimum lokal terdekat
            
        # S/R Flip: Breakout Upward (Resistance Jebol -> Jadi Support Baru)
        elif close_yesterday > resistance_yesterday:
            df.loc[df.index[i], 'Support'] = resistance_yesterday

    # Bersihkan NaN awal dan hapus kolom temporary
    df['Support'] = df['Support'].ffill().bfill()
    df['Resistance'] = df['Resistance'].ffill().bfill()
    df.drop(columns=['ATR_temp', 'Raw_Support', 'Raw_Resistance', 'Support', 'Resistance'], inplace=True)
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = calculate_atr(df)
    df = calculate_technical_indicators(df)
    df = detect_support_resistance(df)
    return df

# ------------------------------------------
# hari_ini = datetime.now().strftime('%Y-%m-%d')
# enam_bulan_lalu = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
# df_bumi = load_stock_data(ticker="BUMI.JK", start_date=enam_bulan_lalu, end_date=hari_ini)
# df = calculate_technical_indicators(df_bumi)
# print(df)
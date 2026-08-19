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

    # ADDITIONAL INDICATOR
    # 5. Momentum MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = (macd_line - signal_line).shift(1)

    # 6. Jarak Harga terhadap MA-20 (Mean Reversion)
    ma20 = df['Close'].rolling(window=20).mean()
    df['Dist_to_MA20'] = ((df['Close'] - ma20) / ma20).shift(1) # Geser 1 hari

    # 7. Volatilitas Harga Historis (Historical Volatility 5 Hari)
    df['Hist_Volatility_5d'] = log_return_raw.rolling(window=5).std().shift(1)
    
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

    close_yesterday = df['Close'].shift(1)
    # Bersihkan NaN awal dan hapus kolom temporary
    df['distance_to_support'] = ((close_yesterday - df['Support_Zone_High']) / (df['Support_Zone_High'] + 1e-9)) * 100
    df['distance_to_resistance'] = ((df['Resistance_Zone_Low'] - close_yesterday) / (close_yesterday + 1e-9)) * 100
    # df['distance_to_support'] = close_yesterday - df['Support_Zone_High']
    # df['distance_to_resistance'] = df['Resistance_Zone_Low'] - close_yesterday

    # Lakukan ffill dan bfill untuk membersihkan NaN jika ada pembagi nol
    df['distance_to_support'] = df['distance_to_support'].ffill().bfill()
    df['distance_to_resistance'] = df['distance_to_resistance'].ffill().bfill()
    df['Support'] = df['Support'].ffill().bfill()
    df['Resistance'] = df['Resistance'].ffill().bfill()

    df.drop(columns=['ATR_temp', 'Raw_Support', 'Raw_Resistance'], inplace=True)
    
    return df

def calculate_climax_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung fitur khusus untuk mendeteksi fase 'Jenuh Jual' (Selling Climax)
    dan penolakan harga (Rejection). Semua di-shift(1) agar bebas data leakage.
    """
    df = df.copy()
    
    # 1. Short-Term RSI (RSI-3) - Sangat sensitif terhadap oversold ekstrim
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=3, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=3, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    rsi_3_raw = 100 - (100 / (1 + rs))
    df['RSI_3'] = rsi_3_raw.shift(1)
    
    # 2. Bollinger Bands %B (MA20 ± 2 StdDev) - Jarak ke luar pita bawah
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    upper_band = ma20 + (2 * std20)
    lower_band = ma20 - (2 * std20)
    bb_pct_raw = (df['Close'] - lower_band) / (upper_band - lower_band + 1e-9)
    df['BB_Pct'] = bb_pct_raw.shift(1)
    
    # 3. Capitulation Volume (Volume Ratio x Log Return) - Panic selling
    volume_ma20 = df['Volume'].rolling(window=20).mean()
    volume_ratio_raw = df['Volume'] / (volume_ma20 + 1e-9)
    log_return_raw = np.log(df['Close'] / df['Close'].shift(1))
    capitulation_vol_raw = volume_ratio_raw * log_return_raw
    df['Capitulation_Vol'] = capitulation_vol_raw.shift(1)
    
    # 4, 5, & 6. Proporsi Candlestick (Lower Shadow, Upper Shadow, Real Body)
    # Daripada mendeteksi 1 pola baku (Hammer), kita ubah menjadi nilai persentase kontinu
    high_low_range = df['High'] - df['Low'] + 1e-9
    min_open_close = df[['Open', 'Close']].min(axis=1)
    max_open_close = df[['Open', 'Close']].max(axis=1)
    
    lower_shadow_pct_raw = (min_open_close - df['Low']) / high_low_range
    upper_shadow_pct_raw = (df['High'] - max_open_close) / high_low_range
    body_pct_raw = (df['Close'] - df['Open']).abs() / high_low_range
    
    df['Lower_Shadow_Pct'] = lower_shadow_pct_raw.shift(1)
    df['Upper_Shadow_Pct'] = upper_shadow_pct_raw.shift(1)
    df['Body_Pct'] = body_pct_raw.shift(1)
    
    # 7. Williams %R (14) - Sensitif terhadap penutupan dekat harga terendah
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    williams_r_raw = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low + 1e-9))
    df['Williams_R'] = williams_r_raw.shift(1)
    
    return df

def calculate_macro_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur konteks makro & volatilitas untuk membantu model membedakan
    'koreksi sehat' (dip) vs 'awal downtrend' (crash).

    Semua di-shift(1) agar bebas dari data leakage.
    
    Fitur yang ditambahkan:
    - Dist_to_MA50     : Jarak harga ke MA50 — apakah harga jauh di bawah MA50 (regime bearish)?
    - EMA50_Slope      : Kemiringan EMA50 dalam 10 hari — apakah trend makro naik atau turun?
    - ATR_Pct_Change   : Perubahan ATR dalam 5 hari — lonjakan tiba-tiba = kepanikan/crash
    - BB_Width         : Lebar Bollinger Band (MA20 ± 2σ) — makin lebar = makin volatil
    - Volume_Spike     : Volume hari ini vs rata-rata Volume 5 hari — lonjakan = selling pressure
    """
    df = df.copy()

    # 1. Jarak ke MA50 — Regime Filter Jangka Menengah
    ma50 = df['Close'].rolling(window=50).mean()
    df['Dist_to_MA50'] = ((df['Close'] - ma50) / (ma50 + 1e-9)).shift(1)

    # 2. Kemiringan EMA50 dalam 10 hari — Menangkap arah trend makro
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA50_Slope'] = ((ema50 - ema50.shift(10)) / (ema50.shift(10) + 1e-9)).shift(1)

    # 3. Perubahan ATR dalam 5 hari — Deteksi lonjakan volatilitas/kepanikan
    if 'ATR' in df.columns:
        atr_raw = df['ATR'].shift(-1)  # Ambil ATR sebelum dishift (nilai asli)
    else:
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_raw = true_range.ewm(span=14, adjust=False).mean()
    df['ATR_Pct_Change'] = (atr_raw.pct_change(periods=5)).shift(1)

    # 4. Lebar Bollinger Band — Makin lebar = volatilitas makin tinggi
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    bb_width_raw = (2 * 2 * std20) / (ma20 + 1e-9)  # Lebar penuh = 4 std / MA20
    df['BB_Width'] = bb_width_raw.shift(1)

    # 5. Volume Spike vs MA5 — Lonjakan volume jangka pendek (tekanan jual mendadak)
    vol_ma5 = df['Volume'].rolling(window=5).mean()
    df['Volume_Spike'] = (df['Volume'] / (vol_ma5 + 1e-9)).shift(1)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = calculate_atr(df)
    df = calculate_technical_indicators(df)
    df = detect_support_resistance(df)
    df = calculate_climax_features(df)
    df = calculate_macro_context_features(df)
    return df

# ------------------------------------------
# hari_ini = datetime.now().strftime('%Y-%m-%d')
# enam_bulan_lalu = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
# df_bumi = load_stock_data(ticker="BUMI.JK", start_date=enam_bulan_lalu, end_date=hari_ini)
# df = calculate_technical_indicators(df_bumi)
# print(df)
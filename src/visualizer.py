import os
import pandas as pd
import matplotlib.dates as mdates
import mplfinance as mpf

def plot_daily_candlestick(df: pd.DataFrame, ticker_name: str, save_dir: str = "../data/processed") -> None:
    """
    Membuat grafik candlestick harian terintegrasi dengan volume.
    Sumbu X otomatis diset tepat pada setiap awal bulan.
    Hasil grafik otomatis diekspor menjadi file gambar PNG.
    
    Parameters:
    - df (pd.DataFrame): Dataframe saham yang memiliki DatetimeIndex dan kolom OHLCV.
    - ticker_name (str): Nama atau kode emiten untuk judul grafik (contoh: 'BUMI').
    - save_dir (str): Folder tempat menyimpan hasil ekspor grafik.
    """
    # 1. Validasi mutlak syarat data mplfinance
    df = df.copy()
    
    # Buang baris terakhir jika High, Low, Close, dan Volume semuanya NaN
    if not df.empty and df.iloc[-1][['High', 'Low', 'Close', 'Volume']].isna().all():
        df = df.iloc[:-1]
        
    df.index = pd.to_datetime(df.index)
    
    # 2. Kustomisasi Tema Grafik Standar Industri (Clean & Professional)
    market_colors = mpf.make_marketcolors(
        up='green', down='red',      # Hijau saat naik, merah saat turun
        edge='inherit',              
        wick='inherit',              # Warna ekor candle mengikuti badan
        volume='in',                 # Warna volume sinkron dengan candle
        inherit=True
    )
    custom_style = mpf.make_mpf_style(
        marketcolors=market_colors, 
        gridstyle='--', 
        gridcolor='#E0E0E0'
    )
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"grafik_harian_{ticker_name.lower()}.png")

    try:
        is_monthly = False
        if len(df) > 150:
            print(f"Data terlalu banyak ({len(df)} baris), melakukan binning bulanan...")
            df = df.resample('MS').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            is_monthly = True
            
        period_str = "Bulanan" if is_monthly else "Harian"
        print(f"Menghasilkan grafik candlestick {period_str.lower()} untuk {ticker_name}...")
        
        # Cari tanggal hari pertama trading di bulan Januari setiap tahunnya untuk garis vertikal
        first_days_of_year = df.groupby(df.index.year).head(1)
        jan_dates = first_days_of_year[first_days_of_year.index.month == 1].index.tolist()
        
        plot_kwargs = dict(
            type='candle',
            volume=True,
            style=custom_style,
            figsize=(14, 8),
            title=f"Grafik Candlestick {period_str} & Volume Emiten {ticker_name.upper()}",
            ylabel="Harga Saham (Rp)",
            ylabel_lower="Volume Transaksi",
            show_nontrading=True,
            returnfig=True
        )
        
        # Tambahkan garis vertikal jika ada tanggal bulan Januari
        if jan_dates:
            plot_kwargs['vlines'] = dict(vlines=jan_dates, colors='#555555', linestyle='--', linewidths=1.5, alpha=0.6)
            
        # 3. Render Candlestick dan Volume (Panel 0 dan Panel 1)
        fig, axlist = mpf.plot(df, **plot_kwargs)

        # --- KORREKSI AXIS X SUPAYA PAS DENGAN VLINES DAN TANGGAL SEBENARNYA ---
        # axlist[0] adalah panel utama (candlestick)
        ax = axlist[0]
        if is_monthly:
            # Jika bulanan, tampilkan major ticks per 1 Tahun tepat di bulan Januari
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        else:
            # Jika harian, tampilkan major ticks setiap awal bulan
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        # Merapikan kemiringan teks tanggal agar tidak bertumpuk
        fig.autofmt_xdate()
        
        # Tampilkan grafik di layar
        mpf.show()
        
    except Exception as e:
        raise RuntimeError(f"Gagal memvisualisasikan grafik untuk {ticker_name}. Error: {str(e)}")
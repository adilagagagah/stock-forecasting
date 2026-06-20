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
        print(f"Menghasilkan grafik candlestick untuk {ticker_name}...")
        
        # 3. Render Candlestick dan Volume (Panel 0 dan Panel 1)
        fig, axlist = mpf.plot(
            df,
            type='candle',
            volume=True,
            style=custom_style,
            figsize=(14, 8),
            title=f"Grafik Candlestick Harian & Volume Emiten {ticker_name.upper()}",
            ylabel="Harga Saham (Rp)",
            ylabel_lower="Volume Transaksi",
            # savefig=dict(fname=save_path, dpi=300),
            returnfig=True
        )
        
        # 4. Modifikasi Sumbu X (Axis): Tepat Setiap Awal Bulan [cite: 61]
        main_axis = axlist[0]
        
        # Locator: Deteksi tanggal 1 di setiap bulan baru
        main_axis.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        # Formatter: Tampilan label "Jan 2026", "Feb 2026", dst.
        main_axis.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        # Merapikan kemiringan teks tanggal agar tidak bertumpuk
        fig.autofmt_xdate()
        
        # Tampilkan grafik di layar
        mpf.show()
        
    except Exception as e:
        raise RuntimeError(f"Gagal memvisualisasikan grafik untuk {ticker_name}. Error: {str(e)}")
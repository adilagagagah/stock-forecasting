import os
import pandas as pd
import math
import matplotlib.dates as mdates
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns

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

def comparison_chart(first_data_list=[], second_data_list=[], 
                        first_label_list=[], second_label_list=[], 
                        title_list=[], set_xlabel_list=[], set_ylabel_list=[],
                        suptitle='', label_format=lambda x:f"{x:.2f}", 
                        point_data_label=False):
    """
    Membuat grafik perbandingan beberapa dataset.
    
    Parameters:
    - figsize (tuple): Ukuran grafik.
    - first_data_list (list): List data pertama.
    - second_data_list (list): List data kedua.
    - first_label_list (list): List label data pertama.
    - second_label_list (list): List label data kedua.
    - title_list (list): List judul grafik.
    - set_xlabel_list (list): List label xlabel.
    - set_ylabel_list (list): List label ylabel.
    - suptitle (str): Judul utama grafik.
    - point_data_label (bool): Apakah akan menampilkan label data di setiap titik grafik.
    - label_format (str): Format label data.
    """
    # Set tema grafik profesional
    sns.set_theme(style="whitegrid")
    n_graphs = len(first_data_list)
    n_col = min(3, n_graphs)
    n_row = math.ceil(n_graphs / n_col)
    figsize = (min(n_graphs * 7, 20), n_row * 6)

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    axes = axes.flatten()
    main_color_list = ['#1f77b4', '#2ca02c', '#9467bd']
    second_color_list = ['#ff7f0e', '#d62728', '#8c564b']

    for i in range(n_graphs):
        axes[i].plot(first_data_list[i], label=first_label_list[i], color=main_color_list[0], linewidth=2)
        axes[i].plot(second_data_list[i], label=second_label_list[i], color=second_color_list[0], linestyle='--', linewidth=2)
        axes[i].set_title(title_list[i], fontsize=12, fontweight='bold')
        axes[i].set_xlabel(set_xlabel_list[i], fontsize=10)
        axes[i].set_ylabel(set_ylabel_list[i], fontsize=10)
        axes[i].legend()

        if point_data_label:
            for j, txt in enumerate(first_data_list[i]):
                axes[i].annotate(label_format(txt), (j, first_data_list[i][j]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color='#1f77b4')
            for j, txt in enumerate(second_data_list[i]):
                axes[i].annotate(label_format(txt), (j, second_data_list[i][j]), textcoords="offset points", xytext=(0,-15), ha='center', color='#ff7f0e')

    for i in range(n_graphs, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.06)
    plt.show()
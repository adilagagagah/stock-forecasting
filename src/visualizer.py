import os
import pandas as pd
import math
import matplotlib.dates as mdates
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns

def plot_daily_candlestick(df: pd.DataFrame, ticker_name: str, save_dir: str = "../data/processed", start_date: str = None, end_date: str = None) -> None:
    """
    Membuat grafik candlestick harian terintegrasi dengan volume.
    Sumbu X otomatis diset tepat pada setiap awal bulan.
    Hasil grafik otomatis diekspor menjadi file gambar PNG.
    
    Parameters:
    - df (pd.DataFrame): Dataframe saham yang memiliki DatetimeIndex dan kolom OHLCV.
    - ticker_name (str): Nama atau kode emiten untuk judul grafik (contoh: 'BUMI').
    - save_dir (str): Folder tempat menyimpan hasil ekspor grafik.
    - start_date (str): Tanggal mulai rentang grafik (format YYYY-MM-DD).
    - end_date (str): Tanggal akhir rentang grafik (format YYYY-MM-DD).
    """
    # 1. Validasi mutlak syarat data mplfinance
    df = df.copy()
    
    # Buang baris terakhir jika High, Low, Close, dan Volume semuanya NaN
    if not df.empty and df.iloc[-1][['High', 'Low', 'Close', 'Volume']].isna().all():
        df = df.iloc[:-1]
        
    df.index = pd.to_datetime(df.index)
    
    # Filter rentang tanggal
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if df.empty:
        print("Data kosong setelah filter tanggal. Tidak ada grafik yang digambar.")
        return
    
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
        if len(df) > 365:
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

def plot_interactive_candlestick(df: pd.DataFrame, ticker_name: str, start_date: str = None, end_date: str = None, predictions_df: pd.DataFrame = None, equity_curve_df: pd.DataFrame = None, trades_df: pd.DataFrame = None, show_raw_buy_signals: bool = False):
    """
    Membuat grafik candlestick interaktif menggunakan Plotly.
    Sangat cocok digunakan di dalam Jupyter Notebook untuk zoom dan pan.
    
    Parameters:
    - df (pd.DataFrame): Dataframe saham yang memiliki DatetimeIndex dan kolom OHLCV.
    - ticker_name (str): Nama atau kode emiten.
    - start_date (str): Tanggal mulai (opsional).
    - end_date (str): Tanggal akhir (opsional).
    - predictions_df (pd.DataFrame): Dataframe hasil prediksi dari backtester (opsional).
    - equity_curve_df (pd.DataFrame): Dataframe historis ekuitas dari backtester (opsional).
    - trades_df (pd.DataFrame): Dataframe hasil trades/penjualan (opsional).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Library 'plotly' tidak ditemukan. Silakan install menggunakan: pip install plotly")
        return
        
    df = df.copy()
    if not df.empty and df.iloc[-1][['High', 'Low', 'Close', 'Volume']].isna().all():
        df = df.iloc[:-1]
        
    df.index = pd.to_datetime(df.index)
    
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
        
    if df.empty:
        print("Data kosong. Tidak ada grafik yang digambar.")
        return

    # Buat figure dengan 3 baris (Candlestick, Equity Curve, dan Volume)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'Candlestick {ticker_name.upper()}', 'Equity Curve', 'Volume'),
                        row_width=[0.15, 0.25, 0.6],
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'),
                  row=1, col=1, secondary_y=False)

    # EMA 5 & EMA 10 & EMA 20
    ema5 = df['Close'].ewm(span=5, adjust=False).mean()
    ema10 = df['Close'].ewm(span=10, adjust=False).mean()
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    
    fig.add_trace(go.Scatter(x=df.index, y=ema5, mode='lines', 
                             line=dict(color='red', width=1.5), 
                             name='EMA 5', hovertemplate='EMA 5: Rp%{y:,.0f}<extra></extra>'),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=ema10, mode='lines', 
                             line=dict(color='orange', width=1.5), 
                             name='EMA 10', hovertemplate='EMA 10: Rp%{y:,.0f}<extra></extra>'),
                  row=1, col=1, secondary_y=False)
                  
    fig.add_trace(go.Scatter(x=df.index, y=ema20, mode='lines', 
                             line=dict(color='purple', width=1.5), 
                             name='EMA 20', hovertemplate='EMA 20: Rp%{y:,.0f}<extra></extra>'),
                  row=1, col=1, secondary_y=False)

    # Tambahkan Hasil Prediksi jika ada
    if predictions_df is not None and not predictions_df.empty:
        pred_df = predictions_df.reindex(df.index)
        
        # 1. Plot Sinyal Mentah (Debug) dari Model Murni (Jika diaktifkan)
        if show_raw_buy_signals and 'raw_buy_signal' in pred_df.columns:
            raw_signals = pred_df[pred_df['raw_buy_signal'] == True]
            if not raw_signals.empty:
                # Letakkan marker panah sedikit di bawah harga Low
                signal_prices = df.loc[raw_signals.index, 'Low'] * 0.98
                
                # Susun data hasil prediksi untuk ditampilkan di hover
                hover_data = raw_signals[['pred_return', 'pred_risk', 'actual_rr_ratio', 'pred_trend_slope', 'pred_days_to_max', 'pred_days_to_min']].values
                
                fig.add_trace(go.Scatter(
                    x=raw_signals.index, y=signal_prices, mode='markers',
                    customdata=hover_data,
                    marker=dict(symbol='triangle-up', size=14, color='magenta', line=dict(width=1, color='darkmagenta')),
                    name='RAW Model Buy Signal (Debug)', 
                    hovertemplate=(
                        '<b>Sinyal Mentah (Model Jenuh Jual)</b><br>' +
                        'Return Pred: %{customdata[0]:.2%}<br>' +
                        'Risk Pred: %{customdata[1]:.2%}<br>' +
                        'RR Ratio: %{customdata[2]:.2f}<br>' +
                        'Trend Slope: %{customdata[3]:.2%}<br>' +
                        'T+ (Max/Min): %{customdata[4]:.0f} Hari / %{customdata[5]:.0f} Hari' +
                        '<extra></extra>'
                    )
                ), row=1, col=1, secondary_y=False)
                
        # 2. Plot Prediksi RR, Return & Risk di Sumbu Y Sekunder
        if 'actual_rr_ratio' in pred_df.columns:
            fig.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df['actual_rr_ratio'], mode='lines', 
                line=dict(width=0), opacity=0, showlegend=False,
                name='Actual RR Ratio', hovertemplate='%{y:.2f}'
            ), row=1, col=1, secondary_y=True)

        if 'pred_return' in pred_df.columns:
            fig.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df['pred_return'], mode='lines', 
                line=dict(dash='dot', color='rgba(0, 128, 0, 0.4)'),
                name='Pred Return', hovertemplate='%{y:.2%}'
            ), row=1, col=1, secondary_y=True)
            
        if 'pred_risk' in pred_df.columns:
            fig.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df['pred_risk'].abs(), mode='lines', 
                line=dict(dash='dot', color='rgba(255, 0, 0, 0.4)'),
                name='Pred Risk', hovertemplate='%{y:.2%}'
            ), row=1, col=1, secondary_y=True)
            
        # 3. Tambahkan 3 variabel lain ke hover (invisible lines agar tidak mengotori chart)
        if 'pred_trend_slope' in pred_df.columns:
            fig.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df['pred_trend_slope'], mode='lines', 
                line=dict(width=0), opacity=0, showlegend=False,
                name='pred Trend Slope', hovertemplate='%{y:.2%}'
            ), row=1, col=1, secondary_y=True)
            
        if 'pred_days_to_max' in pred_df.columns:
            fig.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df['pred_days_to_max'], mode='lines', 
                line=dict(width=0), opacity=0, showlegend=False,
                name='Days to Max', hovertemplate='%{y:.0f} Hari'
            ), row=1, col=1, secondary_y=True)
            
        if 'pred_days_to_min' in pred_df.columns:
            fig.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df['pred_days_to_min'], mode='lines', 
                line=dict(width=0), opacity=0, showlegend=False,
                name='Days to Min', hovertemplate='%{y:.0f} Hari'
            ), row=1, col=1, secondary_y=True)

    # Tambahkan Sinyal Beli dan Jual dari trades_df
    if trades_df is not None and not trades_df.empty:
        t_df = trades_df.copy()
        if 'entry_date' in t_df.columns:
            t_df['entry_date'] = pd.to_datetime(t_df['entry_date'])
        if 'exit_date' in t_df.columns:
            t_df['exit_date'] = pd.to_datetime(t_df['exit_date'])
            
        # --- Plot Buy Signals ---
        t_df_buy = t_df[t_df['entry_date'].isin(df.index)]
        buy_y = []
        buy_hover = []
        for idx, row in t_df_buy.iterrows():
            d = row['entry_date']
            if d in df.index:
                y_pos = df.loc[d, 'Low'] * 0.95
            else:
                y_pos = row['entry_price'] * 0.95
            buy_y.append(y_pos)
            
            text = (f"ID Transaksi: {row.get('trade_id', '')}<br>"
                    f"Beli: {d.strftime('%Y-%m-%d')}<br>"
                    f"Harga: Rp{row['entry_price']:,.2f}<br>"
                    f"Lot: {row['lots']}<br>"
                    f"Value: Rp{row.get('capital_spent', 0):,.2f}<br>"
                    f"TP: Rp{row.get('tp_price', 0):,.2f}<br>"
                    f"SL: Rp{row.get('sl_price', 0):,.2f}")
            buy_hover.append(text)
            
        if not t_df_buy.empty:
            fig.add_trace(go.Scatter(
                x=t_df_buy['entry_date'],
                y=buy_y,
                mode='markers',
                marker=dict(symbol='triangle-up', color='blue', size=12, line=dict(width=1, color='DarkSlateGrey')),
                name='Sinyal Beli',
                text=buy_hover,
                hovertemplate='%{text}<extra></extra>'
            ), row=1, col=1, secondary_y=False)

        # --- Plot Sell Signals ---
        t_df_sell = t_df[t_df['exit_date'].isin(df.index)]
        sell_y = []
        sell_hover = []
        for idx, row in t_df_sell.iterrows():
            d = row['exit_date']
            if d in df.index:
                y_pos = df.loc[d, 'High'] * 1.05
            else:
                y_pos = row['exit_price'] * 1.05
            sell_y.append(y_pos)
            
            text = (f"ID Pembelian: {row.get('trade_id', '')}<br>"
                    f"Jual: {d.strftime('%Y-%m-%d')}<br>"
                    f"Harga: Rp{row['exit_price']:,.2f}<br>"
                    f"Lot: {row['lots']}<br>"
                    f"Alasan Jual: {row['exit_reason']}<br>"
                    f"Profit: Rp{row['net_profit']:,.2f} ({row['roi_pct']:+.2f}%)")
            sell_hover.append(text)
            
        if not t_df_sell.empty:
            fig.add_trace(go.Scatter(
                x=t_df_sell['exit_date'],
                y=sell_y,
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=12, line=dict(width=1, color='DarkSlateGrey')),
                name='Sinyal Jual',
                text=sell_hover,
                hovertemplate='%{text}<extra></extra>'
            ), row=1, col=1, secondary_y=False)

    # Tambahkan Equity Curve jika ada
    if equity_curve_df is not None and not equity_curve_df.empty:
        equity_df = equity_curve_df.copy()
        if 'date' in equity_df.columns:
            equity_df = equity_df.set_index('date')
        equity_df.index = pd.to_datetime(equity_df.index)
        equity_df = equity_df.reindex(df.index)
        if 'cash' in equity_df.columns:
            # Isi ffill untuk mengisi hari-hari di mana tidak ada record transaksi tapi cash tetap
            equity_df['cash'] = equity_df['cash'].ffill()
            fig.add_trace(go.Scatter(
                x=equity_df.index, 
                y=equity_df['cash'], 
                mode='lines', 
                line=dict(color='royalblue', width=2),
                name='Total Cash (Rp)'
            ), row=2, col=1)
            fig.update_yaxes(title_text="Equity (Rp)", row=2, col=1)

    # Volume (Warna hijau jika harga naik, merah jika turun)
    colors = ['green' if close >= open else 'red' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'),
                  row=3, col=1)

    # Layout dan range slider
    fig.update_layout(
        title=f"Grafik Interaktif Emiten {ticker_name.upper()}",
        yaxis_title='Harga Saham (Rp)',
        xaxis_rangeslider_visible=False,
        height=700,
        template='plotly_white',
        hovermode='closest'
    )
    
    if predictions_df is not None and not predictions_df.empty:
        fig.update_yaxes(title_text="Prediksi (%)", tickformat='.0%', range=[-0.2, 0.2], secondary_y=True, row=1, col=1)
    
    # Hide non-trading days (weekends/holidays) by excluding them from x-axis
    # Note: Plotly doesn't natively hide gaps, we can convert x-axis to category but it disables some date features.
    # We will just rely on the default time series axis for now.
    
    fig.show()
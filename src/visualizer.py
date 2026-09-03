import os
import pandas as pd
import math
import matplotlib.dates as mdates
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_interactive_candlestick(
    df: pd.DataFrame, 
    ticker_name: str, 
    start_date: str = None, 
    end_date: str = None, 
    predictions_df: pd.DataFrame = None, 
    df_equity: pd.DataFrame = None, 
    df_trades: pd.DataFrame = None, 
    show_raw_buy_signals: bool = False
):
    """
    Membuat grafik candlestick interaktif menggunakan Plotly.
    Sangat cocok digunakan di dalam Jupyter Notebook untuk zoom dan pan.
    
    Parameters:
    - df (pd.DataFrame): Dataframe saham yang memiliki DatetimeIndex dan kolom OHLCV.
    - ticker_name (str): Nama atau kode emiten.
    - start_date (str): Tanggal mulai (opsional).
    - end_date (str): Tanggal akhir (opsional).
    - predictions_df (pd.DataFrame): Dataframe hasil prediksi dari backtester (opsional).
    - df_equity (pd.DataFrame): Dataframe historis ekuitas dari backtester (opsional).
    - df_trades (pd.DataFrame): Dataframe hasil trades/penjualan (opsional).
    """
        
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

    # EMA 5, EMA 10, EMA 20, & EMA 50
    ema5 = df['Close'].ewm(span=5, adjust=False).mean()
    ema10 = df['Close'].ewm(span=10, adjust=False).mean()
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    
    fig.add_trace(go.Scatter(x=df.index, y=ema5, mode='lines', 
                             line=dict(color='purple', width=1), 
                             name='EMA 5', hovertemplate='EMA 5: Rp%{y:,.0f}<extra></extra>'),

                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=ema10, mode='lines', 
                             line=dict(color='red', width=1), 
                             name='EMA 10', hovertemplate='EMA 10: Rp%{y:,.0f}<extra></extra>'),
                  row=1, col=1, secondary_y=False)
                  
    fig.add_trace(go.Scatter(x=df.index, y=ema20, mode='lines', 
                             line=dict(color='orange', width=1), 
                             name='EMA 20', hovertemplate='EMA 20: Rp%{y:,.0f}<extra></extra>'),
                  row=1, col=1, secondary_y=False)
                  
    fig.add_trace(go.Scatter(x=df.index, y=ema50, mode='lines', 
                             line=dict(color='brown', width=1), 
                             name='EMA 50', hovertemplate='EMA 50: Rp%{y:,.0f}<extra></extra>'),
                  row=1, col=1, secondary_y=False)
    
    # Tambahkan hasil prediksi mentah (hanya terjadi jika (pred_is_dip >= min_dip_proba) or (pred_trend_slope >= min_trend_slope) or (actual_rr_ratio >= min_rr_ratio))
    if predictions_df is not None and not predictions_df.empty:
        pred_df = predictions_df.reindex(df.index)

        if show_raw_buy_signals and 'raw_buy_signal' in pred_df.columns:
            raw_signals = pred_df[pred_df['raw_buy_signal'].isin([True, 1])]

            if not raw_signals.empty:
                signal_prices = df.loc[raw_signals.index, 'Low'] * 0.96
                
                # Cek ketersediaan kolom-kolom prediksi lengkap
                req_cols = ['pred_is_dip', 'pred_return', 'pred_risk', 'actual_rr_ratio', 'pred_trend_slope']
                has_full_pred = all(col in raw_signals.columns for col in req_cols)
                
                if has_full_pred:
                    customdata = raw_signals[req_cols].values
                    text_data = None
                    hovertemplate = (
                        '<b>Sinyal Beli (Prediksi)</b><br>' +
                        'Probabilitas Dip: %{customdata[0]:.2%} [Cond 1]<br>' +
                        'Return Pred: %{customdata[1]:.2%}<br>' +
                        'Risk Pred: %{customdata[2]:.2%}<br>' +
                        'RR Ratio: %{customdata[3]:.2f} [Cond 2]<br>' +
                        'Trend Slope: %{customdata[4]:.2%} [Cond 3]<br>' +
                        '<extra></extra>'
                    )
                else:
                    # Mode Label Training / In-Sample (menampilkan nilai label yang tersedia)
                    hover_texts = []
                    for idx, row in raw_signals.iterrows():
                        date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                        lines = ["<b>Label Training</b>", f"Tanggal: {date_str}"]
                        if 'is_dip' in row and pd.notna(row['is_dip']):
                            lines.append(f"is_dip: {int(row['is_dip'])}")
                        if 'return' in row and pd.notna(row['return']):
                            lines.append(f"Return: {row['return']:.2%}")
                        if 'risk' in row and pd.notna(row['risk']):
                            lines.append(f"Risk: {row['risk']:.2%}")
                        if 'trend_slope' in row and pd.notna(row['trend_slope']):
                            lines.append(f"Trend Slope: {row['trend_slope']:.4f}")
                        if 'actual_rr_ratio' in row and pd.notna(row['actual_rr_ratio']):
                            lines.append(f"RR Ratio: {row['actual_rr_ratio']:.2f}")
                        hover_texts.append("<br>".join(lines))
                    
                    customdata = None
                    text_data = hover_texts
                    hovertemplate = "%{text}<extra></extra>"

                fig.add_trace(go.Scatter(
                    x=raw_signals.index, y=signal_prices, mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='dodgerblue', line=dict(width=1, color='darkblue')),
                    name='Raw Signal / Label (Biru)',
                    customdata=customdata,
                    text=text_data,
                    hovertemplate=hovertemplate
                ), row=1, col=1, secondary_y=False)
            
    # Tambahkan Sinyal Beli dan Jual dari df_trades
    if df_trades is not None and not df_trades.empty:
        df_trds = df_trades.copy()
        if 'entry_date' in df_trds.columns:
            df_trds['entry_date'] = pd.to_datetime(df_trds['entry_date'])
        if 'exit_date' in df_trds.columns:
            df_trds['exit_date'] = pd.to_datetime(df_trds['exit_date'])
            
        # --- Plot Buy Signals ---
        df_trds_buy = df_trds[df_trds['entry_date'].isin(df.index)]
        buy_y = []
        buy_hover = []
        for idx, row in df_trds_buy.iterrows():
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
            
        if not df_trds_buy.empty:
            fig.add_trace(go.Scatter(
                x=df_trds_buy['entry_date'],
                y=buy_y,
                mode='markers',
                marker=dict(symbol='triangle-up', color='limegreen', size=12, line=dict(width=1, color='darkgreen')),
                name='Sinyal Beli (Hijau)',
                text=buy_hover,
                hovertemplate='%{text}<extra></extra>'
            ), row=1, col=1, secondary_y=False)

        # --- Plot Sell Signals ---
        df_trds_sell = df_trds[df_trds['exit_date'].isin(df.index)]
        sell_y = []
        sell_hover = []
        for idx, row in df_trds_sell.iterrows():
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
            
        if not df_trds_sell.empty:
            fig.add_trace(go.Scatter(
                x=df_trds_sell['exit_date'],
                y=sell_y,
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=12, line=dict(width=1, color='DarkSlateGrey')),
                name='Sinyal Jual',
                text=sell_hover,
                hovertemplate='%{text}<extra></extra>'
            ), row=1, col=1, secondary_y=False)
    
    # Tambahkan Equity Curve dari df_equity jika ada
    if df_equity is not None and not df_equity.empty:
        df_eqty = df_equity.copy()
        if 'date' in df_eqty.columns:
            df_eqty = df_eqty.set_index('date')
        df_eqty.index = pd.to_datetime(df_eqty.index)
        df_eqty = df_eqty.reindex(df.index)
        if 'cash' in df_eqty.columns:
            # Isi ffill untuk mengisi hari-hari di mana tidak ada record transaksi tapi cash tetap
            df_eqty['cash'] = df_eqty['cash'].ffill()
            fig.add_trace(go.Scatter(
                x=df_eqty.index, 
                y=df_eqty['cash'], 
                mode='lines', 
                line=dict(color='royalblue', width=2),
                name='Total Cash (Rp)'
            ), row=2, col=1)
            fig.update_yaxes(title_text="Equity (Rp)", row=2, col=1)

    # Volume (Warna hijau jika harga naik, merah jika turun)
    colors = ['green' if close >= open else 'red' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume', showlegend=False),
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
    
    fig.show()

def cek_missing_value(X, y):
    # Pastikan baris tidak mengandung missing values
    total_missing_X = X.isna().sum().sum()
    total_missing_y = y.isna().sum().sum()

    print("--- PEMERIKSAAN DATA ---")
    print(f"Total missing values di X{X.shape}: {total_missing_X}")
    print(f"Total missing values di y{y.shape}: {total_missing_y}")
    print("-" * 30)

    if total_missing_X == 0 and total_missing_y == 0:
        print("Data Bersih, data X dan y 100% bersih dari missing values.")
    else:
        print("Masih ditemukan nilai kosong pada data.")
        if total_missing_X > 0:
            print(f" -> Segera tangani {total_missing_X} nilai kosong pada X.")
        if total_missing_y > 0:
            print(f" -> Segera tangani {total_missing_y} nilai kosong pada y (disarankan untuk menghapus baris yang kosong pada target).")

    # Heatmap variabel yang mengandung missing value
    sns.heatmap(
        pd.concat([X, y], axis=1).isna(), 
        cbar=False, 
        cmap='binary', 
        yticklabels=False
    )
    plt.title('Heatmap Missing Value X dan y (Hitam = Missing value)')
    plt.tight_layout()
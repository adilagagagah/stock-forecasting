import pandas as pd
import yfinance as yf
import os

def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Mengambil data historis saham secara real-time/historical dari Yahoo Finance.
    Menggunakan standard industri dengan handling error dan pembersihan format data.
    
    Parameters:
    - ticker (str): Kode saham (contoh: 'BUMI.JK')
    - start_date (str): Tanggal awal penarikan data (YYYY-MM-DD)
    - end_date (str): Tanggal akhir penarikan data (YYYY-MM-DD)
    """
    # Menjamin kode ticker menggunakan format Yahoo Finance untuk BEI jika lupa ditambahkan
    if not ticker.endswith('.JK') and len(ticker) == 4:
        ticker = f"{ticker.upper()}.JK"
        
    try:
        print(f"Mengunduh data terbaru untuk {ticker} dari Yahoo Finance...")
        
        # Mengunduh data dari yfinance
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Validasi: Apakah data kosong?
        if df.empty:
            raise ValueError(f"Data tidak ditemukan untuk ticker {ticker} pada rentang tanggal tersebut.")
            
        # Standardisasi format DataFrame (menghilangkan MultiIndex jika ada pada yfinance terbaru)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Memastikan kolom penting yang dibutuhkan fitur teknikal tersedia
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Kolom {col} tidak ditemukan dalam respon data API.")
            
        # Mengurutkan berdasarkan tanggal untuk kalkulasi runtun waktu (time-series)
        df.sort_index(inplace=True)
        jumlah_data_awal = len(df)
        print(f"Berhasil memuat {len(df)} baris data untuk {ticker}.")
            
        # Memastikan data bersih dan valid
        df = df.dropna(subset=['Close'])  # Hapus Close NaN
        df = df[df['Volume'] > 0]
                
        jumlah_data_akhir = len(df)
        if jumlah_data_awal != jumlah_data_akhir:
            print(f"Info Data Cleaning: Menghapus {jumlah_data_awal - jumlah_data_akhir} baris data (NaN atau Volume 0).")
            print(f"Berhasil memuat {len(df)} baris data bersih untuk {ticker}.")

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Tambahkan hari ini di baris paling bawah dengan fitur seluruhnya NaN
        today = pd.Timestamp('today', tz=df.index.tz).normalize()
        df.loc[today] = float('nan')
        
        return df
        
    except Exception as e:
        # Menyediakan error message yang jelas untuk keperluan logging produksi
        raise RuntimeError(f"Gagal memuat data pasar untuk {ticker}. Error: {str(e)}")
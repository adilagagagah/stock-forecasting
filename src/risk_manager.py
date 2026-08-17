import math
from typing import Dict, Any

def round_to_idx_tick(price: float) -> int:
    """
    Membulatkan harga ke fraksi (tick size) Bursa Efek Indonesia (IDX) terdekat.
    Aturan fraksi:
    < 200 : 1
    200 - 500 : 2
    500 - 2000 : 5
    2000 - 5000 : 10
    >= 5000 : 25
    """
    if price < 200:
        tick = 1
    elif price < 500:
        tick = 2
    elif price < 2000:
        tick = 5
    elif price < 5000:
        tick = 10
    else:
        tick = 25
        
    return int(round(price / tick) * tick)

def evaluate_trade_risk(
    total_capital: float,
    max_risk_percentage: float,
    entry_price: float,
    pred_trend_slope: float,
    pred_return: float,
    pred_risk: float,
    pred_days_to_max: float,
    pred_days_to_min: float,
    fee_buy: float = 0.0015,
    fee_sell: float = 0.0025,
    min_rr_ratio: float = 2.0,
    max_allocation_percentage: float = 0.25,
    is_uptrend: bool = False
) -> Dict[str, Any]:
    """
    Sistem Manajemen Risiko Kuantitatif Profesional.
    Mengevaluasi kelayakan transaksi, ukuran posisi (Position Sizing), dampak fee aplikasi,
    dan menyusun rencana aksi dinamis (waktu keluar atau tambah modal).

    Parameters:
    - total_capital (float): Total seluruh modal trading yang tersedia saat ini.
    - max_risk_percentage (float): Batas toleransi kerugian dari total modal per trade (contoh: 0.02 = 2%).
    - entry_price (float): Harga open/pembelian saham di pagi hari (T+0).
    - pred_trend_slope (float): Hasil prediksi kemiringan tren dari model.
    - pred_return (float): Hasil prediksi persentase kenaikan harga tertinggi (T+0 s.d T+4).
    - pred_risk (float): Hasil prediksi persentase penurunan harga terendah (T+0 s.d T+4), biasanya bernilai negatif.
    - pred_days_to_max (float): Estimasi hari ke-berapa target kenaikan maksimum akan tercapai.
    - pred_days_to_min (float): Estimasi hari ke-berapa penurunan terendah akan tercapai.
    - fee_buy (float): Persentase biaya beli aplikasi (default: 0.15% = 0.0015).
    - fee_sell (float): Persentase biaya jual aplikasi (default: 0.25% = 0.0025).
    - min_rr_ratio (float): Standar minimal Rasio Risk-to-Reward (default: 2.0).
    - max_allocation_percentage (float): Batas alokasi modal maksimal untuk satu saham (default: 25% = 0.25).
    - is_uptrend (bool): Status apakah pasar saat ini dalam kondisi uptrend yang kuat.
    """
    # 0. Sesuaikan toleransi risiko dan kemiringan tren berdasarkan rezim pasar
    if is_uptrend:
        min_trend_slope = 0.0
    else:
        min_trend_slope = 0.008

    # 1. Konversi prediksi persentase menjadi level harga nominal target
    raw_tp = entry_price * (1 + pred_return)
    raw_sl = entry_price * (1 + pred_risk)  # pred_risk negatif, misal 1 + (-0.03) = 0.97
    
    expected_tp_price = float(round_to_idx_tick(raw_tp))
    expected_sl_price = float(round_to_idx_tick(raw_sl))
    
    # 2. Hitung biaya riil transaksi (Cost Basis vs Net Exit) untuk akurasi net-profit
    cost_basis_per_share = entry_price * (1 + fee_buy)
    net_tp_per_share = expected_tp_price * (1 - fee_sell)
    net_sl_per_share = expected_sl_price * (1 - fee_sell)
    
    # 3. Hitung Reward dan Risk Aktual setelah dikurangi beban fee aplikasi
    actual_reward = net_tp_per_share - cost_basis_per_share
    actual_risk = cost_basis_per_share - net_sl_per_share
    
    # Antisipasi pembagian dengan nol atau risiko tidak wajar
    if actual_risk <= 0:
        actual_risk = 0.01 
        
    actual_rr_ratio = actual_reward / actual_risk
    
    # 4. Engine Ukuran Posisi (Position Sizing) - Best Practice Manajemen Risiko
    # Maksimal uang yang boleh hilang dalam satu transaksi berdasarkan batas toleransi kerugian
    max_money_to_lose = total_capital * max_risk_percentage
    
    # Jumlah lembar saham dan lot yang ideal dibeli (1 lot = 100 lembar saham)
    ideal_shares_to_buy = max_money_to_lose / actual_risk
    ideal_lots_to_buy = math.floor(ideal_shares_to_buy / 100)
    
    # Batasi alokasi dana agar tidak "All-In" pada satu saham demi diversifikasi portofolio
    max_capital_allocation = total_capital * max_allocation_percentage
    required_capital_for_ideal_lots = ideal_lots_to_buy * 100 * cost_basis_per_share
    
    if required_capital_for_ideal_lots > max_capital_allocation:
        final_lots_to_buy = math.floor(max_capital_allocation / (100 * cost_basis_per_share))
    else:
        final_lots_to_buy = ideal_lots_to_buy
        
    final_capital_spent = final_lots_to_buy * 100 * cost_basis_per_share
    
    # 5. Filter Keputusan Eksekusi Transaksi Berlapislah
    # Sinyal mentah (Debug Only): Hanya bergantung pada model murni (tanpa batasan lot/modal)
    raw_buy_signal = (pred_trend_slope >= min_trend_slope) and (actual_rr_ratio >= min_rr_ratio)
    
    # Transaksi dieksekusi HANYA JIKA sinyal mentah True DAN alokasi lot tersedia
    execute_trade = raw_buy_signal and (final_lots_to_buy > 0)
    
    # 6. Penyusunan Rencana Aksi Strategis berbasis Waktu (Days to Max / Min)
    action_plan = "TIDAK ADA AKSI (Sistem memblokir transaksi karena risiko buruk)."
    
    if execute_trade:
        # Logika Tambah Modal (Pyramiding) / Buy the Dip
        if pred_days_to_min < pred_days_to_max and abs(pred_risk) > 0:
            action_plan = (
                f"REKOMENDASI BELI AWAL: Masuk sebanyak {final_lots_to_buy} lot pada harga Rp{round(entry_price, 2)}. "
                f"STRATEGI TAMBAH MODAL (Pyramiding): Model memprediksi penurunan terendah akan terjadi pada "
                f"T+{int(round(pred_days_to_min))}. Jika harga melemah mendekati Rp{round(expected_sl_price, 2)} "
                f"pada hari tersebut namun bertahan di atas garis Cut Loss, Anda dapat menambah modal (Buy the Dip)."
            )
        else:
            action_plan = (
                f"REKOMENDASI BELI LANGSUNG: Masuk sebanyak {final_lots_to_buy} lot pada harga Rp{round(entry_price, 2)}. "
                f"Model memprediksi tren langsung menguat tanpa koreksi dalam."
            )
            
        # Tambahan aturan Time-Stop (Keluar berdasarkan durasi kadaluwarsa tren)
        action_plan += (
            f" STRATEGI KELUAR (Time-Stop): Batas durasi tren maksimum adalah T+{int(round(pred_days_to_max))}. "
            f"Jika pada hari tersebut harga belum menyentuh Target Profit atau Stop Loss, wajib laku-jual "
            f"di pasar secara manual karena jendela penguatan diprediksi telah habis."
        )

    return {
        "execute_trade": execute_trade,
        "raw_buy_signal": raw_buy_signal,
        "action_plan": action_plan,
        "entry_price_raw": round(entry_price, 2),
        "cost_basis_inc_fee": round(cost_basis_per_share, 2),
        "suggested_take_profit": round(expected_tp_price, 2),
        "suggested_stop_loss": round(expected_sl_price, 2),
        "actual_rr_ratio": round(actual_rr_ratio, 2),
        "allocated_lots": final_lots_to_buy,
        "capital_spent_idr": round(final_capital_spent, 2),
        "max_risk_idr": round(final_lots_to_buy * 100 * actual_risk, 2)
    }
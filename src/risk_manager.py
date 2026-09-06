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
    entry_price: float,
    pred_trend_slope: float,
    pred_return: float,
    pred_risk: float,
    pred_is_dip: float,
    min_dip_proba: float,
    min_rr_ratio: float,
    max_risk_percentage: float,
    max_allocation_percentage: float,
    fee_buy: float = 0.0015,
    fee_sell: float = 0.0025,
    regime_series: int = 2,
    rule_based_swing: bool = False,
    swing_sl_price: float = None,
    swing_tp_price: float = None
) -> Dict[str, Any]:
    """
    Sistem Manajemen Risiko Kuantitatif Profesional.
    Mengevaluasi kelayakan transaksi, ukuran posisi (Position Sizing), dampak fee aplikasi,
    dan menyusun rencana aksi dinamis (waktu keluar atau tambah modal).

    Parameters:
    - total_capital (float): Total seluruh modal trading yang tersedia saat ini.
    - entry_price (float): Harga open/pembelian saham di pagi hari (T+0).
    - pred_trend_slope (float): Hasil prediksi kemiringan tren dari model.
    - pred_return (float): Hasil prediksi persentase kenaikan harga tertinggi (T+0 s.d T+4).
    - pred_risk (float): Hasil prediksi persentase penurunan harga terendah (T+0 s.d T+4), biasanya bernilai negatif.
    - min_dip_proba (float): Hasil prediksi probabilitas Buy The Dip dari model.
    - min_rr_ratio (float): Standar minimal Rasio Risk-to-Reward.
    - max_risk_percentage (float): Batas toleransi kerugian dari total modal per trade (contoh: 0.02 = 2%).
    - max_allocation_percentage (float): Batas alokasi modal maksimal per entry.
    - fee_buy (float): Persentase biaya beli aplikasi (default: 0.15% = 0.0015).
    - fee_sell (float): Persentase biaya jual aplikasi (default: 0.25% = 0.0025).
    - regime_series (int): Status apakah pasar saat ini dalam kondisi uptrend, sideways, atau downtrend.
      1 = Strong Uptrend, 2 = Sideways, 3 = Downtrend, 4 = Bear Bounce.
    - rule_based_swing (bool): True jika Rule-Based Swing Detector mendeteksi kondisi reversal.
    - swing_sl_price (float): Level harga Stop Loss struktural (retest low).
    - swing_tp_price (float): Level harga Take Profit struktural.
    """
    # Penentuan batas minimum slope berdasarkan regime
    regime_mode = "normal"  # Track regime mode untuk logging
    if regime_series == 1:  # Uptrend
        min_trend_slope = 0.0
        min_dip_proba = max(0.1, min_dip_proba - 0.10)
        regime_mode = "uptrend"
    elif regime_series == 2:  # Sideways
        min_trend_slope = 0.008
        regime_mode = "sideways"
    else:  # Downtrend
        min_trend_slope = 0.015
        regime_mode = "downtrend"

    # Override sizing untuk Precision Swing Trade
    if rule_based_swing:
        regime_mode = "bear_swing"
        max_allocation_percentage = min(max_allocation_percentage, 0.30)  # Cap 30% alokasi
        max_risk_percentage = min(max_risk_percentage, 0.03)  # Cap 3% risiko per trade

    # 1. Konversi prediksi persentase menjadi level harga nominal target
    # Jika ada level SL/TP struktural dari rule-based swing, prioritaskan itu
    if rule_based_swing and swing_sl_price is not None and swing_sl_price > 0:
        raw_sl = swing_sl_price
        raw_tp = swing_tp_price if (swing_tp_price is not None and swing_tp_price > 0) else entry_price * 1.12
    else:
        raw_tp = entry_price * (1 + pred_return)
        raw_sl = entry_price * (1 + pred_risk)  # pred_risk negatif, misal 1 + (-0.03) = 0.97
    
    expected_tp_price = float(round_to_idx_tick(raw_tp))
    expected_sl_price = float(round_to_idx_tick(raw_sl))
    
    # 2. Hitung biaya riil transaksi (Cost Basis vs Net Exit) untuk akurasi net-profit
    cost_buy_per_share = entry_price * (1 + fee_buy)
    net_tp_per_share = expected_tp_price * (1 - fee_sell)
    net_sl_per_share = expected_sl_price * (1 - fee_sell)
    
    # 3. Hitung Reward dan Risk Aktual setelah dikurangi beban fee aplikasi
    actual_reward = net_tp_per_share - cost_buy_per_share
    actual_risk = cost_buy_per_share - net_sl_per_share
    if actual_risk <= 0:
        # Reject trade: stop >= entry merupakan kondisi invalid (audit Section XI.C)
        return {
            "execute_trade": False,
            "raw_buy_signal": False,
            "entry_price": round(entry_price, 2),
            "entry_price_inc_fee": round(cost_buy_per_share, 2),
            "suggested_take_profit": round(expected_tp_price, 2),
            "suggested_stop_loss": round(expected_sl_price, 2),
            "actual_rr_ratio": 0.0,
            "allocated_lots": 0,
            "capital_spent_idr": 0.0,
            "regime_mode": regime_mode,
            "reject_reason": "INVALID_RISK_DISTANCE"
        }
    actual_rr_ratio = actual_reward / actual_risk
    
    # 4. Engine Ukuran Posisi (Position Sizing) - Best Practice Manajemen Risiko
    # Maksimal uang yang boleh hilang dalam satu transaksi berdasarkan batas toleransi kerugian
    max_money_to_lose = total_capital * max_risk_percentage
    ideal_shares_to_buy = max_money_to_lose / actual_risk
    ideal_lots_to_buy = math.floor(ideal_shares_to_buy / 100)
    
    # Batasi alokasi dana agar tidak "All-In" pada satu kali trade
    max_capital_allocation = total_capital * max_allocation_percentage
    required_capital_for_ideal_lots = ideal_lots_to_buy * 100 * cost_buy_per_share
    
    if required_capital_for_ideal_lots > max_capital_allocation:
        final_lots_to_buy = math.floor(max_capital_allocation / (100 * cost_buy_per_share))
    else:
        final_lots_to_buy = ideal_lots_to_buy
        
    final_capital_spent = final_lots_to_buy * 100 * cost_buy_per_share
    
    # 5. Filter Keputusan Eksekusi Transaksi Berlapis
    # Sinyal mentah (Debug Only): Hanya bergantung pada model murni (tanpa batasan lot/modal)
    raw_buy_signal = (pred_is_dip >= min_dip_proba) or (pred_trend_slope >= min_trend_slope) or (actual_rr_ratio >= min_rr_ratio)
    
    # Jalur sinyal utama ML-based
    ml_buy_signal = (pred_is_dip >= min_dip_proba) or ((pred_trend_slope >= min_trend_slope) and (actual_rr_ratio >= min_rr_ratio))
    
    # Jalur sinyal Rule-Based Swing Detector (lapisan kedua)
    # Aktif sebagai fallback presisi jika ML tidak menangkap posisi bagus di bear market
    # ML harus konfirmasi minimal +5% MFE upside (Sec. XII: layered decision, jangan bypass risk framework)
    swing_buy_signal = bool(rule_based_swing and (actual_risk > 0) and (pred_return >= 0.05))
    
    # Gabungan: ML sinyal OR rule-based swing
    act_buy_signal = ml_buy_signal or swing_buy_signal
    
    # Override sizing untuk swing trade: cap allocation 30%
    if swing_buy_signal and not ml_buy_signal:
        max_capital_allocation_swing = total_capital * 0.30
        if final_capital_spent > max_capital_allocation_swing:
            final_lots_to_buy = math.floor(max_capital_allocation_swing / (100 * cost_buy_per_share))
            final_capital_spent = final_lots_to_buy * 100 * cost_buy_per_share
    
    execute_trade = act_buy_signal and (final_lots_to_buy > 0)
  
    return {
        "execute_trade": execute_trade,
        "raw_buy_signal": act_buy_signal,
        "entry_price": round(entry_price, 2),
        "entry_price_inc_fee": round(cost_buy_per_share, 2),
        "suggested_take_profit": round(expected_tp_price, 2),
        "suggested_stop_loss": round(expected_sl_price, 2),
        "actual_rr_ratio": round(actual_rr_ratio, 2),
        "allocated_lots": final_lots_to_buy,
        "capital_spent_idr": round(final_capital_spent, 2),
        "regime_mode": regime_mode,
        "swing_triggered": swing_buy_signal and not ml_buy_signal
    }



# df_bumi = calculate_atr(df_bumi)
# latest_data = df_bumi.iloc[-1]
# current_close = float(latest_data['Close'])
# current_atr = float(latest_data['ATR'])

# resisten_terdekat = 186.0  

# trade_analysis = evaluate_trade_risk(
#     entry_price=current_close,
#     atr=current_atr,
#     resistance_level=resisten_terdekat,
#     min_rr_ratio=2.0
# )

# # Cetak Hasil Validasi Hitam-di-Atas-Putih
# print("=== LIVE SYSTEM RISK EVALUATION ===")
# print(f"Harga Masuk Terkini     : Rp{trade_analysis['entry_price']}")
# print(f"Nilai Volatilitas (ATR) : Rp{round(current_atr, 2)}")
# print(f"Garis Aman Cut Loss (SL): Rp{trade_analysis['suggested_stop_loss']} (Terproteksi dari noise)")
# print(f"Target Resisten (TP)    : Rp{trade_analysis['target_profit_resistance']}")
# print(f"Rasio R-to-R Aktual     : {trade_analysis['actual_rr_ratio']}x")
# print("-----------------------------------")

# if trade_analysis['execute_trade']:
#     print("KEPUTUSAN SISTEM: ISI ORDER BELI! Matematika risiko mendukung penuh.")
# else:
#     print("KEPUTUSAN SISTEM: BLOKIR TRANSAKSI! Jarak ke resisten terlalu sempit, batalkan emosi Anda.")
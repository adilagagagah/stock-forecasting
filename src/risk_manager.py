from typing import Dict, Any

def evaluate_trade_risk(
    entry_price: float, 
    atr: float, 
    resistance_level: float, 
    min_rr_ratio: float = 2.0
) -> Dict[str, Any]:
    """
    Mengevaluasi rasio Risk-to-Reward secara objektif sebelum posisi diambil.
    Sistem wajib memblokir perdagangan jika target profit tidak realistis (RR < 2).
    """
    # 1. Amankan batas risiko: Jarak Stop Loss = 1.5 * Volatilitas Pasar (ATR)
    atr_distance = 1.5 * atr
    stop_loss = entry_price - atr_distance
    risk = entry_price - stop_loss
    
    # 2. Ukur potensi profit berdasarkan resisten teknikal terdekat yang nyata
    potential_reward = resistance_level - entry_price
    
    # 3. Hitung Rasio Risk-to-Reward (RR) Aktual
    actual_rr_ratio = potential_reward / risk if risk > 0 else 0
    
    # 4. Filter Keputusan: Eksekusi HANYA JIKA rasio memenuhi standar minimal
    execute_decision = actual_rr_ratio >= min_rr_ratio
    
    return {
        "entry_price": round(entry_price, 2),
        "suggested_stop_loss": round(stop_loss, 2),
        "target_profit_resistance": round(resistance_level, 2),
        "risk_per_share": round(risk, 2),
        "reward_per_share": round(potential_reward, 2),
        "actual_rr_ratio": round(actual_rr_ratio, 2),
        "execute_trade": execute_decision
    }
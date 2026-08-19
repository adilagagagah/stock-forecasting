"""
Deep Analysis Script: Buy the Dip Detector
==========================================
Jalankan fungsi-fungsi ini dari modelling_classifier.ipynb.
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif

from src.data_loader import load_stock_data
from src.features import create_features
from src.label import create_labels, buy_the_dip_label
from src.model_config import FEATURES
from xgboost import XGBClassifier

plt.style.use('seaborn-v0_8-darkgrid')


def get_base_xgb(y):
    """ Helper: Standardize XGBoost parameters with dynamic scale_pos_weight """
    n_pos = int(np.sum(y))
    spw = max(1.0, (len(y) - n_pos) / (n_pos + 1e-9))
    return XGBClassifier(
        n_estimators=100, 
        max_depth=3,            # [Reg] Kurangi kedalaman pohon (default 6, sebelumnya 5)
        learning_rate=0.01,     # [Reg] Turunkan learning rate untuk pembelajaran perlahan
        subsample=0.7,          # [Reg] Gunakan hanya 70% data acak per pohon
        colsample_bytree=0.7,   # [Reg] Gunakan hanya 70% fitur acak per pohon
        min_child_weight=5,     # [Reg] Cegah pemotongan node yang datanya terlalu sedikit (noise)
        gamma=2.0,              # [Reg] Syarat minimum loss reduction untuk membuat cabang baru
        reg_alpha=0.5,          # [Reg] L1 Regularization (penalti bobot absolut)
        reg_lambda=1.5,         # [Reg] L2 Regularization (penalti bobot kuadrat)
        random_state=42, 
        verbosity=0, 
        eval_metric='logloss',
        scale_pos_weight=spw
    )

def get_xgb_pipeline(y):
    """ Helper: Standardize Pipeline creation """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', get_base_xgb(y))
    ])


def analyze_label_thresholds(df_full, feature_cols=FEATURES, end_date='2023-12-31'):
    """
    Menampilkan distribusi label positif untuk berbagai kombinasi
    min_return dan max_drawdown. Target ideal: 5-20% positif.
    """
    print("\n" + "=" * 70)
    print("  ANALISIS THRESHOLD LABEL — Mencari 'Definisi Dip' yang Tepat")
    print("=" * 70)
    print(f"  {'min_return':>10} | {'max_drawdown':>12} | {'% Positif':>10} | {'Jumlah':>7} | {'Status':>20}")
    print("  " + "-" * 65)

    results = []
    for min_ret in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.15, 0.17, 0.20]:
        for max_dd in [-0.02, -0.03, -0.05]:
            df_tmp = buy_the_dip_label(df_full.copy(), min_return=min_ret, max_drawdown=max_dd)
            if end_date:
                df_tmp = df_tmp.loc[:end_date]
            n_pos = int(df_tmp['is_buy_dip'].sum())
            n_total = int(df_tmp['is_buy_dip'].count())
            pct = n_pos / n_total * 100

            if pct < 8:
                status = "IDEAL (Langka)"
            elif pct < 20:
                status = "Cukup"
            elif pct < 35:
                status = "Terlalu Sering"
            else:
                status = "Tidak Ada Edge"

            results.append({'min_return': min_ret, 'max_drawdown': max_dd, 'pct_positive': round(pct, 1), 'count': n_pos, 'status': status})
            print(f"  {min_ret:>10.0%} | {max_dd:>12.0%} | {pct:>10.1f}% | {n_pos:>7} | {status}")

    print()
    return pd.DataFrame(results)


def analyze_feature_importance(df_full, target_col='is_buy_dip', feature_cols=FEATURES, end_date='2023-12-31'):
    """
    Dua pendekatan feature importance:
    1. Mutual Information (statistik non-linear, tidak bergantung model)
    2. XGBoost Feature Importance (Gain-based)
    """
    print("\n" + "=" * 70)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    df_tmp = df_full.copy()
    if end_date:
        df_tmp = df_tmp.loc[:end_date]
    df_clean = df_tmp.dropna(subset=feature_cols + [target_col])
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values.astype(int)

    # 1. Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': feature_cols, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)

    print("\n  [Mutual Information Scores]")
    for _, row in mi_df.iterrows():
        bar = "█" * int(row['MI_Score'] * 200)
        print(f"  {row['Feature']:>25} | {row['MI_Score']:.4f} | {bar}")

    # 2. XGBoost Feature Importance
    X_sc = StandardScaler().fit_transform(X)
    xgb = get_base_xgb(y)
    xgb.fit(X_sc, y)
    gain_df = pd.DataFrame({'Feature': feature_cols, 'XGB_Gain': xgb.feature_importances_}).sort_values('XGB_Gain', ascending=False)

    print("\n  [XGBoost Feature Gain]")
    for _, row in gain_df.iterrows():
        bar = "█" * int(row['XGB_Gain'] * 100)
        print(f"  {row['Feature']:>25} | {row['XGB_Gain']:.4f} | {bar}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Feature Importance — Buy the Dip Detector", fontsize=14, fontweight='bold')
    axes[0].barh(mi_df['Feature'], mi_df['MI_Score'], color='steelblue')
    axes[0].set_title("Mutual Information")
    axes[0].invert_yaxis()
    axes[1].barh(gain_df['Feature'], gain_df['XGB_Gain'], color='darkorange')
    axes[1].set_title("XGBoost Feature Gain")
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.show()

    top_mi = set(mi_df.head(len(feature_cols) // 2)['Feature'])
    top_xgb = set(gain_df.head(len(feature_cols) // 2)['Feature'])
    strong_features = sorted(top_mi & top_xgb)
    weak_features = sorted(set(feature_cols) - (top_mi | top_xgb))

    print(f"\n  FITUR KUAT (Top-50% di kedua metode): {strong_features}")
    print(f"  FITUR LEMAH (Bottom-50% di kedua metode): {weak_features}")

    return mi_df, gain_df, strong_features


def robustness_test(df_full, target_col='is_buy_dip', feature_cols=FEATURES, n_splits=5, n_permutations=50, end_date='2023-12-31'):
    """
    3 Tes Robustness:
    A. Cross-Period Stability: Apakah Val F1 stabil di setiap fold time series?
    B. Permutation Test: Apakah model mengalahkan label acak?
    C. Feature Subset Test: Apakah hanya memakai fitur kuat lebih baik?
    """
    print("\n" + "=" * 70)
    print("  ROBUSTNESS TEST")
    print("=" * 70)

    df_tmp = df_full.copy()
    if end_date:
        df_tmp = df_tmp.loc[:end_date]
    df_clean = df_tmp.dropna(subset=feature_cols + [target_col])
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values.astype(int)

    pipeline = get_xgb_pipeline(y)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # A. Cross-Period Stability
    print(f"\n  [A] Cross-Period Stability (F1 per fold):")
    real_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='f1', n_jobs=-1)
    
    for i, f1 in enumerate(real_scores):
        print(f"    Fold {i+1}: F1={f1:.3f}")

    mean_f1 = real_scores.mean()
    std_f1 = real_scores.std()
    cv_coef = std_f1 / (mean_f1 + 1e-9)
    print(f"\n    Mean F1: {mean_f1:.3f} ± {std_f1:.3f} | CV={cv_coef:.2f}")
    if cv_coef < 0.25:
        print("    => STABIL — Edge konsisten lintas periode")
    elif cv_coef < 0.5:
        print("    => KURANG STABIL — Ada periode di mana model kehilangan edge")
    else:
        print("    => TIDAK STABIL — Model tidak punya edge yang nyata, hanya noise")

    # B. Permutation Test
    print(f"\n  [B] Permutation Test ({n_permutations} iterasi — apakah model > random?):")
    permuted_scores = []
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        y_shuf = rng.permutation(y)
        sc = cross_val_score(pipeline, X, y_shuf, cv=tscv, scoring='f1', n_jobs=-1)
        permuted_scores.append(sc.mean())

    p_value = np.mean(np.array(permuted_scores) >= mean_f1)
    pct_beaten = p_value * 100
    print(f"    Real F1         : {mean_f1:.3f}")
    print(f"    Random F1 (mean): {np.mean(permuted_scores):.3f} ± {np.std(permuted_scores):.3f}")
    print(f"    p-value         : {p_value:.3f} (Random yg mengalahkan Real: {pct_beaten:.1f}%)")
    if p_value < 0.05:
        print("    => SIGNIFIKAN — Model punya edge nyata (p < 0.05)")
    elif p_value < 0.15:
        print("    => LEMAH — Sedikit di atas keberuntungan")
    else:
        print("    => TIDAK SIGNIFIKAN — Tidak lebih baik dari tebakan acak!")

    # C. Feature Subset Test
    print(f"\n  [C] Feature Subset Test:")
    _, _, strong_feats = analyze_feature_importance(df_full, target_col, feature_cols, end_date=end_date)
    if strong_feats and len(strong_feats) < len(feature_cols):
        df_c2 = df_tmp.dropna(subset=strong_feats + [target_col])
        X2 = df_c2[strong_feats].values
        y2 = df_c2[target_col].values.astype(int)
        
        pl2 = get_xgb_pipeline(y2)
        sc2 = cross_val_score(pl2, X2, y2, cv=tscv, scoring='f1', n_jobs=-1)
        print(f"    Full ({len(feature_cols)} fitur): F1={mean_f1:.3f}")
        print(f"    Strong only ({len(strong_feats)} fitur): F1={sc2.mean():.3f} ± {sc2.std():.3f}")
        if sc2.mean() > mean_f1 + 0.01:
            print("    => Membuang fitur lemah MENINGKATKAN performa! Gunakan daftar strong_feats.")
        else:
            print("    => Tidak ada perbedaan signifikan. Fitur lemah tidak banyak merugikan.")

    return mean_f1, std_f1


def threshold_sensitivity_analysis(df_full, feature_cols=FEATURES, end_date='2023-12-31'):
    """
    Sweep kombinasi threshold label untuk menemukan definisi 'Buy the Dip'
    yang paling mudah dipelajari model (F1 tertinggi).
    """
    print("\n" + "=" * 70)
    print("  THRESHOLD SENSITIVITY — Mana Threshold yang Paling Bisa Dipelajari?")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for min_ret in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.15, 0.17, 0.20]:
        for max_dd in [-0.02, -0.03, -0.05]:
            df_tmp = buy_the_dip_label(df_full.copy(), min_return=min_ret, max_drawdown=max_dd)
            if end_date:
                df_tmp = df_tmp.loc[:end_date]
            df_cl = df_tmp.dropna(subset=feature_cols + ['is_buy_dip'])
            X = df_cl[feature_cols].values
            y = df_cl['is_buy_dip'].values.astype(int)

            if y.sum() < 20:
                continue

            pl = get_xgb_pipeline(y)

            try:
                scores = cross_val_score(pl, X, y, cv=tscv, scoring='f1', n_jobs=-1)
                pct_pos = y.mean() * 100
                print(f"  Return≥{min_ret:.0%}, DD≥{max_dd:.0%}: Positif={pct_pos:.1f}% | F1={scores.mean():.3f} ± {scores.std():.3f}")
                results.append({'min_return': min_ret, 'max_drawdown': max_dd,
                                 'pct_positive': round(pct_pos, 1),
                                 'mean_f1': round(scores.mean(), 3),
                                 'std_f1': round(scores.std(), 3)})
            except Exception as e:
                print(f"  Return≥{min_ret:.0%}, DD≥{max_dd:.0%}: ERROR — {e}")

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        best = df_res.loc[df_res['mean_f1'].idxmax()]
        print(f"\n  TERBAIK: min_return={best['min_return']:.0%}, max_drawdown={best['max_drawdown']:.0%}")
        print(f"  Positif={best['pct_positive']}% | F1={best['mean_f1']:.3f} ± {best['std_f1']:.3f}")
    return df_res


def find_sweet_spot_threshold(df_full, feature_cols=FEATURES, n_permutations=50, end_date='2023-12-31'):
    """
    Cari threshold label yang memenuhi DUA syarat sekaligus:
    1. F1 cukup (> 0.10) - model bisa belajar
    2. Signifikan secara statistik (permutation p < 0.10)

    Hasilnya diurutkan berdasarkan 'Combined Score' = F1 * (1 - p_value).
    Ini lebih akurat daripada mencari F1 tertinggi atau p-value terkecil sendiri-sendiri.
    """
    print("\n" + "=" * 70)
    print("  SWEET SPOT FINDER — F1 Cukup + Statistik Signifikan")
    print("=" * 70)
    print("  (Mungkin memakan waktu 5-15 menit...)\n")

    tscv = TimeSeriesSplit(n_splits=5)
    rng = np.random.default_rng(42)
    results = []

    candidates = [
        (0.03, -0.02), (0.03, -0.03), (0.03, -0.05),
        (0.05, -0.02), (0.05, -0.03), (0.05, -0.05),
        (0.06, -0.02), (0.06, -0.03), (0.06, -0.05),
        (0.07, -0.02), (0.07, -0.03), (0.07, -0.05),
        (0.08, -0.02), (0.08, -0.03), (0.08, -0.05),
        (0.10, -0.02), (0.10, -0.03), (0.10, -0.05),
        (0.12, -0.02), (0.12, -0.03), (0.12, -0.05),
        (0.14, -0.02), (0.14, -0.03), (0.14, -0.05),
        (0.15, -0.02), (0.15, -0.03), (0.15, -0.05),
        (0.17, -0.02), (0.17, -0.03), (0.17, -0.05),
    ]

    for min_ret, max_dd in candidates:
        df_tmp = buy_the_dip_label(df_full.copy(), min_return=min_ret, max_drawdown=max_dd)
        if end_date:
            df_tmp = df_tmp.loc[:end_date]
        df_cl = df_tmp.dropna(subset=feature_cols + ['is_buy_dip'])
        X = df_cl[feature_cols].values
        y = df_cl['is_buy_dip'].values.astype(int)

        n_pos = y.sum()
        pct_pos = y.mean() * 100

        if n_pos < 15:
            print(f"  {min_ret:.0%}/{max_dd:.0%}: SKIP (hanya {n_pos} positif, tidak cukup)")
            continue

        pl = get_xgb_pipeline(y)

        real_scores = cross_val_score(pl, X, y, cv=tscv, scoring='f1', n_jobs=-1)
        real_f1 = real_scores.mean()

        perm_scores = []
        for _ in range(n_permutations):
            y_shuf = rng.permutation(y)
            sc = cross_val_score(pl, X, y_shuf, cv=tscv, scoring='f1', n_jobs=-1)
            perm_scores.append(sc.mean())

        p_value = np.mean(np.array(perm_scores) >= real_f1)
        cv_coef = real_scores.std() / (real_f1 + 1e-9)

        # ----------------------------------------------------------------
        # SCORING FORMULA: 3 Komponen (masing-masing harus terpenuhi)
        # ----------------------------------------------------------------
        # 1. Significance Score: lebih rendah p_value lebih baik
        sig_score = max(0.0, 1.0 - p_value / 0.15)  # score=1 jika p=0, score=0 jika p>=0.15

        # 2. Frequency Score: puncak di 10-12%, drop ke 0 jika <5% atau >25%
        #    Gaussian centered di 11%, std=6%
        freq_score = np.exp(-0.5 * ((pct_pos - 11.0) / 6.0) ** 2)

        # 3. Stability Score: 1 jika CV=0, turun ke 0 jika CV>=1.0
        stab_score = max(0.0, 1.0 - cv_coef)

        # Final score: harus bagus di KETIGA komponen (perkalian)
        combined_score = real_f1 * sig_score * freq_score * stab_score

        # Filter keras — langsung disqualify jika salah satu gagal
        disqualified = []
        if pct_pos < 5 or pct_pos > 25:
            disqualified.append(f"Freq={pct_pos:.1f}%")
        if p_value >= 0.20:
            disqualified.append(f"p={p_value:.2f}")
        if cv_coef >= 1.2:
            disqualified.append(f"CV={cv_coef:.2f}")

        sig_tag  = "Sig✅" if p_value < 0.10 else ("Sig⚠️" if p_value < 0.20 else "Sig❌")
        f1_tag   = "F1✅"  if real_f1 > 0.10  else "F1❌"
        freq_tag = "Freq✅" if 5 <= pct_pos <= 20 else "Freq❌"
        stab_tag = "Stbl✅" if cv_coef < 0.5 else ("Stbl⚠️" if cv_coef < 1.0 else "Stbl❌")
        dq_str   = f" [DISQUALIFIED: {', '.join(disqualified)}]" if disqualified else ""

        print(f"  {min_ret:.0%}/{max_dd:.0%}: Pos={pct_pos:.1f}% | F1={real_f1:.3f}±{real_scores.std():.3f} | p={p_value:.2f} | CV={cv_coef:.2f} | Score={combined_score:.4f} | {sig_tag} {f1_tag} {freq_tag} {stab_tag}{dq_str}")

        results.append({
            'min_return': min_ret,
            'max_drawdown': max_dd,
            'pct_positive': round(pct_pos, 1),
            'mean_f1': round(real_f1, 4),
            'std_f1': round(real_scores.std(), 4),
            'p_value': round(p_value, 3),
            'cv_coef': round(cv_coef, 3),
            'sig_score': round(sig_score, 4),
            'freq_score': round(freq_score, 4),
            'stab_score': round(stab_score, 4),
            'combined_score': round(combined_score, 4),
            'n_positive': int(n_pos),
            'disqualified': bool(disqualified)
        })

    df_res = pd.DataFrame(results)

    if not df_res.empty:
        df_res = df_res.sort_values('combined_score', ascending=False)
        best = df_res.iloc[0]
        print(f"\n  SWEET SPOT TERBAIK:")
        print(f"     min_return={best['min_return']:.0%} | max_drawdown={best['max_drawdown']:.0%}")
        print(f"     Positif={best['pct_positive']}% | F1={best['mean_f1']:.3f}±{best['std_f1']:.3f}")
        print(f"     p-value={best['p_value']:.3f} | CV={best['cv_coef']:.3f} | Combined Score={best['combined_score']:.4f}")
        print(f"\n  => Gunakan threshold ini di buy_the_dip_label(min_return={best['min_return']}, max_drawdown={best['max_drawdown']})")

    return df_res

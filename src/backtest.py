import sys
import os
import pandas as pd
import joblib
import math

sys.path.append(os.path.abspath(os.path.join('..')))
from src.risk_manager import evaluate_trade_risk

class WalkForwardBacktester:
    def __init__(
        self, 
        ticker: str, 
        min_dip_proba: float,
        min_rr_ratio: float,
        max_risk_pct: float,
        max_alloc_pct: float,
        max_holding_days: int = 5,
        fee_buy: float = 0.0015,
        fee_sell: float = 0.0025,
        initial_capital: float = 5_000_000.0,
        ma_type: str = 'ema',
        enable_pyramiding: bool = True,
        max_pyramid_adds: int = 2,
        pyramid_profit_threshold: float = 0.05
    ):
        """
        Mesin Backtest Walk-Forward Analysis (WFA) berbasis Expanding Window.
        
        Parameters:
        - ticker : Kode emiten saham (misal: 'BUMI.JK')
        - min_dip_proba : Probabilitas minimal untuk beli
        - max_risk_pct : Toleransi risiko per transaksi
        - max_alloc_pct : Batas alokasi dana per posisi
        - min_rr_ratio : Minimal rasio Risk-to-Reward aktual net-fee
        - max_holding_days : Maksimum hari menahan posisi (default: 5)
        - fee_buy / fee_sell : Biaya aplikasi beli & jual (default: 0.15% & 0.25%)
        - initial_capital : Modal awal simulasi (default: Rp 5.000.000)
        - ma_type : Jenis MA untuk trailing exit saat Strong Uptrend ('ema' atau 'sma', default: 'ema')
        - enable_pyramiding : Mengaktifkan penambahan posisi saat tren terkonfirmasi (default: True)
        - max_pyramid_adds : Maksimum penambahan posisi per transaksi aktif (default: 2)
        - pyramid_profit_threshold : Keuntungan minimum sebelum boleh pyramiding (default: +5%)
        """
        self.ticker = ticker
        self.min_dip_proba = min_dip_proba
        self.min_rr_ratio = min_rr_ratio
        self.max_risk_pct = max_risk_pct
        self.max_alloc_pct = max_alloc_pct
        self.max_holding_days = max_holding_days
        self.fee_buy = fee_buy
        self.fee_sell = fee_sell
        self.initial_capital = initial_capital
        self.ma_type = ma_type.lower()
        self.enable_pyramiding = enable_pyramiding
        self.max_pyramid_adds = max_pyramid_adds
        self.pyramid_profit_threshold = pyramid_profit_threshold
        self.ma5_series = None
        self.ma10_series = None

        # Struktur penampung portofolio & riwayat
        self.active_trades = []      # Daftar posisi aktif (T+0 s.d T+4)
        self.trade_history = []      # Log eksekusi jual (Closed Trades)
        self.equity_curve = []       # Record harian perubahan total modal

        # 5 Target spesifik sesuai blueprint sistem
        self.targets = ['is_dip', 'return', 'risk', 'trend_slope']
        self.trained_pipelines = {}  # Menampung model aktif hasil retraining
        self.features_used = {}      # Menampung fitur asli yang digunakan saat pre-training
        self.daily_predictions = []  # Menampung hasil prediksi harian
        self.predictions_df = None   # DataFrame hasil prediksi
        self.rsi_series = None       # RSI 14 untuk deteksi oversold (Bear Bounce)
        self.macd_hist_series = None # MACD Histogram untuk konfirmasi momentum pembalikan arah
        self.ma20_series = None      # EMA 20
        self.ma50_series = None      # EMA 50
        self.last_swing_exit_date = None  # Cooldown tracker: tanggal terakhir keluar dari swing trade

    def run_backtest(
        self, 
        X_in: pd.DataFrame, 
        y_in_dict: dict, 
        X_oos: pd.DataFrame, 
        y_oos_dict: dict, 
        df_raw_prices: pd.DataFrame,
        feature_cols: list
    ):
        """
        Eksekutor utama siklus Walk-Forward Analysis (WFA) pada periode Out-of-Sample.
        """
        # Reset state agar bisa dijalankan berulang kali dengan instance yang sama
        self.current_capital = self.initial_capital
        self.active_trades = []
        self.trade_history = []
        self.equity_curve = []
        self.daily_predictions = []
        self.last_swing_exit_date = None

        print("=" * 70)
        print(f" MEMULAI SIMULASI WALK-FORWARD ANALYSIS ({self.ticker})")
        print(f" Modal Awal: Rp{self.initial_capital:,.2f}")
        print("=" * 70)

        # A. Load model awal dari folder models/
        ticker_name = self.ticker.split('.')[0].lower()
        print(f"\n[INIT] Memuat model awal dari folder models/{ticker_name}...")
        for target in self.targets:
            model_path = os.path.join('..', 'models', ticker_name, f"model_{target}_{ticker_name}.pkl")
            if not os.path.exists(model_path):
                 model_path = os.path.join('models', ticker_name, f"model_{target}_{ticker_name}.pkl")
            
            try:
                payload = joblib.load(model_path)
                self.trained_pipelines[target] = payload['pipeline']
                self.features_used[target] = payload['metadata']['features_used']
            except FileNotFoundError:
                print(f"[ERROR] File model '{target}' tidak ditemukan di: {model_path}")
                raise

        # B. Penentuan Regime Trend & Moving Average Exit
        # Hitung MA 5 & 10 untuk trailing exit dinamis saat Strong Uptrend
        if self.ma_type == 'sma':
            self.ma5_series = df_raw_prices['Close'].rolling(window=5).mean()
            self.ma10_series = df_raw_prices['Close'].rolling(window=10).mean()
        else:
            self.ma5_series = df_raw_prices['Close'].ewm(span=5, adjust=False).mean()
            self.ma10_series = df_raw_prices['Close'].ewm(span=10, adjust=False).mean()

        # Hitung indikator EMA untuk penentuan regime tren
        ema5 = df_raw_prices['Close'].ewm(span=5, adjust=False).mean()
        ema10 = df_raw_prices['Close'].ewm(span=10, adjust=False).mean()
        self.ma20_series = df_raw_prices['Close'].ewm(span=20, adjust=False).mean()
        self.ma50_series = df_raw_prices['Close'].ewm(span=50, adjust=False).mean()
        ema20 = self.ma20_series
        ema50 = self.ma50_series

        # Daily trend condition based on EMA arrangement (Bullish vs Bearish alignment)
        daily_uptrend = (ema5 > ema10) & (ema10 > ema20) & (ema20 > ema50)
        daily_downtrend = (ema5 < ema10) & (ema10 < ema20) & (ema20 < ema50)

        # Evaluate condition for holding for 5 consecutive days (rolling window = 5)
        is_uptrend_5d = (daily_uptrend.rolling(window=5).sum() == 5)
        is_downtrend_5d = (daily_downtrend.rolling(window=5).sum() == 5)

        regime_series = pd.Series(2, index=df_raw_prices.index)  # Default: 2 (Sideways)
        regime_series[is_uptrend_5d] = 1   # 1: Uptrend jika 5 hari berturut-turut naik/bullish
        regime_series[is_downtrend_5d] = 3 # 3: Downtrend jika 5 hari berturut-turut turun/bearish

        # Hitung MACD Histogram untuk konfirmasi momentum pembalikan arah
        ema12 = df_raw_prices['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_raw_prices['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        self.macd_hist_series = macd - macd_signal

        # Hitung RSI 14 untuk deteksi oversold & divergence
        delta = df_raw_prices['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        self.rsi_series = 100 - (100 / (1 + rs))

        # Geser mundur satu hari agar kondisi diketahui saat open market hari T (mencegah look-ahead bias)
        regime_series = regime_series.shift(1).fillna(2).astype(int)

        # Inisialisasi Expanding Window training data (mengakumulasi data seiring waktu)
        X_train_current = X_in.copy()
        if isinstance(y_in_dict, dict):
            y_train_current = {k: v.copy() for k, v in y_in_dict.items()}
        else:
            y_train_current = y_in_dict.copy()

        # C. TRADING LOOP (Iterasi harian sepanjang bulan berjalan)
        # Grouping data Out-of-Sample (2024-2025) berdasarkan Tahun dan Bulan
        oos_period_months = X_oos.groupby([X_oos.index.year, X_oos.index.month])

        for (year, month), month_features in oos_period_months:
            print("-" * 50)
            print(f"PERIODE KELOLAAN: {year}-{month:02d} | OOS: {len(month_features)} Hari Bursa")
            print("-" * 50)

            # Daily trading loop
            for current_date, daily_row in month_features.iterrows():

                # C0. Melakukan Prediksi 4 Target
                preds = {}
                for target in self.targets:
                    target_features = self.features_used[target]
                    feature_vector = daily_row[target_features].values.reshape(1, -1)
                    if target == 'is_dip':
                        preds[target] = self.trained_pipelines[target].predict_proba(feature_vector)[0][1]
                    else:
                        preds[target] = self.trained_pipelines[target].predict(feature_vector)[0]

                # C1. RISK MANAGER EVALUATION: Hitung kelayakan transaksi & position sizing
                entry_price_today = df_raw_prices.loc[current_date, 'Open']
                current_regime = regime_series.loc[current_date] if current_date in regime_series.index else 2
                
                # Rule-Based Swing Detector (Lapisan Kedua: Presisi Timing Entry)
                swing_info = self._detect_rule_based_swing(current_date, df_raw_prices, current_regime)
                
                # Cooldown: blokir swing entry selama 5 hari kerja setelah swing trade terakhir keluar
                if swing_info['is_swing'] and self.last_swing_exit_date is not None:
                    days_since_swing_exit = len(df_raw_prices.loc[self.last_swing_exit_date:current_date]) - 1
                    if days_since_swing_exit < 5:
                        swing_info = {"is_swing": False, "sl_price": None, "tp_price": None}
                
                risk_eval = evaluate_trade_risk(
                    total_capital=self.current_capital,
                    entry_price=entry_price_today,
                    pred_trend_slope=preds['trend_slope'],
                    pred_return=preds['return'],
                    pred_risk=preds['risk'],
                    pred_is_dip=preds['is_dip'],
                    min_dip_proba=self.min_dip_proba,
                    min_rr_ratio=self.min_rr_ratio,
                    max_risk_percentage=self.max_risk_pct,
                    max_allocation_percentage=self.max_alloc_pct,
                    fee_buy=self.fee_buy,
                    fee_sell=self.fee_sell,
                    regime_series=current_regime,
                    rule_based_swing=swing_info['is_swing'],
                    swing_sl_price=swing_info['sl_price'],
                    swing_tp_price=swing_info['tp_price']
                )

                # C2. Entry atau Pyramiding : EXECUTION ENGINE (OPENING)
                if risk_eval['execute_trade']:
                    if len(self.active_trades) == 0:
                        # Initial Entry (Maks 1 posisi aktif utama)
                        self._execute_buy_order(current_date, risk_eval, preds, current_regime=current_regime)
                    elif self.enable_pyramiding:
                        # Pyramiding: Tambah posisi ke trade aktif yang sedang berjalan jika tren terkonfirmasi kuat
                        active_trade = self.active_trades[0]
                        avg_cost = active_trade['capital_spent'] / (active_trade['shares'] + 1e-9)
                        unrealized_gain = (entry_price_today - avg_cost) / avg_cost
                        pyramid_count = active_trade.get('pyramid_count', 0)

                        # Syarat Pyramiding Terintegrasi:
                        # 1. Sedang dalam kondisi Strong Uptrend (terkonfirmasi)
                        # 2. Posisi sudah untung minimal +5% (Anti-Martingale: hanya tambah saat profit)
                        # 3. Jumlah penambahan belum mencapai batas maksimal (pyramid_count < max_pyramid_adds)
                        # 4. Kas tunai masih mencukupi reserve aman
                        min_cash_reserve = min(1_000_000.0, self.initial_capital * 0.10)
                        if (active_trade.get('is_strong_uptrend', False) and 
                            unrealized_gain >= self.pyramid_profit_threshold and 
                            pyramid_count < self.max_pyramid_adds and 
                            self.current_capital > min_cash_reserve):
                            
                            # Alokasi pyramiding: 40% dari sisa kas (atau 25% modal awal jika kas berlebih)
                            pyramid_alloc = min(self.current_capital * 0.40, self.initial_capital * 0.25)
                            cost_buy = entry_price_today * (1 + self.fee_buy)
                            add_lots = math.floor(pyramid_alloc / (100 * cost_buy))
                            
                            if add_lots > 0:
                                add_capital = add_lots * 100 * cost_buy
                                self.current_capital -= add_capital
                                
                                active_trade['shares'] += add_lots * 100
                                active_trade['lots'] += add_lots
                                active_trade['capital_spent'] += add_capital
                                active_trade['cost_per_share'] = active_trade['capital_spent'] / active_trade['shares']
                                active_trade['entry_price'] = active_trade['cost_per_share']  # Weighted average entry price
                                active_trade['entry_price_inc_fee'] = active_trade['cost_per_share']
                                active_trade['pyramid_count'] = pyramid_count + 1
                                
                                if 'pyramid_adds' not in active_trade:
                                    active_trade['pyramid_adds'] = []
                                active_trade['pyramid_adds'].append({
                                    'date': current_date,
                                    'price': entry_price_today,
                                    'lots': add_lots,
                                    'capital': add_capital,
                                    'unrealized_gain_at_add': unrealized_gain
                                })
                                
                                # Trailing Stop Loss untuk melindungi modal akumulasi
                                active_trade['sl_price'] = max(active_trade['sl_price'], entry_price_today * 0.90)
                                
                                market_val = active_trade['shares'] * entry_price_today
                                floating_pnl_rp = market_val - active_trade['capital_spent']
                                floating_pnl_pct = (floating_pnl_rp / active_trade['capital_spent']) * 100
                                total_equity = self.current_capital + market_val
                                print(f"  [PYRAMID ADD #{active_trade['pyramid_count']}] {current_date.date()} | Add Value: Rp{market_val:>11,.2f} | Beli Tambah @ Rp{entry_price_today:<7,.2f} | Tambah Lot: +{add_lots:<4} | Total Lot: {active_trade['lots']:<4} | Avg Entry: Rp{active_trade['entry_price']:<7,.2f}")
                                print(f"       |->  [STATUS]     Total Invested: Rp{active_trade['capital_spent']:>11,.2f} | P&L: Rp{floating_pnl_rp:>+11,.2f} ({floating_pnl_pct:>+6.2f}%) | Trading Balance: Rp{self.current_capital:>11,.2f} | Total Equity: Rp{total_equity:>11,.2f}")

                # C3. Exit : END-OF-DAY EVALUATION: Cek Exit (TP, SL, Time-Stop) untuk posisi aktif
                self._process_active_exits(current_date, df_raw_prices, regime_series=current_regime) # output : self.trade_history.append(log_entry)

                # Simpan Prediksi Harian
                self.daily_predictions.append({
                    'date': current_date,
                    'raw_buy_signal': risk_eval['raw_buy_signal'],
                    'pred_trend_slope': preds['trend_slope'],
                    'pred_return': preds['return'],
                    'pred_risk': preds['risk'],
                    'pred_is_dip': preds['is_dip'],
                    'actual_rr_ratio': risk_eval['actual_rr_ratio'],
                    'signal_buy': risk_eval['execute_trade']
                })

                # Record Ekuitas Harian
                active_value = sum(t['shares'] * df_raw_prices.loc[current_date, 'Close'] for t in self.active_trades)
                self.equity_curve.append({
                    'date': current_date,
                    'cash': self.current_capital,
                    'invested': active_value,
                    'total_equity': self.current_capital + active_value,
                    'active_positions': len(self.active_trades)
                })

            # D. Retrain : EXPANDING WINDOW UPDATE
            X_month_actual = month_features[feature_cols]
            X_train_current = pd.concat([X_train_current, X_month_actual])

            if isinstance(y_oos_dict, dict):
                for target in self.targets:
                    if target in y_oos_dict:
                        y_month_target = y_oos_dict[target].loc[month_features.index]
                        if target in y_train_current:
                            y_train_current[target] = pd.concat([y_train_current[target], y_month_target])
                        else:
                            y_train_current[target] = y_month_target
            else:
                y_month_actual = y_oos_dict.loc[month_features.index]
                y_train_current = pd.concat([y_train_current, y_month_actual])

            self._retrain_models(X_train_current, y_train_current)

        # Force close any remaining active trades on the last available date
        if self.active_trades:
            last_date = X_oos.index[-1]
            print(f"\n[INFO] Menutup paksa {len(self.active_trades)} posisi aktif di akhir backtest ({last_date.date()})")
            self._process_active_exits(last_date, df_raw_prices, force_close=True)

        # Konversi prediksi menjadi DataFrame
        if self.daily_predictions:
            self.predictions_df = pd.DataFrame(self.daily_predictions).set_index('date')
            self.predictions_df.index = pd.to_datetime(self.predictions_df.index)

        return self._summarize_performance()

    def _execute_buy_order(self, date, risk_eval: dict, preds: dict, current_regime: int = 2):
        """Mencatat transaksi beli baru ke portofolio aktif"""
        capital_needed = risk_eval['capital_spent_idr']
        
        # Validasi ketersediaan modal tunai
        if capital_needed > self.current_capital or risk_eval['allocated_lots'] <= 0:
            return

        # Potong Kas
        self.current_capital -= capital_needed
        is_strong_uptrend = (current_regime == 1)
        is_bear_swing = risk_eval.get('swing_triggered', False)
        
        # Susun struktur data trade aktif
        trade = {
            'trade_id': len(self.trade_history) + len(self.active_trades) + 1,
            'entry_date': date,
            'entry_price': risk_eval['entry_price'],
            'entry_price_inc_fee': risk_eval['entry_price_inc_fee'],
            'cost_per_share': risk_eval['entry_price'],
            'lots': risk_eval['allocated_lots'],
            'shares': risk_eval['allocated_lots'] * 100,
            'capital_spent': capital_needed,
            'tp_price': risk_eval['suggested_take_profit'],
            'sl_price': risk_eval['suggested_stop_loss'],
            'days_held': 0,
            'is_strong_uptrend': is_strong_uptrend,
            'is_bear_swing': is_bear_swing,
            'regime_mode': risk_eval.get('regime_mode', 'normal'),
            'pyramid_count': 0,
            'pyramid_adds': []
        }
        
        self.active_trades.append(trade)
        total_equity = self.current_capital + capital_needed
        if is_bear_swing:
            regime_tag = "[BEAR SWING]"
        elif is_strong_uptrend:
            regime_tag = "[STRONG UPTREND]"
        else:
            regime_tag = "[NORMAL]"
        print(f"  [BUY LOG]      {date.date()} | Beli {regime_tag:<16} @ Rp{trade['entry_price']:<7,.2f} | Lot: {trade['lots']:<4} | Avg Entry: Rp{trade['entry_price']*1.0015:<7,.2f}")
        print(f"       |->  [STATUS]     Invested: Rp{capital_needed:>11,.2f} | P&L: Rp{0.0:>11,.2f} ( +0.00%) | Trading Balance: Rp{self.current_capital:>11,.2f} | Total Equity: Rp{total_equity:>11,.2f}")
        print(f"       |->  [REASON]     Dip: {preds['return']*100:.2f}% | Slope: {preds['trend_slope']:.4f} | Ret: {preds['return']*100:.2f}% | Risk: {preds['risk']*100:.2f}% | RR: {risk_eval['actual_rr_ratio']:.2f} | Dip: {preds['is_dip']*100:.2f}%")
        print(f"       |->  [TARGET]     TP: {trade['tp_price']:<7,.2f} | SL: {trade['sl_price']:<7,.2f}")

    def _process_active_exits(self, current_date, df_raw_prices: pd.DataFrame, force_close: bool = False, regime_series: int = 2):
        """Mengevaluasi Hard-Exit (TP, SL, MA Crossover, Time-Stop) pada akhir hari bursa"""
        if not self.active_trades or current_date not in df_raw_prices.index:
            return

        today_price = df_raw_prices.loc[current_date]
        high_price = today_price['High']
        low_price = today_price['Low']
        close_price = today_price['Close']

        ma5_curr = self.ma5_series.loc[current_date] if self.ma5_series is not None and current_date in self.ma5_series.index else None
        ma10_curr = self.ma10_series.loc[current_date] if self.ma10_series is not None and current_date in self.ma10_series.index else None

        for trade in self.active_trades[:]:
            is_entry_day = (trade['entry_date'] == current_date)
            
            if not force_close and not is_entry_day:
                trade['days_held'] += 1
            
            # Upgrade ke strong uptrend jika:
            # 1. Pasar hari ini terkonfirmasi regime 1 (Uptrend)
            # 2. Posisi swing trade berhasil mencetak Golden Cross EMA 5 > EMA 10 (riding rally baru)
            if regime_series == 1:
                trade['is_strong_uptrend'] = True
            elif trade.get('is_bear_swing', False) and ma5_curr is not None and ma10_curr is not None and ma5_curr > ma10_curr:
                trade['had_golden_cross'] = True
                trade['is_strong_uptrend'] = True

            in_strong_uptrend = trade.get('is_strong_uptrend', False)

            hit_tp = False
            hit_ma_cross = False
            hit_time_stop = False

            if in_strong_uptrend:
                # Batas TP dihilangkan untuk menangkap reli bullish besar
                hit_tp = False
                # Langsung exit ketika MA 5 berpotongan / menembus ke bawah MA 10
                if not is_entry_day and ma5_curr is not None and ma10_curr is not None and ma5_curr <= ma10_curr:
                    hit_ma_cross = True

                # SL tetap melindungi modal dari crash/gap down
                hit_sl = low_price <= trade['sl_price'] * 0.75
                # Time-stop dihilangkan agar riding bullish tidak terputus prematur
                hit_time_stop = False
            else:
                hit_tp = high_price >= trade['tp_price']
                hit_sl = low_price <= trade['sl_price']
                # Swing trade holding limit: 5 hari jika belum golden cross
                max_hold = 5 if trade.get('is_bear_swing', False) else self.max_holding_days
                hit_time_stop = trade['days_held'] >= max_hold

            if hit_tp or hit_sl or hit_time_stop or hit_ma_cross or force_close:
                # Penentuan Harga Jual Realistis
                if force_close:
                    exit_price = close_price
                    exit_reason = "Akhir Backtest"
                elif hit_sl:
                    exit_price = trade['sl_price']
                    exit_reason = "Stop Loss"
                elif hit_ma_cross:
                    exit_price = close_price
                    exit_reason = f"{self.ma_type.upper()} 5/10 Cross"
                elif hit_tp:
                    exit_price = trade['tp_price']
                    exit_reason = "Take Profit"
                else:
                    exit_price = close_price
                    exit_reason = "Time-Stop"

                # Kalkulasi Hasil Penjualan Bersih (Dikurangi Fee Jual)
                gross_proceeds = trade['shares'] * exit_price
                net_proceeds = gross_proceeds * (1 - self.fee_sell)
                net_profit = net_proceeds - trade['capital_spent']
                roi_pct = (net_profit / trade['capital_spent']) * 100

                # Kembalikan dana ke kas tunai
                self.current_capital += net_proceeds
                
                # Kalkulasi total ekuitas setelah jual
                active_value = sum(t['shares'] * close_price for t in self.active_trades if t != trade)
                total_equity = self.current_capital + active_value

                # Catat ke riwayat
                log_entry = {
                    'trade_id': trade['trade_id'],
                    'entry_date': trade['entry_date'],
                    'exit_date': current_date,
                    'entry_price': trade['entry_price'],
                    'exit_price': exit_price,
                    'lots': trade['lots'],
                    'capital_spent': trade['capital_spent'],
                    'net_proceeds': net_proceeds,
                    'net_profit': net_profit,
                    'roi_pct': roi_pct,
                    'days_held': trade['days_held'],
                    'exit_reason': exit_reason,
                    'tp_price': trade['tp_price'],
                    'sl_price': trade['sl_price'],
                    'pyramid_count': trade.get('pyramid_count', 0),
                    'pyramid_adds': trade.get('pyramid_adds', [])
                }
                
                self.trade_history.append(log_entry)
                self.active_trades.remove(trade)

                # Update cooldown tracker jika ini adalah swing trade
                if trade.get('is_bear_swing', False):
                    self.last_swing_exit_date = current_date

                pyramid_tag = f" [PYRAMID x{trade.get('pyramid_count', 0)}]" if trade.get('pyramid_count', 0) > 0 else ""
                swing_tag = " [SWING]" if trade.get('is_bear_swing', False) else ""
                tag = pyramid_tag if pyramid_tag else swing_tag
                print(f"  [SELL LOG]     {current_date.date()} | Jual ({exit_reason:<16}){tag:<15} @ Rp{exit_price:<7,.2f} | Lot: {trade['lots']:<4}")
                print(f"       |->  [STATUS]     Invested: Rp{trade['capital_spent']:>11,.2f} | P&L: Rp{net_profit:>+11,.2f} ({roi_pct:>+6.2f}%) | Trading Balance: Rp{self.current_capital:>11,.2f} | Total Equity: Rp{total_equity:>11,.2f}\n")

    def _detect_rule_based_swing(self, current_date, df_raw_prices: pd.DataFrame, current_regime: int) -> dict:
        """
        Precision Wave Bottom Detector — Lapisan kedua untuk menangkap dasar gelombang di bear market.
        
        Prinsip Utama:
        1. Jangan pernah beli saat pisau jatuh (falling knife) — kemarin HARUS candle hijau pembalikan!
        2. Jangan pernah mengejar harga yang sudah melonjak terlalu jauh dari dasar (> 18% di atas low 3 hari).
        3. RSI kemarin harus masih dalam zona akumulasi/recovering (<= 46), bukan sudah overbought (> 50).
        4. Support bertahan (Higher Low): Low kemarin tidak menembus titik terendah 10 hari sebelumnya.
        5. Momentum MACD mulai membaik (MACD Histogram kemarin > 2 hari lalu).
        6. Stop Loss diletakkan di bawah retest low kemarin (bukan persentase sembarangan).
        
        Returns:
            dict: {
                "is_swing": bool,
                "sl_price": float or None,
                "tp_price": float or None
            }
        """
        null_res = {"is_swing": False, "sl_price": None, "tp_price": None}
        
        if current_date not in df_raw_prices.index:
            return null_res
        
        idx = df_raw_prices.index.get_loc(current_date)
        if idx < 50:
            return null_res
            
        current_open = df_raw_prices.loc[current_date, 'Open']
        yesterday = df_raw_prices.iloc[idx - 1]
        prev_day = df_raw_prices.iloc[idx - 2]
        
        y_close = yesterday['Close']
        y_open = yesterday['Open']
        y_low = yesterday['Low']
        
        y_ema5 = self.ma5_series.iloc[idx - 1] if self.ma5_series is not None else y_close
        y_ema10 = self.ma10_series.iloc[idx - 1] if self.ma10_series is not None else y_close
        y_ema20 = self.ma20_series.iloc[idx - 1] if self.ma20_series is not None else y_close
        
        # 1. Regime condition:
        # Established downtrend (regime 3) ATAU Deep Crash Reversal (DD50 <= -35% dan EMA5 < EMA10 < EMA20)
        # Menangkap crash bear market 2026 tanpa tertipu noise sideways 2024-2025
        is_established_downtrend = (current_regime == 3)
        high_50d = df_raw_prices['High'].iloc[max(0, idx-51):idx].max()
        dd_50d = (y_close - high_50d) / (high_50d + 1e-9)
        is_deep_crash = (dd_50d <= -0.35) and (y_ema5 < y_ema10) and (y_ema10 < y_ema20)
        
        if not (is_established_downtrend or is_deep_crash):
            return null_res
        
        # 1. Precondition: RSI dalam 8 hari terakhir sempat menyentuh oversold (< 38)
        if self.rsi_series is None or current_date not in self.rsi_series.index:
            return null_res
            
        rsi_8d_min = self.rsi_series.iloc[max(0, idx-9):idx].min()
        if pd.isna(rsi_8d_min) or rsi_8d_min >= 38:
            return null_res
            
        # 2. Anti-Chasing Filter 1: RSI kemarin harus masih <= 46 (belum overbought / overextended)
        rsi_yesterday = self.rsi_series.iloc[idx - 1]
        if pd.isna(rsi_yesterday) or rsi_yesterday > 46:
            return null_res
            
        # 3. Confirmation Candle: Kemarin candle hijau & tembus ke atas EMA5
        y_ema5 = self.ma5_series.iloc[idx - 1] if self.ma5_series is not None else y_close
        is_green_reversal = (y_close > y_open) and (y_close > prev_day['Close']) and (y_close >= y_ema5)
        if not is_green_reversal:
            return null_res
            
        # 4. Anti-Chasing Filter 2: Open hari ini tidak boleh > 18% di atas low 3 hari terakhir
        recent_low_3d = df_raw_prices['Low'].iloc[max(0, idx-4):idx].min()
        chase_pct = (current_open - recent_low_3d) / (recent_low_3d + 1e-9)
        if chase_pct > 0.18:
            return null_res
            
        # 5. Higher Low / Support Defense:
        # Low kemarin tidak boleh jebol low 10 hari sebelumnya
        prior_low_10d = df_raw_prices['Low'].iloc[max(0, idx-11):idx-1].min()
        if y_low < (prior_low_10d * 0.99):
            return null_res
            
        # 6. Momentum MACD membaik
        if self.macd_hist_series is not None:
            y_macd = self.macd_hist_series.iloc[idx - 1]
            prev_macd = self.macd_hist_series.iloc[idx - 2]
            if y_macd <= prev_macd:
                return null_res
                
        # 7. Structural SL & TP
        sl_price = float(round(y_low * 0.96))  # 4% di bawah retest low kemarin
        risk_pct = (current_open - sl_price) / current_open
        if not (0.02 <= risk_pct <= 0.10):
            return null_res
            
        tp_price = float(round(current_open * (1 + max(0.08, risk_pct * 1.8))))
        
        return {
            "is_swing": True,
            "sl_price": sl_price,
            "tp_price": tp_price
        }

    def _retrain_models(self, X_train: pd.DataFrame, y_train_dict):
        """
        Melatih ulang model independen menggunakan pipeline yang sudah dimuat sebelumnya.
        """
        print(f"\n[RETRAIN] Melatih ulang {len(self.trained_pipelines)} model dengan data Expanding Window ({len(X_train)} baris)...")
        
        for target in self.targets:
            if target not in self.trained_pipelines or target not in self.features_used:
                continue
                
            target_features = self.features_used[target]
            
            # Gabungkan X_train dan y_train khusus target ini
            df_temp = X_train.copy()
            if isinstance(y_train_dict, dict):
                if target in y_train_dict:
                    df_temp[target] = y_train_dict[target]
                else:
                    continue
            else:  # pd.DataFrame
                if target in y_train_dict.columns:
                    df_temp[target] = y_train_dict[target]
                else:
                    continue
            
            # Bersihkan baris yang mengandung NaN
            df_clean = df_temp.dropna(subset=target_features + [target])
            X_clean = df_clean[target_features].values
            y_clean = df_clean[target].values
            
            # Retrain model pipeline scikit-learn yang sudah dimuat
            self.trained_pipelines[target].fit(X_clean, y_clean)

        print(f"[RETRAIN] Seluruh model berhasil diperbarui.\n")

    def _summarize_performance(self):
        """Menghitung metrik evaluasi akhir portofolio"""
        df_trades = pd.DataFrame(self.trade_history)
        df_equity = pd.DataFrame(self.equity_curve)

        if df_trades.empty:
            print("\n[PERINGATAN] Tidak ada transaksi yang dieksekusi selama periode backtest.")
            return None, df_equity

        total_trades = len(df_trades)
        winning_trades = df_trades[df_trades['net_profit'] > 0]
        losing_trades = df_trades[df_trades['net_profit'] <= 0]

        win_rate = (len(winning_trades) / total_trades) * 100
        total_net_profit = df_trades['net_profit'].sum()
        final_equity = self.current_capital
        total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        expected_value = df_trades['net_profit'].mean()

        print("\n" + "=" * 70)
        print("                RINGKASAN PERFORMA BACKTEST (WFA)")
        print("=" * 70)
        print(f" Modal Awal            : Rp{self.initial_capital:,.2f}")
        print(f" Modal Akhir           : Rp{final_equity:,.2f}")
        print(f" Total Net Return      : Rp{total_net_profit:,.2f} ({total_return_pct:+.2f}%)")
        print(f" Expected Value (EV)   : Rp{expected_value:,.2f} / trade")
        print(f" Total Transaksi       : {total_trades}")
        print(f" Win Rate              : {win_rate:.2f}% ({len(winning_trades)} Menang / {len(losing_trades)} Kalah)")
        print("=" * 70)

        return df_trades, df_equity

    def save_trade_history_to_csv(self, df_raw_prices: pd.DataFrame, save_csv_path: str = None) -> pd.DataFrame:
        """Menghasilkan jurnal harian terintegrasi antara data OHLCV dan hasil trading. Bisa langsung disave ke CSV jika save_csv_path diisi."""
        if not self.equity_curve:
            return df_raw_prices.copy()
            
        start_date = self.equity_curve[0]['date']
        end_date = self.equity_curve[-1]['date']
        
        journal = df_raw_prices.loc[start_date:end_date].copy()
        
        # Tambahkan kolom default
        journal['Keputusan'] = '-'
        journal['Harga Eksekusi'] = 0.0
        journal['Lot'] = 0
        journal['Value Transaksi'] = 0.0
        journal['Invested'] = 0.0
        journal['Sisa Cash'] = 0.0
        journal['Total Equity'] = 0.0
        
        # Mapping data dari equity_curve
        df_eq = pd.DataFrame(self.equity_curve).set_index('date')
        if 'cash' in df_eq.columns:
            journal['Sisa Cash'] = df_eq['cash']
        if 'invested' in df_eq.columns:
            journal['Invested'] = df_eq['invested']
        if 'total_equity' in df_eq.columns:
            journal['Total Equity'] = df_eq['total_equity']
            
        # Mapping transaksi (Buy & Sell) dari trade_history (sudah closed)
        for trade in self.trade_history:
            entry_d = trade['entry_date']
            if entry_d in journal.index:
                if journal.at[entry_d, 'Keputusan'] == '-':
                    journal.at[entry_d, 'Keputusan'] = 'Buy'
                else:
                    journal.at[entry_d, 'Keputusan'] += ' & Buy'
                journal.at[entry_d, 'Harga Eksekusi'] = trade['entry_price']
                journal.at[entry_d, 'Lot'] = trade['lots']
                journal.at[entry_d, 'Value Transaksi'] = trade['capital_spent']
                
            for add in trade.get('pyramid_adds', []):
                add_d = add['date']
                if add_d in journal.index:
                    if journal.at[add_d, 'Keputusan'] == '-':
                        journal.at[add_d, 'Keputusan'] = 'Pyramid Buy'
                    else:
                        journal.at[add_d, 'Keputusan'] += ' & Pyramid Buy'
                    journal.at[add_d, 'Harga Eksekusi'] = add['price']
                    journal.at[add_d, 'Lot'] = add['lots']
                    journal.at[add_d, 'Value Transaksi'] = add['capital']

            exit_d = trade['exit_date']
            if exit_d in journal.index:
                if journal.at[exit_d, 'Keputusan'] == '-':
                    journal.at[exit_d, 'Keputusan'] = 'Sell'
                else:
                    journal.at[exit_d, 'Keputusan'] += ' & Sell'
                journal.at[exit_d, 'Harga Eksekusi'] = trade['exit_price']
                journal.at[exit_d, 'Lot'] = trade['lots']
                journal.at[exit_d, 'Value Transaksi'] = trade['net_proceeds']

        # Mapping transaksi (Buy) yang masih aktif (belum closed) jika ada
        for trade in self.active_trades:
            entry_d = trade['entry_date']
            if entry_d in journal.index:
                if journal.at[entry_d, 'Keputusan'] == '-':
                    journal.at[entry_d, 'Keputusan'] = 'Buy'
                else:
                    journal.at[entry_d, 'Keputusan'] += ' & Buy'
                journal.at[entry_d, 'Harga Eksekusi'] = trade['entry_price']
                journal.at[entry_d, 'Lot'] = trade['lots']
                journal.at[entry_d, 'Value Transaksi'] = trade['capital_spent']

            for add in trade.get('pyramid_adds', []):
                add_d = add['date']
                if add_d in journal.index:
                    if journal.at[add_d, 'Keputusan'] == '-':
                        journal.at[add_d, 'Keputusan'] = 'Pyramid Buy'
                    else:
                        journal.at[add_d, 'Keputusan'] += ' & Pyramid Buy'
                    journal.at[add_d, 'Harga Eksekusi'] = add['price']
                    journal.at[add_d, 'Lot'] = add['lots']
                    journal.at[add_d, 'Value Transaksi'] = add['capital']
                
        # Forward fill equity if any NaN
        journal['Invested'] = journal['Invested'].ffill()
        journal['Sisa Cash'] = journal['Sisa Cash'].ffill()
        journal['Total Equity'] = journal['Total Equity'].ffill()
        
        # Reorder columns: 1-5 (OHLCV), 6-12 custom
        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Keputusan', 'Harga Eksekusi', 'Lot', 'Value Transaksi', 'Invested', 'Sisa Cash', 'Total Equity']
        # If there are other existing columns, keep them at the end
        other_cols = [c for c in journal.columns if c not in cols]
        
        final_journal = journal[cols + other_cols].copy()
        
        # Format nilai ke dalam standar ribuan/jutaan dengan koma
        format_float = ['Open', 'High', 'Low', 'Close', 'Harga Eksekusi', 'Value Transaksi', 'Invested', 'Sisa Cash', 'Total Equity']
        format_int = ['Volume', 'Lot']
        
        for col in format_float:
            if col in final_journal.columns:
                final_journal[col] = final_journal[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
                
        for col in format_int:
            if col in final_journal.columns:
                final_journal[col] = final_journal[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else x)
        
        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path) or '.', exist_ok=True)
            final_journal.to_csv(save_csv_path)
            print(f"[INFO] Jurnal berhasil disimpan ke: {save_csv_path}")
            
        return final_journal

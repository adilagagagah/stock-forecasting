import pandas as pd
import numpy as np
import joblib
import os
import datetime
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scikeras.wrappers import KerasRegressor
from sklearn.base import BaseEstimator, TransformerMixin

class StandardScaler3D(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        if X.ndim == 3:
            n, t, f = X.shape
            self.scaler.fit(X.reshape(-1, f))
        else:
            self.scaler.fit(X)
        return self
        
    def transform(self, X):
        if X.ndim == 3:
            n, t, f = X.shape
            return self.scaler.transform(X.reshape(-1, f)).reshape(n, t, f)
        return self.scaler.transform(X)

class TargetModellingPipeline:
    def __init__(self, target_name: str, feature_cols: list, n_splits=5):
        """
        Pipeline untuk melatih, mengoptimasi (GridSearch), dan mengevaluasi 
        satu variabel target spesifik secara terisolasi.
        """
        self.target_name = target_name
        self.feature_cols = feature_cols
        self.n_splits = n_splits
        
        self.best_pipeline = None
        self.model_name = None
        self.best_params = None
        self.metrics = {}

    def prepare_data(self, df: pd.DataFrame):
        """ Membersihkan NaN khusus untuk kombinasi fitur dan target ini saja """
        df_clean = df.dropna(subset=self.feature_cols + [self.target_name]).copy()
        X = df_clean[self.feature_cols].values
        y = df_clean[self.target_name].values
        return X, y

    def tune_and_fit(self, X, y, model_instance, param_grid: dict):
        """ Runs GridSearch menggunakan TimeSeriesSplit untuk menghindari data leakage """
        is_keras_3d = getattr(model_instance, 'model', None) is not None and model_instance.__class__.__name__ == 'KerasRegressor'
        
        if is_keras_3d:
            # Ubah data 2D menjadi 3D secara sliding window (mundur 10 hari)
            timesteps = 10
            X_new, y_new = [], []
            for i in range(len(X) - timesteps + 1):
                X_new.append(X[i:i+timesteps])
                y_new.append(y[i+timesteps-1])
            X_model = np.array(X_new)
            y_model = np.array(y_new)
            
            pipeline = Pipeline([
                ('scaler', StandardScaler3D()),
                ('model', model_instance)
            ])
        else:
            X_model = X
            y_model = y
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model_instance)
            ])
            
        # Sesuaikan key param_grid agar cocok dengan penamaan di dalam Pipeline ('model__')
        adjusted_grid = {f"model__{k}": v for k, v in param_grid.items()}
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=adjusted_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        
        # Penanganan khusus: TabNet mewajibkan target (y) berbentuk 2D
        if model_instance.__class__.__name__ == 'TabNetRegressor':
            y_model = y_model.reshape(-1, 1)

        grid_search.fit(X_model, y_model)
        
        # Simpan objek pipeline terbaik hasil tuning
        self.best_pipeline = grid_search.best_estimator_
        self.model_name = model_instance.__class__.__name__
        
        # Bersihkan prefix 'model__' dari nama hyperparameter agar rapi saat dibaca & disimpan
        self.best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}

        # Val_MAE: Rata-rata error pada Out-of-Fold (Validation)
        val_mae = -grid_search.best_score_
        
        # Train_MAE & Train_R2: Error murni pada data yang dilihatnya (In-Sample)
        pure_train_preds = self.best_pipeline.predict(X_model)
        
        # Ratakan dimensi y dan preds ke 1D untuk memastikan kalkulasi metrik aman (terutama untuk TabNet)
        y_eval = y_model.ravel()
        preds_eval = pure_train_preds.ravel()
        
        train_mae = mean_absolute_error(y_eval, preds_eval)
        train_r2 = r2_score(y_eval, preds_eval)

        # DIRECTIONAL ACCURACY (Hit Rate) - Sangat Krusial untuk Trading!
        # Menghitung seberapa sering model benar menebak arah (Naik/Turun)
        if self.target_name == 'trend_slope':
            valid_idx = y_eval != 0
            if valid_idx.sum() > 0:
                correct = np.sign(y_eval[valid_idx]) == np.sign(preds_eval[valid_idx])
                hit_rate = np.mean(correct) * 100
            else:
                hit_rate = 0.0
        elif 'days_to_' in self.target_name:
            pred_rounded = np.clip(np.round(preds_eval), 1, 5)
            correct = np.abs(y_eval - pred_rounded) <= 1
            hit_rate = np.mean(correct) * 100
        elif self.target_name == 'return':
            # BAGUS: Kenyataan (y) LEBIH TINGGI atau SAMA DENGAN Prediksi (dikurangi toleransi meleset 1%)
            # Cth: Pred 2%, Aktual 5% -> 5% >= (2% - 1%) -> TRUE (Take Profit Tersentuh)
            # Cth: Pred 5%, Aktual 1% -> 1% >= (5% - 1%) -> FALSE (Gagal)
            valid_preds = preds_eval > 0 # Hanya hitung jika model menyuruh beli (prediksi positif)
            if valid_preds.sum() > 0:
                correct = y_eval[valid_preds] >= (preds_eval[valid_preds] - 0.01)
                hit_rate = np.mean(correct) * 100
            else:
                hit_rate = 0.0     
        elif self.target_name == 'risk':
            # BAGUS: Kenyataan (y) TIDAK LEBIH DALAM dari Prediksi (ditambah toleransi 1%)
            # Cth: Pred -5%, Aktual -3% -> -3% >= (-5% - 1%) -> TRUE (Stop Loss Aman)
            # Cth: Pred -5%, Aktual -10% -> -10% >= (-5% - 1%) -> FALSE (Stop Loss Jebol)
            valid_preds = preds_eval < 0 # Hanya hitung prediksi penurunan
            if valid_preds.sum() > 0:
                correct = y_eval[valid_preds] >= (preds_eval[valid_preds] - 0.01)
                hit_rate = np.mean(correct) * 100
            else:
                hit_rate = 0.0
        else:
            hit_rate = 0.0
        
        # Tampilkan informasi model terbaik ke konsol secara eksplisit
        print(f"   -> [HYPERPARAMETERS]: {self.best_params}")
        print(f"   -> [MODEL METRICS]: val MAE {val_mae:.4f} | Hit Rate {hit_rate:.2f}%")
        
        self.metrics = {
            'Train_MAE': train_mae,
            'Val_MAE': val_mae,
            'Train_R2': train_r2,
            'Hit_Rate_%': hit_rate
        }
        return self.best_pipeline

    def save(self, folder_path: str, ticker: str = None):
        """ Simpan model beserta metadata lengkap ke dalam satu file .pkl """
        if self.best_pipeline is None:
            raise ValueError("Model belum dilatih.")

        base_folder = Path(folder_path)
        
        if ticker:
            target_folder = base_folder / ticker
            filepath = target_folder / f"model_{self.target_name}_{ticker}.pkl"
        else:
            target_folder = base_folder
            filepath = target_folder / f"model_{self.target_name}.pkl"
        
        # SOLUSI FileNotFoundError: Otomatis buat folder jika belum ada
        target_folder.mkdir(parents=True, exist_ok=True)
        
        # Menyusun struktur metadata ke dalam dictionary payload
        metadata_payload = {
            'pipeline': self.best_pipeline,                     # Objek model + scaler asli
            'metadata': {
                'model_name': self.model_name,
                'target_name': self.target_name,
                'best_params': self.best_params,
                'in_sample_metrics': self.metrics,
                'features_used': self.feature_cols,
                'trained_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        if ticker:
            metadata_payload['metadata']['ticker'] = ticker
        
        # Ekspor berkas paket lengkap
        joblib.dump(metadata_payload, filepath)
        print(f"[SAVED] Berhasil mengekspor paket model + metadata ke: {filepath.as_posix()}")


class ModelEvaluator:
    """
    Class untuk mengevaluasi dan membandingkan performa model machine learning 
    terhadap baseline berdasarkan karakteristik variabel target.
    """
    def __init__(self, target_name, baseline_name="DummyBaseline"):
        self.target_name = target_name
        self.baseline_name = baseline_name

    def _sort_challengers(self, df_results):
        """Mengurutkan seluruh model penantang dari yang terbaik hingga terburuk"""
        challenger_data = df_results[df_results['Model'] != self.baseline_name]

        if self.target_name in ['return', 'trend_slope']:
            # Prioritas: Hit Rate tertinggi -> Val MAE terendah
            return challenger_data.sort_values(
                by=['Hit_Rate_%', 'Val_MAE'], 
                ascending=[False, True]
            )
        else: # risk, days_to_max, days_to_min
            # Prioritas: Val MAE terendah -> Hit Rate tertinggi
            return challenger_data.sort_values(
                by=['Val_MAE', 'Hit_Rate_%'], 
                ascending=[True, False]
            )

    def _check_fit_status(self, train_mae, val_mae, train_r2):
        """Logika Overfit Detector"""
        if train_mae == 0:
            return "Overfit Parah"
        if train_r2 > 0.95: 
            return "Overfit Parah (Menghafal Data, R2>95%)"
        
        mae_degradation = (val_mae - train_mae) / train_mae
        if train_r2 > 0.80 and mae_degradation > 0.15:
            # R2 tinggi tapi error membesar di validasi
            return "Overfitting (R2 Tinggi & Error Melebar)"
        elif mae_degradation > 0.25:
            return "Overfitting"
        elif mae_degradation < -0.10: 
            return "Aneh (Val lebih baik)"
        else:
            return "Good Fit"
    
    def _select_optimal_challenger(self, sorted_challengers):
        """
        Mencari model terbaik yang berstatus Good Fit. 
        Jika semua model overfit, otomatis menggunakan model peringkat pertama (fallback).
        """
        if sorted_challengers.empty:
            raise ValueError("Tidak ada model penantang yang tersedia untuk dievaluasi.")

        skipped_models = []
        for index, row in sorted_challengers.iterrows():
            c_train_mae = row['Train_MAE']
            c_val_mae = row['Val_MAE']
            c_train_r2 = row['Train_R2']
            
            status = self._check_fit_status(c_train_mae, c_val_mae, c_train_r2)
            
            # Jika menemukan yang Good Fit, langsung kembalikan baris tersebut dan hentikan pencarian
            if status == "Good Fit":
                reason = "Memiliki skor metrik prioritas tertinggi DAN berstatus 'Good Fit' (tidak overfitting)."
                if skipped_models:
                    reason += f" Sistem mendiskualifikasi ({', '.join(skipped_models)}) karena statusnya Overfit."
                return row, reason
            else:
                skipped_models.append(row['Model'])
                
        # Fallback: Jika loop selesai dan tidak ada yang 'Good Fit', ambil peringkat 1
        best_fallback_data = sorted_challengers.iloc[0]
        return best_fallback_data, "Semua model berstatus Overfit. Memaksa memilih model peringkat ke-1 sebagai fallback (Harap waspada)."

    def _make_decision(self, c_hit_rate, bl_hit_rate, c_val_mae, bl_val_mae, fit_status):
        """Menentukan apakah model lolos mengalahkan baseline"""
        if self.target_name in ['return', 'trend_slope']:
            if c_hit_rate > bl_hit_rate:
                 return "Lolos" if fit_status == "Good Fit" else "Lolos (Awas Overfit)"
            else:
                 return "Gagal (Kalah Hit Rate dr Baseline)"
        else:
            if c_val_mae < bl_val_mae:
                 return "Lolos" if fit_status == "Good Fit" else "Lolos (Awas Overfit)"
            else:
                 return "Gagal (Kalah MAE dr Baseline)"

    def generate_evaluation_report(self, df_results):
        """Method utama untuk mengeksekusi seluruh logika evaluasi dan mengembalikan dictionary laporan"""
        baseline_data = df_results[df_results['Model'] == self.baseline_name].iloc[0]
        sorted_challengers = self._sort_challengers(df_results)
        best_challenger_data, selection_reason = self._select_optimal_challenger(sorted_challengers)

        c_train_mae = best_challenger_data['Train_MAE']
        c_val_mae = best_challenger_data['Val_MAE']
        c_hit_rate = best_challenger_data['Hit_Rate_%']
        c_train_r2 = best_challenger_data['Train_R2']
        
        bl_train_mae = baseline_data['Train_MAE']
        bl_val_mae = baseline_data['Val_MAE']
        bl_hit_rate = baseline_data['Hit_Rate_%']
        bl_train_r2 = baseline_data['Train_R2']

        challenger_fit_status = self._check_fit_status(c_train_mae, c_val_mae, c_train_r2)
        baseline_fit_status = self._check_fit_status(bl_train_mae, bl_val_mae, bl_train_r2)
        decision = self._make_decision(c_hit_rate, bl_hit_rate, c_val_mae, bl_val_mae, challenger_fit_status)

        # Logika penentuan akhir 'Model Terpilih'
        if "Lolos" in decision:
            final_model = best_challenger_data['Model']
            best_hit_rate = c_hit_rate
            final_fit_status = challenger_fit_status
            final_reason = selection_reason
        else:
            final_model = baseline_data['Model']
            best_hit_rate = bl_hit_rate
            final_fit_status = baseline_fit_status
            final_reason = "Model penantang gagal mengalahkan Baseline (Kalah metrik dasar)."

        metric_priority = "Hit Rate terbesar -> Val MAE terkecil" if self.target_name in ['return', 'trend_slope'] else "Val MAE terkecil -> Hit Rate terbesar"

        return {
            'Target': self.target_name,
            'Baseline Train MAE': round(bl_train_mae, 5),
            'Baseline Val MAE': round(bl_val_mae, 5),
            'Baseline Hit Rate (%)': f"{bl_hit_rate:.2f}%",
            'Challenger Train MAE': round(c_train_mae, 5),
            'Challenger Val MAE': round(c_val_mae, 5),
            'Challenger Hit Rate (%)': f"{c_hit_rate:.2f}%",
            'Keputusan': decision,
            'Model Terpilih': final_model,
            'Best Hit Rate': f"{round(best_hit_rate, 2)}%",
            'Fit Status': final_fit_status,
            'Prioritas Seleksi': metric_priority,
            'Alasan Pemilihan': final_reason
        }
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import datetime

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
        # Bungkus dengan Pipeline untuk memastikan scaling terjadi per fold CV
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
        
        grid_search.fit(X, y)
        
        # Simpan objek pipeline terbaik hasil tuning
        self.best_pipeline = grid_search.best_estimator_
        self.model_name = model_instance.__class__.__name__
        
        # Bersihkan prefix 'model__' dari nama hyperparameter agar rapi saat dibaca & disimpan
        self.best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
        
        # Tampilkan informasi model terbaik ke konsol secara eksplisit
        print(f"   -> [MODEL TERBAIK]: {self.model_name}")
        print(f"   -> [HYPERPARAMETERS]: {self.best_params}")

        # Val_MAE: Rata-rata error pada Out-of-Fold (Validation)
        val_mae = -grid_search.best_score_
        
        # Train_MAE & Train_R2: Error murni pada data yang dilihatnya (In-Sample)
        pure_train_preds = self.best_pipeline.predict(X)
        train_mae = mean_absolute_error(y, pure_train_preds)
        train_r2 = r2_score(y, pure_train_preds)

        # DIRECTIONAL ACCURACY (Hit Rate) - Sangat Krusial untuk Trading!
        # Menghitung seberapa sering model benar menebak arah (Naik/Turun)
        # Mengabaikan target yang nilainya 0 untuk menghindari bias
        if self.target_name == 'trend_slope':
            # Target Tren: Hit Rate diukur dari kesamaan arah (Positif=Naik, Negatif=Turun)
            valid_idx = y != 0
            if valid_idx.sum() > 0:
                correct = np.sign(y[valid_idx]) == np.sign(pure_train_preds[valid_idx])
                hit_rate = np.mean(correct) * 100
            else:
                hit_rate = 0.0
        elif 'days_to_' in self.target_name:
            # Target Waktu: Hit Rate diukur jika tebakan meleset MAKSIMAL 1 hari (Toleransi ketat)
            pred_rounded = np.clip(np.round(pure_train_preds), 1, 5)
            correct = np.abs(y - pred_rounded) <= 1
            hit_rate = np.mean(correct) * 100
        elif self.target_name == 'return':
            # BAGUS: Kenyataan (y) LEBIH TINGGI atau SAMA DENGAN Prediksi (dikurangi toleransi meleset 1%)
            # Cth: Pred 2%, Aktual 5% -> 5% >= (2% - 1%) -> TRUE (Take Profit Tersentuh)
            # Cth: Pred 5%, Aktual 1% -> 1% >= (5% - 1%) -> FALSE (Gagal)
            valid_preds = pure_train_preds > 0 # Hanya hitung jika model menyuruh beli (prediksi positif)
            if valid_preds.sum() > 0:
                correct = y[valid_preds] >= (pure_train_preds[valid_preds] - 0.01)
                hit_rate = np.mean(correct) * 100
            else:
                hit_rate = 0.0     
        elif self.target_name == 'risk':
            # BAGUS: Kenyataan (y) TIDAK LEBIH DALAM dari Prediksi (ditambah toleransi 1%)
            # Cth: Pred -5%, Aktual -3% -> -3% >= (-5% - 1%) -> TRUE (Stop Loss Aman)
            # Cth: Pred -5%, Aktual -10% -> -10% >= (-5% - 1%) -> FALSE (Stop Loss Jebol)
            valid_preds = pure_train_preds < 0 # Hanya hitung prediksi penurunan
            if valid_preds.sum() > 0:
                correct = y[valid_preds] >= (pure_train_preds[valid_preds] - 0.01)
                hit_rate = np.mean(correct) * 100
            else:
                hit_rate = 0.0
        else:
            hit_rate = 0.0
        
        self.metrics = {
            'Train_MAE': train_mae,
            'Val_MAE': val_mae,
            'Train_R2': train_r2,
            'Hit_Rate_%': hit_rate
        }
        return self.best_pipeline

    def save(self, folder_path: str):
        """ Simpan model beserta metadata lengkap ke dalam satu file .pkl """
        if self.best_pipeline is None:
            raise ValueError("Model belum dilatih.")
        
        # SOLUSI FileNotFoundError: Otomatis buat folder jika belum ada
        os.makedirs(folder_path, exist_ok=True)
        
        filepath = os.path.join(folder_path, f"model_{self.target_name}.pkl")
        
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
        
        # Ekspor berkas paket lengkap
        joblib.dump(metadata_payload, filepath)
        print(f"[SAVED] Berhasil mengekspor paket model + metadata ke: {filepath}")
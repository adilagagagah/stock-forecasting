import pandas as pd
import numpy as np
import joblib
import datetime
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score
)

# Target klasifikasi — selain ini dianggap regresi
CLASSIFICATION_TARGETS = ['is_dip']


# ------------------------------------------------------------------
# MODELLING PIPELINE
# ------------------------------------------------------------------
class ModellingPipeline:
    """
    Pipeline terpadu untuk melatih, mengoptimasi (GridSearch), dan mengevaluasi
    satu variabel target secara terisolasi. Mendukung regresi dan klasifikasi
    secara otomatis berdasarkan nama target.
    """

    def __init__(self, target_name: str, feature_cols: list, n_splits: int = 5):
        self.target_name = target_name
        self.feature_cols = feature_cols
        self.n_splits = n_splits
        self.is_classifier = target_name in CLASSIFICATION_TARGETS

        self.best_pipeline = None
        self.model_name = None
        self.best_params = None
        self.metrics = {}

    # DATA PREPARATION
    def prepare_data(self, df: pd.DataFrame):
        """Membersihkan NaN khusus untuk kombinasi fitur dan target ini saja."""
        df_clean = df.dropna(subset=self.feature_cols + [self.target_name]).copy()
        X = df_clean[self.feature_cols].values
        y = df_clean[self.target_name].values

        if self.is_classifier:
            y = y.astype(int)

        return X, y

    # TRAINING + GRID SEARCH
    def tune_and_fit(self, X, y, model_instance, param_grid: dict):
        """
        Menjalankan GridSearchCV dengan TimeSeriesSplit.
        Otomatis memilih scoring dan metrik berdasarkan tipe target.
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_instance)
        ])

        # Sesuaikan key param_grid agar cocok dengan penamaan Pipeline ('model__')
        adjusted_grid = {f"model__{k}": v for k, v in param_grid.items()}
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        scoring = 'f1' if self.is_classifier else 'neg_mean_absolute_error'

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=adjusted_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y)

        # Simpan objek pipeline terbaik hasil tuning
        self.best_pipeline = grid_search.best_estimator_
        self.model_name = model_instance.__class__.__name__
        self.best_params = {
            k.replace('model__', ''): v
            for k, v in grid_search.best_params_.items()
        }

        # Tampilkan hyperparameter terbaik
        print(f"   -> [HYPERPARAMETERS]: {self.best_params}")

        if self.is_classifier:
            self._compute_classification_metrics(X, y, grid_search.best_score_)
        else:
            self._compute_regression_metrics(X, y, grid_search.best_score_)

        return self.best_pipeline

    # METRIK: KLASIFIKASI
    def _compute_classification_metrics(self, X, y, val_f1):
        """Menghitung metrik khusus klasifikasi (Precision, Recall, F1)."""
        train_preds = self.best_pipeline.predict(X)
        train_precision = precision_score(y, train_preds, zero_division=0)
        train_recall = recall_score(y, train_preds, zero_division=0)
        train_f1 = f1_score(y, train_preds, zero_division=0)

        # Distribusi label
        n_positive = int(y.sum())
        n_total = len(y)

        print(f"   -> [LABEL DIST]: {n_positive}/{n_total} positif ({n_positive/n_total*100:.1f}%)")
        print(f"   -> [TRAIN]: Precision {train_precision:.2%} | Recall {train_recall:.2%} | F1 {train_f1:.2%}")
        print(f"   -> [VAL F1]: {val_f1:.2%}")

        self.metrics = {
            'Train_Precision': round(train_precision * 100, 2),
            'Train_Recall': round(train_recall * 100, 2),
            'Train_F1': round(train_f1 * 100, 2),
            'Val_F1': round(val_f1 * 100, 2),
            'Label_Positive_Pct': round(n_positive / n_total * 100, 2)
        }

    # METRIK: REGRESI
    def _compute_regression_metrics(self, X, y, best_score):
        """Menghitung metrik khusus regresi (MAE, R2, Hit Rate)."""
        val_mae = -best_score

        # In-sample prediction
        pure_train_preds = self.best_pipeline.predict(X)

        y_eval = y.ravel()
        preds_eval = pure_train_preds.ravel()

        train_mae = mean_absolute_error(y_eval, preds_eval)
        train_r2 = r2_score(y_eval, preds_eval)

        # DIRECTIONAL ACCURACY (Hit Rate) — krusial untuk Trading
        hit_rate = self._calculate_hit_rate(y_eval, preds_eval)

        print(f"   -> [TRAIN]: train MAE {train_mae:.4f} | Hit Rate {hit_rate:.2f}%")
        print(f"   -> [EVAL]: val MAE {val_mae:.4f}")

        self.metrics = {
            'Train_MAE': train_mae,
            'Val_MAE': val_mae,
            'Train_R2': train_r2,
            'Hit_Rate_%': hit_rate
        }

    def _calculate_hit_rate(self, y_eval, preds_eval):
        """Menghitung hit rate berdasarkan karakteristik target regresi."""
        if self.target_name == 'trend_slope':
            valid_idx = y_eval != 0
            if valid_idx.sum() > 0:
                correct = np.sign(y_eval[valid_idx]) == np.sign(preds_eval[valid_idx])
                return np.mean(correct) * 100
            return 0.0

        elif self.target_name == 'return':
            # BAGUS: Kenyataan (y) LEBIH TINGGI atau SAMA DENGAN Prediksi (dikurangi toleransi 1%)
            valid_preds = preds_eval > 0
            if valid_preds.sum() > 0:
                correct = y_eval[valid_preds] >= (preds_eval[valid_preds] - 0.01)
                return np.mean(correct) * 100
            return 0.0

        elif self.target_name == 'risk':
            # BAGUS: Kenyataan (y) TIDAK LEBIH DALAM dari Prediksi (ditambah toleransi 1%)
            valid_preds = preds_eval < 0
            if valid_preds.sum() > 0:
                correct = y_eval[valid_preds] >= (preds_eval[valid_preds] - 0.01)
                return np.mean(correct) * 100
            return 0.0

        return 0.0

    # SAVE MODEL
    def save(self, folder_path: str, ticker: str = None):
        """Simpan model beserta metadata lengkap ke dalam satu file .pkl"""
        if self.best_pipeline is None:
            raise ValueError("Model belum dilatih.")

        base_folder = Path(folder_path)

        if ticker:
            target_folder = base_folder / ticker
            filepath = target_folder / f"model_{self.target_name}_{ticker}.pkl"
        else:
            target_folder = base_folder
            filepath = target_folder / f"model_{self.target_name}.pkl"

        target_folder.mkdir(parents=True, exist_ok=True)

        metadata_payload = {
            'pipeline': self.best_pipeline,
            'metadata': {
                'model_name': self.model_name,
                'target_name': self.target_name,
                'best_params': self.best_params,
                'in_sample_metrics': self.metrics,
                'features_used': self.feature_cols,
                'model_type': 'classifier' if self.is_classifier else 'regressor',
                'trained_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        if ticker:
            metadata_payload['metadata']['ticker'] = ticker

        joblib.dump(metadata_payload, filepath)
        model_type_label = "classifier" if self.is_classifier else "regressor"
        print(f"[SAVED] Berhasil mengekspor model {model_type_label} ke: {filepath.as_posix()}")


# ------------------------------------------------------------------
# MODEL EVALUATOR
# ------------------------------------------------------------------
class ModelEvaluator:
    """
    Evaluator terpadu untuk membandingkan performa model machine learning
    terhadap baseline. Mendukung regresi dan klasifikasi secara otomatis.
    """

    def __init__(self, target_name: str, baseline_name: str = "DummyBaseline"):
        self.target_name = target_name
        self.baseline_name = baseline_name
        self.is_classifier = target_name in CLASSIFICATION_TARGETS

    # SORTING CHALLENGERS
    def _sort_challengers(self, df_results):
        """Mengurutkan seluruh model penantang dari yang terbaik hingga terburuk."""
        challenger_data = df_results[df_results['Model'] != self.baseline_name]

        # Klasifikasi: Val_F1 tertinggi → Train_Precision tertinggi
        if self.is_classifier:
            return challenger_data.sort_values(
                by=['Val_F1', 'Train_Precision'],
                ascending=[False, False]
            )

        # Regresi: Hit rate tinggi → Val_MAE rendah
        if self.target_name in ['return', 'trend_slope']:
            return challenger_data.sort_values(
                by=['Hit_Rate_%', 'Val_MAE'],
                ascending=[False, True]
            )
        # risk : Val_MAE rendah → Hit rate tinggi
        else:
            return challenger_data.sort_values(
                by=['Val_MAE', 'Hit_Rate_%'],
                ascending=[True, False]
            )

    # OVERFIT DETECTOR
    def _check_fit_status(self, row):
        """Mendeteksi status overfitting berdasarkan tipe task."""
        if self.is_classifier:
            return self._check_fit_classifier(row)
        else:
            return self._check_fit_regressor(row)

    def _check_fit_classifier(self, row):
        """Overfit Detector khusus Classifier."""
        train_f1 = row['Train_F1']
        val_f1 = row['Val_F1']

        if train_f1 == 0 and val_f1 == 0:
            return "Tidak Belajar (F1=0)"
        if train_f1 > 0 and val_f1 == 0:
            return "Overfit Parah (Val F1=0)"

        f1_degradation = (train_f1 - val_f1) / (train_f1 + 1e-9)
        if f1_degradation > 0.40:
            return "Overfitting (F1 Drop >40%)"
        elif f1_degradation > 0.25:
            return "Overfitting"
        return "Good Fit"

    def _check_fit_regressor(self, row):
        """Overfit Detector khusus Regressor."""
        train_mae = row['Train_MAE']
        val_mae = row['Val_MAE']
        train_r2 = row['Train_R2']

        if train_mae == 0:
            return "Overfit Parah"
        if train_r2 > 0.95:
            return "Overfit Parah (Menghafal Data, R2>95%)"

        mae_degradation = (val_mae - train_mae) / train_mae
        if train_r2 > 0.80 and mae_degradation > 0.15:
            return "Overfitting (R2 Tinggi & Error Melebar)"
        elif mae_degradation > 0.25:
            return "Overfitting"
        elif mae_degradation < -0.10:
            return "Aneh (Val lebih baik)"
        return "Good Fit"

    # OPTIMAL CHALLENGER SELECTION
    def _select_optimal_challenger(self, sorted_challengers):
        """
        Mencari model terbaik yang berstatus Good Fit.
        Jika semua model overfit, otomatis menggunakan model peringkat pertama (fallback).
        """
        if sorted_challengers.empty:
            raise ValueError("Tidak ada model penantang yang tersedia untuk dievaluasi.")

        skipped_models = []
        for _, row in sorted_challengers.iterrows():
            status = self._check_fit_status(row)

            if status == "Good Fit":
                reason = self._build_selection_reason(row, skipped_models)
                return row, reason
            else:
                skipped_models.append(row['Model'])

        # Fallback: semua overfit
        best_fallback = sorted_challengers.iloc[0]
        return best_fallback, "Semua model berstatus Overfit. Memaksa memilih model peringkat ke-1 sebagai fallback (Harap waspada)."

    def _build_selection_reason(self, row, skipped_models):
        """Membangun string alasan pemilihan model."""
        if self.is_classifier:
            reason = f"Val F1 tertinggi ({row['Val_F1']:.2f}%) DAN berstatus 'Good Fit'."
        else:
            reason = "Memiliki skor metrik prioritas tertinggi DAN berstatus 'Good Fit' (tidak overfitting)."

        if skipped_models:
            reason += f" Sistem mendiskualifikasi ({', '.join(skipped_models)}) karena statusnya Overfit."
        return reason

    # DECISION LOGIC
    def _make_decision(self, best_challenger, baseline_data, fit_status):
        """Menentukan apakah model penantang lolos mengalahkan baseline."""
        if self.is_classifier:
            if best_challenger['Val_F1'] > baseline_data['Val_F1']:
                return "Lolos" if fit_status == "Good Fit" else "Lolos (Awas Overfit)"
            return "Gagal (Kalah F1 dari Baseline)"

        # Regresi
        if self.target_name in ['return', 'trend_slope']:
            if best_challenger['Hit_Rate_%'] > baseline_data['Hit_Rate_%']:
                return "Lolos" if fit_status == "Good Fit" else "Lolos (Awas Overfit)"
            return "Gagal (Kalah Hit Rate dr Baseline)"
        else:
            if best_challenger['Val_MAE'] < baseline_data['Val_MAE']:
                return "Lolos" if fit_status == "Good Fit" else "Lolos (Awas Overfit)"
            return "Gagal (Kalah MAE dr Baseline)"

    # ------------------------------------------------------------------
    # REPORT GENERATION
    # ------------------------------------------------------------------
    def generate_evaluation_report(self, df_results):
        """Method utama: eksekusi seluruh logika evaluasi dan kembalikan dictionary laporan."""
        baseline_data = df_results[df_results['Model'] == self.baseline_name].iloc[0]
        sorted_challengers = self._sort_challengers(df_results)
        best_challenger, selection_reason = self._select_optimal_challenger(sorted_challengers)

        challenger_fit = self._check_fit_status(best_challenger)
        baseline_fit = self._check_fit_status(baseline_data)
        decision = self._make_decision(best_challenger, baseline_data, challenger_fit)

        if self.is_classifier:
            return self._build_classification_report(
                best_challenger, baseline_data,
                challenger_fit, baseline_fit,
                decision, selection_reason
            )
        return self._build_regression_report(
            best_challenger, baseline_data,
            challenger_fit, baseline_fit,
            decision, selection_reason
        )

    def _build_classification_report(self, challenger, baseline, c_fit, bl_fit, decision, reason):
        """Menyusun laporan evaluasi untuk model klasifikasi."""
        if "Lolos" in decision:
            final_model = challenger['Model']
            final_fit = c_fit
            final_reason = reason
        else:
            final_model = baseline['Model']
            final_fit = bl_fit
            final_reason = "Tidak ada model yang mengalahkan Baseline."

        return {
            'Target': self.target_name,
            'Baseline Val_F1': f"{baseline['Val_F1']:.2f}%",
            'Challenger Model': challenger['Model'],
            'Challenger Val_F1': f"{challenger['Val_F1']:.2f}%",
            'Challenger Precision': f"{challenger['Train_Precision']:.2f}%",
            'Challenger Recall': f"{challenger['Train_Recall']:.2f}%",
            'Keputusan': decision,
            'Model Terpilih': final_model,
            'Fit Status': final_fit,
            'Alasan Pemilihan': final_reason
        }

    def _build_regression_report(self, challenger, baseline, c_fit, bl_fit, decision, reason):
        """Menyusun laporan evaluasi untuk model regresi."""
        if "Lolos" in decision:
            final_model = challenger['Model']
            best_hit_rate = challenger['Hit_Rate_%']
            final_fit = c_fit
            final_reason = reason
        else:
            final_model = baseline['Model']
            best_hit_rate = baseline['Hit_Rate_%']
            final_fit = bl_fit
            final_reason = "Model penantang gagal mengalahkan Baseline (Kalah metrik dasar)."

        metric_priority = (
            "Hit Rate terbesar -> Val MAE terkecil"
            if self.target_name in ['return', 'trend_slope']
            else "Val MAE terkecil -> Hit Rate terbesar"
        )

        return {
            'Target': self.target_name,
            'Baseline Train MAE': round(baseline['Train_MAE'], 5),
            'Baseline Val MAE': round(baseline['Val_MAE'], 5),
            'Baseline Hit Rate (%)': f"{baseline['Hit_Rate_%']:.2f}%",
            'Challenger Train MAE': round(challenger['Train_MAE'], 5),
            'Challenger Val MAE': round(challenger['Val_MAE'], 5),
            'Challenger Hit Rate (%)': f"{challenger['Hit_Rate_%']:.2f}%",
            'Keputusan': decision,
            'Model Terpilih': final_model,
            'Best Hit Rate': f"{round(best_hit_rate, 2)}%",
            'Fit Status': final_fit,
            'Prioritas Seleksi': metric_priority,
            'Alasan Pemilihan': final_reason
        }
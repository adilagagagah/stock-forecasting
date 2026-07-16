from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, PoissonRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

FEATURES = [
    'ATR', 'RSI', 'Volume_Ratio', 'Log_Return', 'distance_to_support', 'distance_to_resistance',
    'MACD_Hist', 'Dist_to_MA20', 'Hist_Volatility_5d'
]

TARGET = ['return', 'risk', 'trend_slope', 'days_to_max', 'days_to_min']

BASELINE_CONFIG = {
    'name': 'DummyBaseline',
    'model_object': DummyRegressor(strategy='mean'),
    'grid_params': {} # Tidak butuh hyperparameter tuning
}

MODEL_CONFIGS = {
    # 1. Linear-Regularized Model
    'ridge_regression': {
        'model_object': Ridge(random_state=42),
        'grid_params': {
            'alpha': [0.1, 1.0, 10.0, 100.0] # Kekuatan regularisasi
        }
    },
    'elastic_net': {
        'model_object': ElasticNet(random_state=42, max_iter=2000),
        'grid_params': {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.2, 0.5, 0.8] # Porsi L1 regularization vs L2
        }
    },
    # 2. Tree Based Models (basic)
    'random_forest': {
        'model_object': RandomForestRegressor(random_state=42),
        'grid_params': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 8],
            'min_samples_split': [2, 5]
        }
    },
    'extra_trees': {
        'model_object': ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'grid_params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 8],
            'min_samples_split': [2, 5]
        }
    },
    # 3. Gradient Boosting
    'xgboost': {
        'model_object': XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
        'grid_params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    },
    'lightgbm': {
        'model_object': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, force_col_wise=True),
        'grid_params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'num_leaves': [15, 31, 63]
        }
    },
    'catboost': {
        'model_object': CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False), # verbose=0 agar log tidak penuh
        'grid_params': {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8]
        }
    },
    # 4. Distance-Based Models
    'svr': {
        'model_object': SVR(),
        'grid_params': {
            'kernel': ['rbf'], # RBF paling umum untuk pola non-linear saham
            'C': [0.1, 1.0, 10.0], # Parameter penalti
            'epsilon': [0.01, 0.1, 0.2] # Batas toleransi error
        }
    },
    'knn': {
        'model_object': KNeighborsRegressor(n_jobs=-1),
        'grid_params': {
            'n_neighbors': [3, 5, 11, 21], # Jumlah tetangga terdekat
            'weights': ['uniform', 'distance'] # Pembobotan berdasarkan jarak
        }
    },
    # 5. Generalized Linear Models
    'poisson_regression': {
        'model_object': PoissonRegressor(),
        'grid_params': {
            'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0] # Regularisasi
        }
    }
}
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, PoissonRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Dropout, LayerNormalization, MultiHeadAttention
from tcn import TCN # dari keras-tcn
from scikeras.wrappers import KerasRegressor

FEATURES = [
    'ATR', 'RSI', 'Volume_Ratio', 'Log_Return', 'distance_to_support', 'distance_to_resistance',
    'MACD_Hist', 'Dist_to_MA20', 'Hist_Volatility_5d',
    'RSI_3', 'BB_Pct', 'Capitulation_Vol', 'Lower_Shadow_Pct', 'Upper_Shadow_Pct', 'Body_Pct', 'Williams_R'
]

TARGET = ['return', 'risk', 'trend_slope', 'days_to_max', 'days_to_min']

TIMESTEPS = 10 
FEATURES_COUNT = len(FEATURES) 

def build_lstm(hidden_units=64, dropout_rate=0.2):
    model = Sequential([
        LSTM(hidden_units, input_shape=(TIMESTEPS, FEATURES_COUNT), return_sequences=False),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def build_gru(hidden_units=64, dropout_rate=0.2):
    model = Sequential([
        GRU(hidden_units, input_shape=(TIMESTEPS, FEATURES_COUNT), return_sequences=False),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def build_tcn(nb_filters=64, kernel_size=3):
    model = Sequential([
        TCN(nb_filters=nb_filters, kernel_size=kernel_size, input_shape=(TIMESTEPS, FEATURES_COUNT), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def build_transformer(head_size=32, num_heads=2, dropout_rate=0.2):
    inputs = Input(shape=(TIMESTEPS, FEATURES_COUNT))
    # Attention Block
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout_rate)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    # Gunakan Global Average Pooling untuk meratakan 3D ke 2D
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mae')
    return model


BASELINE_CONFIG = {
    'name': 'DummyBaseline',
    'model_object': DummyRegressor(strategy='mean'),
    'grid_params': {}, # Tidak butuh hyperparameter tuning
    'data_format': '2d'
}

MODEL_CONFIGS = {
    # 1. Linear-Regularized Model
    'ridge_regression': {
        'model_object': Ridge(random_state=42),
        'grid_params': {
            'alpha': [0.1, 1.0, 10.0, 100.0] # Kekuatan regularisasi
        },
        'data_format': '2d'
    },
    'elastic_net': {
        'model_object': ElasticNet(random_state=42, max_iter=2000),
        'grid_params': {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.2, 0.5, 0.8] # Porsi L1 regularization vs L2
        },
        'data_format': '2d'
    },
    # 2. Tree Based Models (basic)
    'random_forest': {
        'model_object': RandomForestRegressor(random_state=42),
        'grid_params': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 8],
            'min_samples_split': [2, 5]
        },
        'data_format': '2d'
    },
    'extra_trees': {
        'model_object': ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'grid_params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 8],
            'min_samples_split': [2, 5]
        },
        'data_format': '2d'
    },
    # 3. Gradient Boosting
    'xgboost': {
        'model_object': XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
        'grid_params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        },
        'data_format': '2d'
    },
    'lightgbm': {
        'model_object': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, force_col_wise=True),
        'grid_params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'num_leaves': [15, 31, 63]
        },
        'data_format': '2d'
    },
    'catboost': {
        'model_object': CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False), # verbose=0 agar log tidak penuh
        'grid_params': {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8]
        },
        'data_format': '2d'
    },
    # 4. Distance-Based Models
    'svr': {
        'model_object': SVR(),
        'grid_params': {
            'kernel': ['rbf'], # RBF paling umum untuk pola non-linear saham
            'C': [0.1, 1.0, 10.0], # Parameter penalti
            'epsilon': [0.01, 0.1, 0.2] # Batas toleransi error
        },
        'data_format': '2d'
    },
    'knn': {
        'model_object': KNeighborsRegressor(n_jobs=-1),
        'grid_params': {
            'n_neighbors': [3, 5, 11, 21], # Jumlah tetangga terdekat
            'weights': ['uniform', 'distance'] # Pembobotan berdasarkan jarak
        },
        'data_format': '2d'
    },
    # 5. Generalized Linear Models
    'poisson_regression': {
        'model_object': PoissonRegressor(),
        'grid_params': {
            'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0] # Regularisasi
        },
        'data_format': '2d'
    },
    # 6. Deep Learning & Advanced Models (2D)
    'ngboost': {
        'model_object': NGBRegressor(random_state=42, verbose=False),
        'grid_params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05],
            # Minibatch untuk mempercepat NGBoost di data saham
            'minibatch_frac': [1.0, 0.5] 
        },
        'data_format': '2d'
    },
    'tabnet': {
        'model_object': TabNetRegressor(verbose=0, seed=42),
        'grid_params': {
            'n_d': [8, 16], # Dimensi decision prediction layer
            'n_a': [8, 16], # Dimensi attention layer
            'n_steps': [3, 5] # Jumlah arsitektur bertingkat
        },
        'data_format': '2d'
    },
    # 7. Deep Learning & Advanced Models (3D)
    'lstm': {
        'model_object': KerasRegressor(model=build_lstm, verbose=0, epochs=20, batch_size=32),
        'grid_params': {
            'model__hidden_units': [32, 64],
            'model__dropout_rate': [0.1, 0.2]
        },
        'data_format': '3d'
    },
    'gru': {
        'model_object': KerasRegressor(model=build_gru, verbose=0, epochs=20, batch_size=32),
        'grid_params': {
            'model__hidden_units': [32, 64]
        },
        'data_format': '3d'
    },
    'tcn': {
        'model_object': KerasRegressor(model=build_tcn, verbose=0, epochs=20, batch_size=32),
        'grid_params': {
            'model__nb_filters': [32, 64],
            'model__kernel_size': [2, 3]
        },
        'data_format': '3d'
    },
    'transformer': {
        'model_object': KerasRegressor(model=build_transformer, verbose=0, epochs=20, batch_size=32),
        'grid_params': {
            'model__num_heads': [2, 4],
            'model__head_size': [32, 64]
        },
        'data_format': '3d'
    }
    
}
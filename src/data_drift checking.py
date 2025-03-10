import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

class LSTMDriftDetector:
    def __init__(self, threshold_ks=0.05, threshold_wd=0.2, mse_threshold=1.5):
        """
        Initializes the drift detector with threshold values.
        - threshold_ks: p-value threshold for Kolmogorov-Smirnov test (default 0.05)
        - threshold_wd: threshold for Wasserstein Distance (default 0.2)
        - mse_threshold: multiplier for MSE drift detection (default 1.5)
        """
        self.threshold_ks = threshold_ks
        self.threshold_wd = threshold_wd
        self.mse_threshold = mse_threshold
        self.mse_baseline = None  # Store initial model error for drift detection

    def detect_ks_drift(self, old_data, new_data):
        """Performs the Kolmogorov-Smirnov test to check for data drift."""
        stat, p_value = ks_2samp(old_data, new_data)
        return p_value < self.threshold_ks  # If p-value < threshold, drift is detected

    def detect_wasserstein_drift(self, old_data, new_data):
        """Calculates Wasserstein Distance to measure shift in data distribution."""
        drift_score = wasserstein_distance(old_data, new_data)
        return drift_score > self.threshold_wd  # If distance > threshold, drift is detected

    def detect_mse_drift(self, y_true_old, y_pred_old, y_true_new, y_pred_new):
        """Detects concept drift based on an increase in MSE over time."""
        mse_old = mean_squared_error(y_true_old, y_pred_old)
        mse_new = mean_squared_error(y_true_new, y_pred_new)
        
        if self.mse_baseline is None:
            self.mse_baseline = mse_old  # Set baseline MSE at first call
        
        return mse_new > (self.mse_baseline * self.mse_threshold)

    def train_autoencoder(self, X_train):
        """Trains an LSTM Autoencoder for detecting reconstruction drift."""
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            LSTM(50, activation='relu', return_sequences=False),
            RepeatVector(X_train.shape[1]),
            LSTM(50, activation='relu', return_sequences=True),
            TimeDistributed(Dense(X_train.shape[2]))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, X_train, epochs=10, batch_size=16, verbose=0)
        
        return model

    def detect_autoencoder_drift(self, autoencoder, X_test_old, X_test_new):
        """Uses an LSTM Autoencoder to detect drift by measuring reconstruction error."""
        reconstructed_old = autoencoder.predict(X_test_old)
        reconstructed_new = autoencoder.predict(X_test_new)

        error_old = np.mean(np.abs(reconstructed_old - X_test_old))
        error_new = np.mean(np.abs(reconstructed_new - X_test_new))

        return error_new > (error_old * self.mse_threshold)

    def detect_drift(self, train_df, new_df, feature_cols):
        """Runs multiple drift detection methods and returns the results."""
        drift_results = {}

        for col in feature_cols:
            old_data = train_df[col].dropna()
            new_data = new_df[col].dropna()

            drift_results[f"{col}_KS_Drift"] = self.detect_ks_drift(old_data, new_data)
            drift_results[f"{col}_WD_Drift"] = self.detect_wasserstein_drift(old_data, new_data)

        return drift_results

"""
Feature scaling utilities
"""

import numpy as np
from typing import Dict, List


class FeatureScaler:
    """
    Feature scaler for normalizing features to similar scales.
    Uses z-score normalization (StandardScaler approach) to handle features
    with different scales (e.g., UPDRS scores 0-100+ vs. percentages 0-1).
    
    Handles missing values properly: only normalizes observed values,
    missing values remain as 0 with mask=1.
    """
    
    def __init__(self):
        """Initialize empty scaler (must call fit() before use)"""
        self.means_ = {}
        self.stds_ = {}
        self.fitted_ = False
    
    def fit(self, feature_dict: Dict[str, np.ndarray], feature_names: List[str]):
        """
        Fit scaler on training data.
        
        Args:
            feature_dict: Dict mapping feature names to arrays of observed values
                         (missing values should be excluded)
            feature_names: List of feature names in order
        """
        self.means_ = {}
        self.stds_ = {}
        
        for i, name in enumerate(feature_names):
            if name in feature_dict:
                values = feature_dict[name]
                # Only use non-missing values for computing statistics
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    self.means_[name] = float(np.mean(valid_values))
                    std = float(np.std(valid_values))
                    # Avoid division by zero for constant features
                    self.stds_[name] = std if std > 1e-8 else 1.0
                else:
                    # All values missing - use defaults
                    self.means_[name] = 0.0
                    self.stds_[name] = 1.0
            else:
                # Feature not in data - use defaults
                self.means_[name] = 0.0
                self.stds_[name] = 1.0
        
        self.fitted_ = True
    
    def transform(self, values: np.ndarray, mask: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Normalize feature values using fitted statistics.
        
        Args:
            values: [n_samples, n_features] array of feature values (missing filled with 0)
            mask: [n_samples, n_features] array where 1=missing, 0=present
            feature_names: List of feature names in order
            
        Returns:
            Normalized values array (same shape as input)
        """
        if not self.fitted_:
            raise ValueError("Scaler must be fitted before transform")
        
        normalized = values.copy()
        
        for i, name in enumerate(feature_names):
            if i >= values.shape[1]:
                continue
            
            if name in self.means_:
                mean = self.means_[name]
                std = self.stds_[name]
                
                # Only normalize non-missing values (where mask == 0)
                # Missing values (mask == 1) remain as 0
                non_missing = (mask[:, i] == 0)
                if non_missing.any():
                    normalized[non_missing, i] = (values[non_missing, i] - mean) / std
        
        return normalized
    
    def fit_transform(self, values: np.ndarray, mask: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Fit scaler and transform in one step.
        
        Args:
            values: [n_samples, n_features] array of feature values
            mask: [n_samples, n_features] array where 1=missing, 0=present
            feature_names: List of feature names in order
            
        Returns:
            Normalized values array
        """
        # Extract observed values for fitting
        feature_dict = {}
        for i, name in enumerate(feature_names):
            if i < values.shape[1]:
                # Get non-missing values for this feature
                non_missing = (mask[:, i] == 0)
                if non_missing.any():
                    feature_dict[name] = values[non_missing, i]
        
        self.fit(feature_dict, feature_names)
        return self.transform(values, mask, feature_names)
    
    def get_params(self) -> Dict:
        """Get scaler parameters for saving"""
        return {
            'means': self.means_,
            'stds': self.stds_,
            'fitted': self.fitted_
        }
    
    def set_params(self, params: Dict):
        """Set scaler parameters from saved state"""
        self.means_ = params.get('means', {})
        self.stds_ = params.get('stds', {})
        self.fitted_ = params.get('fitted', False)

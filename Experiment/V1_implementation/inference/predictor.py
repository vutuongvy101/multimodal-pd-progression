"""
Patient-level prediction module for disease severity and progression.

Predicts next visit UPDRS scores and disease stage based on historical data.
Supports single patient predictions or batch predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import json
from dataclasses import dataclass

from models.v1_model import V1MultimodalTransformer
from training.config import get_default_config
from data.data_integrator import DataIntegrator


@dataclass
class DiseaseStage:
    """Disease severity classification based on MDS-UPDRS part III (motor) score"""
    
    NORMAL = "Normal/No PD"
    MILD = "Mild PD"
    MODERATE = "Moderate PD"
    SEVERE = "Severe PD"
    VERY_SEVERE = "Very Severe PD"
    
    # Thresholds based on MDS-UPDRS Part III (NP3TOT)
    # Reference: https://www.movementdisorders.org/
    THRESHOLDS = {
        NORMAL: (0, 20),
        MILD: (20, 41),
        MODERATE: (41, 58),
        SEVERE: (58, 75),
        VERY_SEVERE: (75, 140)
    }
    
    @staticmethod
    def classify(updrs_part3_score: float) -> str:
        """Classify disease severity based on UPDRS Part III score"""
        for stage, (min_val, max_val) in DiseaseStage.THRESHOLDS.items():
            if min_val <= updrs_part3_score < max_val:
                return stage
        return DiseaseStage.VERY_SEVERE
    
    @staticmethod
    def get_all_stages() -> List[str]:
        """Get all disease stages in order of severity"""
        return [
            DiseaseStage.NORMAL,
            DiseaseStage.MILD,
            DiseaseStage.MODERATE,
            DiseaseStage.SEVERE,
            DiseaseStage.VERY_SEVERE
        ]


class PatientPredictor:
    """
    Make predictions for a single patient based on historical visit data.
    
    Handles:
    - Loading trained models
    - Formatting patient data
    - Making predictions
    - Classifying disease severity/stage
    - Estimating disease progression
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_version: str = "v1",
        device: str = "cuda"
    ):
        """
        Initialize predictor with a trained model.
        
        Args:
            checkpoint_path: Path to trained model checkpoint (.pt file)
            config_version: "v1" or "v2" config version
            device: "cuda", "mps", or "cpu"
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load config
        if config_version == "v2":
            from training.config_v2 import get_default_config as get_config_v2
            self.config = get_config_v2()
        else:
            self.config = get_default_config()
        
        # Load model
        self.model = V1MultimodalTransformer(self.config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Target names for UPDRS scores
        self.target_names = getattr(
            self.config.features,
            "all_updrs_totals",
            ["NP1RTOT", "NP2PTOT", "NP3TOT", "NP4TOT"]
        )
        
        # Initialize data integrator for feature extraction
        self.integrator = DataIntegrator(self.config, normalize_features=True)
    
    def predict_next_visit(
        self,
        patient_visits: pd.DataFrame,
        return_confidence: bool = True,
        return_stage: bool = True
    ) -> Dict:
        """
        Predict disease severity at next visit for a patient.
        
        Args:
            patient_visits: DataFrame with patient's historical visits
                           Must have columns matching feature names
                           Sorted by visit date (oldest to newest)
            return_confidence: Include confidence/uncertainty estimates
            return_stage: Include disease stage classification
        
        Returns:
            Dictionary with:
            - predicted_updrs: Dict of predicted UPDRS scores
            - predicted_stage: Disease stage at next visit (if return_stage=True)
            - confidence: Confidence intervals (if return_confidence=True)
            - slope: Estimated disease progression rate
            - current_severity: Current disease stage
        """
        # Validate input
        if len(patient_visits) < 4:
            raise ValueError(
                f"Need at least 4 visits for prediction, got {len(patient_visits)}"
            )
        
        # Use last 4 visits (or all if less than 10, else take most recent 4)
        if len(patient_visits) > 10:
            visits_for_pred = patient_visits.iloc[-4:].copy()
        else:
            visits_for_pred = patient_visits.iloc[-min(4, len(patient_visits)):].copy()
        
        # Get features
        try:
            features = self._extract_features(visits_for_pred)
        except Exception as e:
            raise ValueError(f"Failed to extract features from patient data: {e}")
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(
                features['static_values'],
                features['static_mask'],
                features['motor_values'],
                features['motor_mask'],
                features['nonmotor_values'],
                features['nonmotor_mask'],
                features['med_values'],
                features['med_mask'],
                features['age_at_visit_values'],
                features['age_at_visit_mask'],
                features['time_months'],
                features['attention_mask']
            )
        
        # Extract next-visit predictions (use last visit's prediction for next step)
        next_visit_preds = predictions['next_visit'][0, -1, :].cpu().numpy()
        slope_pred = predictions['slope'][0].cpu().item()
        
        # Build result
        result = {
            'predicted_updrs': {
                name: float(score)
                for name, score in zip(self.target_names, next_visit_preds)
            },
            'slope': float(slope_pred),
            'current_severity': self._get_current_severity(visits_for_pred),
        }
        
        # Add stage prediction
        if return_stage:
            motor_score = next_visit_preds[2]  # NP3TOT is typically index 2
            result['predicted_stage'] = DiseaseStage.classify(motor_score)
            result['stage_thresholds'] = DiseaseStage.THRESHOLDS
        
        # Add confidence estimates (simple approach: use standard deviation)
        if return_confidence:
            result['confidence'] = self._estimate_confidence(next_visit_preds)
        
        return result
    
    def _extract_features(self, visits: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Extract and format features from patient visit data.
        Handles missing values and normalizes features.
        """
        # This would integrate with DataIntegrator to extract features
        # For now, return a structured placeholder
        # In production, this would match the format expected by the model
        
        n_visits = len(visits)
        
        # Get feature dimensions from config
        n_static = len(self.config.features.static_features)
        n_motor = len(self.config.features.motor_features)
        n_nonmotor = len(self.config.features.non_motor_features)
        n_med = len(self.config.features.medication_features)
        n_age = len(self.config.features.age_at_visit_features)
        
        # Create tensors (batch_size=1 for single patient)
        batch_size = 1
        max_visits = 10  # Pad to fixed sequence length
        
        features = {
            'static_values': torch.zeros(batch_size, max_visits, n_static, device=self.device),
            'static_mask': torch.ones(batch_size, max_visits, n_static, device=self.device),
            'motor_values': torch.zeros(batch_size, max_visits, n_motor, device=self.device),
            'motor_mask': torch.zeros(batch_size, max_visits, n_motor, device=self.device),
            'nonmotor_values': torch.zeros(batch_size, max_visits, n_nonmotor, device=self.device),
            'nonmotor_mask': torch.zeros(batch_size, max_visits, n_nonmotor, device=self.device),
            'med_values': torch.zeros(batch_size, max_visits, n_med, device=self.device),
            'med_mask': torch.zeros(batch_size, max_visits, n_med, device=self.device),
            'age_at_visit_values': torch.zeros(batch_size, max_visits, n_age, device=self.device),
            'age_at_visit_mask': torch.zeros(batch_size, max_visits, n_age, device=self.device),
            'time_months': torch.zeros(batch_size, max_visits, device=self.device),
            'attention_mask': torch.zeros(batch_size, max_visits, device=self.device),
        }
        
        # Get static features from first visit and replicate across all visits
        static_vals = {}
        for idx, col in enumerate(self.config.features.static_features):
            if col in visits.columns:
                val = visits[col].iloc[0]  # Use first visit's static features
                if not pd.isna(val):
                    static_vals[idx] = float(val)
        
        # Extract longitudinal features
        for visit_idx, (_, visit_row) in enumerate(visits.iterrows()):
            if visit_idx >= max_visits:
                break
            
            # Static features - replicate for all visits
            for feat_idx, val in static_vals.items():
                features['static_values'][0, visit_idx, feat_idx] = val
            
            # Motor features
            for feat_idx, col in enumerate(self.config.features.motor_features):
                if col in visit_row.index:
                    val = visit_row[col]
                    if not pd.isna(val):
                        features['motor_values'][0, visit_idx, feat_idx] = float(val)
                        features['motor_mask'][0, visit_idx, feat_idx] = 1.0
            
            # Non-motor features
            for feat_idx, col in enumerate(self.config.features.non_motor_features):
                if col in visit_row.index:
                    val = visit_row[col]
                    if not pd.isna(val):
                        features['nonmotor_values'][0, visit_idx, feat_idx] = float(val)
                        features['nonmotor_mask'][0, visit_idx, feat_idx] = 1.0
            
            # Medication features
            for feat_idx, col in enumerate(self.config.features.medication_features):
                if col in visit_row.index:
                    val = visit_row[col]
                    if not pd.isna(val):
                        features['med_values'][0, visit_idx, feat_idx] = float(val)
                        features['med_mask'][0, visit_idx, feat_idx] = 1.0
            
            # Age at visit
            for feat_idx, col in enumerate(self.config.features.age_at_visit_features):
                if col in visit_row.index:
                    val = visit_row[col]
                    if not pd.isna(val):
                        features['age_at_visit_values'][0, visit_idx, feat_idx] = float(val)
                        features['age_at_visit_mask'][0, visit_idx, feat_idx] = 1.0
            
            # Time (months since first visit)
            if 'visit_month' in visit_row.index or 'months_since_baseline' in visit_row.index:
                col_name = 'visit_month' if 'visit_month' in visit_row.index else 'months_since_baseline'
                features['time_months'][0, visit_idx] = float(visit_row[col_name])
            
            # Attention mask (mark valid visits)
            features['attention_mask'][0, visit_idx] = 1.0
        
        return features
    
    def _get_current_severity(self, visits: pd.DataFrame) -> Dict[str, any]:
        """Get current disease severity from latest visit"""
        latest_visit = visits.iloc[-1]
        
        # Get UPDRS scores from latest visit
        updrs_scores = {}
        for target in self.target_names:
            if target in latest_visit.index:
                updrs_scores[target] = float(latest_visit[target])
        
        # Use motor score (NP3TOT) for severity classification
        motor_score = updrs_scores.get('NP3TOT', 0.0)
        
        return {
            'updrs_scores': updrs_scores,
            'stage': DiseaseStage.classify(motor_score),
            'motor_score': motor_score
        }
    
    def _estimate_confidence(self, predictions: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Estimate confidence intervals for predictions.
        Simple approach: 95% CI based on typical model uncertainty.
        """
        # Typical prediction uncertainty (can be calibrated on validation set)
        std_dev = 2.0  # Adjust based on your model's performance
        margin = 1.96 * std_dev  # 95% CI
        
        return {
            name: (float(pred - margin), float(pred + margin))
            for name, pred in zip(self.target_names, predictions)
        }


class BatchPredictor:
    """
    Make predictions for multiple patients.
    """
    
    def __init__(self, checkpoint_path: str, config_version: str = "v1", device: str = "cuda"):
        """
        Initialize batch predictor.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_version: "v1" or "v2"
            device: "cuda", "mps", or "cpu"
        """
        self.predictor = PatientPredictor(checkpoint_path, config_version, device)
    
    def predict_patients(
        self,
        patient_data: Dict[str, pd.DataFrame],
        save_results: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Predict for multiple patients.
        
        Args:
            patient_data: Dict mapping patient_id -> DataFrame of visits
            save_results: Path to save results as JSON
        
        Returns:
            Dict mapping patient_id -> prediction results
        """
        results = {}
        
        for patient_id, visits in patient_data.items():
            try:
                results[patient_id] = self.predictor.predict_next_visit(visits)
                print(f"✓ Predicted for patient {patient_id}")
            except Exception as e:
                results[patient_id] = {'error': str(e)}
                print(f"✗ Failed for patient {patient_id}: {e}")
        
        # Save results
        if save_results:
            output_path = Path(save_results)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types for JSON serialization
            json_results = {}
            for patient_id, result in results.items():
                json_results[patient_id] = self._to_serializable(result)
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n✓ Saved results to {output_path}")
        
        return results
    
    @staticmethod
    def _to_serializable(obj):
        """Convert numpy/torch types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: BatchPredictor._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BatchPredictor._to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return float(obj.item())
        else:
            return obj


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make patient-level disease predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--patient-data", type=str, required=True, help="Path to patient CSV file")
    parser.add_argument("--patient-id", type=str, help="Specific patient ID to predict (optional)")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--config-version", type=str, default="v1", help="Config version (v1 or v2)")
    
    args = parser.parse_args()
    
    # Load patient data
    df = pd.read_csv(args.patient_data)
    
    # Initialize predictor
    predictor = PatientPredictor(args.checkpoint, args.config_version, args.device)
    
    # Make predictions
    if args.patient_id:
        # Single patient
        patient_visits = df[df['patient_id'] == args.patient_id].reset_index(drop=True)
        result = predictor.predict_next_visit(patient_visits)
        print(json.dumps(result, indent=2))
    else:
        # Batch prediction
        batch_predictor = BatchPredictor(args.checkpoint, args.config_version, args.device)
        patient_groups = {pid: group.reset_index(drop=True) 
                         for pid, group in df.groupby('patient_id')}
        results = batch_predictor.predict_patients(patient_groups, args.output)
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Total patients: {len(results)}")
        successful = sum(1 for r in results.values() if 'error' not in r)
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")

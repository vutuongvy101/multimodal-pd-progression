"""
Inference module for disease stage prediction.

Provides:
- PatientPredictor: Single patient prediction
- BatchPredictor: Multiple patient prediction
- DiseaseStage: Disease severity classification
"""

from .predictor import PatientPredictor, BatchPredictor, DiseaseStage

__all__ = ['PatientPredictor', 'BatchPredictor', 'DiseaseStage']

"""
Data loaders for V1 model
Each loader is independent and can be developed/tested separately
"""

from genetics_loader import GeneticsLoader
from demographics_loader import DemographicsLoader
from updrs_loader import UPDRSLoader
from clinical_loader import ClinicalAssessmentsLoader
from medication_loader import MedicationLoader

__all__ = [
    'GeneticsLoader',
    'DemographicsLoader',
    'UPDRSLoader',
    'ClinicalAssessmentsLoader',
    'MedicationLoader'
]

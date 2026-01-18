"""
Data loaders for V1 model
Each loader is independent and can be developed/tested separately
"""

try:
    # Relative imports when used as a module
    from .genetics_loader import GeneticsLoader
    from .demographics_loader import DemographicsLoader
    from .updrs_loader import UPDRSLoader
    from .clinical_loader import ClinicalAssessmentsLoader
    from .medication_loader import MedicationLoader
except ImportError:
    # Absolute imports when run as script
    from data.loaders.genetics_loader import GeneticsLoader
    from data.loaders.demographics_loader import DemographicsLoader
    from data.loaders.updrs_loader import UPDRSLoader
    from data.loaders.clinical_loader import ClinicalAssessmentsLoader
    from data.loaders.medication_loader import MedicationLoader

__all__ = [
    'GeneticsLoader',
    'DemographicsLoader',
    'UPDRSLoader',
    'ClinicalAssessmentsLoader',
    'MedicationLoader'
]

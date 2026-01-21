"""
Data loaders for V1 model
Each loader is independent and can be developed/tested separately
"""

try:
    # Relative imports when used as a module
    # from .genetics_loader_v2 import GeneticsLoader
    # from .demographics_loader_v2 import DemographicsLoader
    from .motor_loader_v2 import UPDRSLoader
    from .non_motor_loader_v2 import NonMotorAssessmentsLoader
    # from .medication_loader_v2 import MedicationLoader
    # from .age_at_visit_loader_v2 import AgeAtVisitLoader
except ImportError:
    # Absolute imports when run as script
    # from data.loaders.genetics_loader import GeneticsLoader
    # from data.loaders.demographics_loader import DemographicsLoader
    from data.loaders.updrs_loader import UPDRSLoader
    from data.loaders.non_motor_loader import NonMotorAssessmentsLoader
    # from data.loaders.medication_loader import MedicationLoader
    # from data.loaders.age_at_visit_loader import AgeAtVisitLoader

__all__ = [
    # 'GeneticsLoader',
    # 'DemographicsLoader',
    'UPDRSLoader',
    'NonMotorAssessmentsLoader',
    # 'MedicationLoader',
    # 'AgeAtVisitLoader'
]

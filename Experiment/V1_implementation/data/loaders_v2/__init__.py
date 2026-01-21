"""
Data loaders for V1 model
Each loader is independent and can be developed/tested separately
"""

try:
    # Relative imports when used as a module
    # from .genetics_loader_v2 import GeneticsLoader
    # from .demographics_loader_v2 import DemographicsLoader
    from .motor_loader import MotorLoader
    from .behavioral_loader import BehavioralLoader
    from .cognitive_loader import CognitiveLoader
    from .sleep_loader import SleepLoader
    from .smell_loader import SmellLoader
    # from .medication_loader_v2 import MedicationLoader
    # from .age_at_visit_loader_v2 import AgeAtVisitLoader
except ImportError:
    # Absolute imports when run as script
    # from data.loaders.genetics_loader import GeneticsLoader
    # from data.loaders.demographics_loader import DemographicsLoader
    from data.loaders.motor_loader import MotorLoader
    from data.loaders.behavioral_loader import BehavioralLoader
    from data.loaders.cognitive_loader import CognitiveLoader
    from data.loaders.sleep_loader import SleepLoader
    from data.loaders.smell_loader import SmellLoader
    # from data.loaders.medication_loader import MedicationLoader
    # from data.loaders.age_at_visit_loader import AgeAtVisitLoader

__all__ = [
    # 'GeneticsLoader',
    # 'DemographicsLoader',
    'MotorLoader',
    'BehavioralLoader',
    'CognitiveLoader',
    'SleepLoader',
    'SmellLoader',
    # 'MedicationLoader',
    # 'AgeAtVisitLoader'
]

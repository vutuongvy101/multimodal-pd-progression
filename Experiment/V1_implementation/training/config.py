"""
Configuration for V1 Multimodal Longitudinal Transformer
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import os
import torch


@dataclass
class FeatureConfig:
    """Configuration for feature groups"""

    # Static features (genetics + demographics)
    # Internal genetic feature components (use genetics_features property for public access)
    monogenic_variants_features: List[str] = field(default_factory=lambda: [
        'LRRK2', 'GBA', 'SNCA', 'PRKN', 'APOE', 'PATHVAR_COUNT', 'VAR_GENE'
    ])
    polygenic_features: List[str] = field(default_factory=lambda: [
        'GP2_PGS', 'META5_PGS', 'META5_excl_LRRK2_GBA_PGS',
    ])
    genetic_principal_components_features: List[str] = field(default_factory=lambda: [
        'Genetic_PRS_PC1', 'Genetic_PRS_PC2', 'Genetic_PRS_PC3', 'Genetic_PRS_PC4', 'Genetic_PRS_PC5',
        'Genetic_PRS_PC6', 'Genetic_PRS_PC7', 'Genetic_PRS_PC8', 'Genetic_PRS_PC9', 'Genetic_PRS_PC10',
    ])

    @property
    def genetics_features(self) -> List[str]:
        """Combined genetics features"""
        return (
                self.monogenic_variants_features +
                self.polygenic_features +
                self.genetic_principal_components_features
        )

    demographics_features: List[str] = field(default_factory=lambda: [
        # Education years
        'EDUCYRS',
        # Demographics
        'SEX', 'HANDED',
        # Descent
        'AFICBERB', 'ASHKJEW', 'BASQUE',
        # Sexuality
        'HOWLIVE', 'GAYLES', 'HETERO', 'BISEXUAL', 'PANSEXUAL', 'ASEXUAL', 'OTHSEXUALITY',
        # Ethnicity/Race
        'HISPLAT', 'RAASIAN', 'RABLACK', 'RAHAWOPI', 'RAINDALS', 'RANOS', 'RAWHITE', 'RAUNKNOWN',
        # Family history
        'ANYFAMPD',
        # 1st degree family
        'BIOMOM', 'BIOMOMPD', 'BIODAD', 'BIODADPD',
        'FULSIB', 'FULBRO', 'FULSIS', 'FULSIBPD', 'FULBROPD', 'FULSISPD',
        'KIDSPD',
        # 2nd degree family
        'HAFSIB', 'PAHAFSIB', 'MAHAFSIB', 'HAFSIBPD', 'MAHAFSIBPD',
        'PAHAFSIBPD', 'MAGPAR', 'MAGPARPD', 'MAGFATHPD', 'MAGMOTHPD', 'PAGPAR',
        'PAGPARPD', 'PAGFATHPD', 'PAGMOTHPD', 'MATAU', 'MATAUPD', 'PATAU',
        'PATAUPD', 'MATCOUS', 'MATCOUSPD',
        'PATCOUS', 'PATCOUSPD',
        'DISFAMPD'
    ])

    @property
    def static_features(self) -> List[str]:
        """Combined static features (genetics + demographics)"""
        return self.genetics_features + self.demographics_features

    part1_questionnaire_features: List[str] = field(default_factory=lambda: [
        'NP1SLPN', 'NP1SLPD', 'NP1URIN', 'NP1CNST', 'NP1LTHD', 'NP1PAIN', 'NP1FATG'
    ])

    part1_uprs_features: List[str] = field(default_factory=lambda: [
        'NP1DPRS', 'NP1ANXS', 'NP1APAT', 'NP1COG', 'NP1HALL', 'NP1DDS',
        'NP1RTOT'  # Part I total
    ])

    @property
    def part1_features(self) -> List[str]:
        """Part I - Non-motor experiences of daily living (combined questionnaire + UPDRS)"""
        return self.part1_questionnaire_features + self.part1_uprs_features

    # Part II - Motor experiences of daily living (patient-reported)
    part2_features: List[str] = field(default_factory=lambda: [
        'NP2SPCH', 'NP2SALV', 'NP2SWAL', 'NP2EAT', 'NP2DRES', 'NP2HYGN',
        'NP2HWRT', 'NP2HOBB', 'NP2TURN', 'NP2RISE', 'NP2WALK', 'NP2FREZ', 'NP2TRMR',
        'NP2PTOT'  # Part II total
    ])

    # Part III - Motor examination (clinician-observed)
    part3_features: List[str] = field(default_factory=lambda: [
        'NP3SPCH', 'NP3FACXP', 'NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL',
        'NP3FTAPR', 'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL',
        'NP3TTAPR', 'NP3TTAPL', 'NP3LGAGR', 'NP3LGAGL', 'NP3RISNG', 'NP3GAIT',
        'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR', 'NP3BRADY', 'NP3PTRMR', 'NP3PTRML',
        'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL',
        'NP3RTALJ', 'NP3RTCON',
        'NP3TOT'  # Part III total
    ])

    # Part IV - Motor complications
    part4_features: List[str] = field(default_factory=lambda: [
        'NP4WDYSK', 'NP4DYSKI', 'NP4OFF', 'NP4FLCTI', 'NP4FLCTX', 'NP4DYSTN',
        'NP4TOT'  # Part IV total
    ])

    # Convenience groupings for backward compatibility
    @property
    def motor_features(self) -> List[str]:
        """Combined motor features (Parts II + III + IV)

        Part II: Motor experiences of daily living (patient-reported)
        Part III: Motor examination (clinician-observed)
        Part IV: Motor complications (dyskinesia, OFF time, fluctuations, dystonia)
        """
        return self.part2_features + self.part3_features + self.part4_features

    @property
    def nonmotor_features(self) -> List[str]:
        """Combined non-motor features (Part I + other assessments)"""
        return self.part1_features + self.other_nonmotor_features

    @property
    def all_updrs_totals(self) -> List[str]:
        """All UPDRS total scores"""
        return ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']

    # Clinical assessment source column names
    # Cognitive
    moca_features: List[str] = field(default_factory=lambda: ['MCATOT'])
    # Sleep (Epworth Sleepiness Scale)
    ess_features: List[str] = field(
        default_factory=lambda: ['ESS1', 'ESS2', 'ESS3', 'ESS4', 'ESS5', 'ESS6', 'ESS7', 'ESS8'])
    # Autonomic function
    scopa_aut_features: List[str] = field(default_factory=lambda: [
        'SCAU1', 'SCAU2', 'SCAU3', 'SCAU4', 'SCAU5', 'SCAU6', 'SCAU7', 'SCAU8', 'SCAU9', 'SCAU10',
        'SCAU11', 'SCAU12', 'SCAU13', 'SCAU14', 'SCAU15', 'SCAU16', 'SCAU17', 'SCAU18', 'SCAU19', 'SCAU20',
        'SCAU21', 'SCAU22', 'SCAU23', 'SCAU24', 'SCAU25'])

    schwab_features: List[str] = field(default_factory=lambda: ['MSEADLG'])

    @property
    def other_nonmotor_features(self) -> List[str]:
        """Additional non-motor assessments (normalized output column names)"""
        return self.moca_features + self.ess_features + self.scopa_aut_features + self.schwab_features

    # Medication source column names
    # Levodopa equivalent daily dose
    ledd_features: List[str] = field(
        default_factory=lambda: ['LEDD', 'LED', 'LEDD_TOTAL', 'LEDDTOT'])  # LEDD preferred, then variants
    # ON=1, OFF=0
    pdmedyn_features: List[str] = field(
        default_factory=lambda: ['PDMEDYN', 'ON_OFF', 'PD_MED_USE'])  # PDMEDYN preferred, then alternatives

    # Medication context (normalized output column names)
    medication_features: List[str] = field(default_factory=lambda: [
        'PDMEDYN',
        'LEDD',
        'HOURS_SINCE_DOSE',  # If available
    ])


def calculate_mlp_dims(n_features: int, d_model: int = 256,
                       min_hidden: int = 64, max_hidden: int = 256) -> List[int]:
    """
    Dynamically calculate MLP dimensions based on number of features.
    
    Strategy:
    - Input dimension = n_features * 2 (values + missing masks)
    - First hidden layer: scales with input size
      * Small inputs (< 30): use min_hidden (64)
      * Medium inputs (30-50): use 128
      * Large inputs (> 50): use max_hidden (256) or 128
    - Second hidden layer: always d_model for transformer compatibility
    
    Args:
        n_features: Number of input features
        d_model: Output embedding dimension (default: 256)
        min_hidden: Minimum hidden layer size (default: 64)
        max_hidden: Maximum hidden layer size (default: 256)
    
    Returns:
        List of hidden layer dimensions [first_hidden, d_model]
    
    Examples:
        >>> calculate_mlp_dims(14)  # Small modality
        [64, 256]
        >>> calculate_mlp_dims(33)  # Large modality
        [128, 256]
        >>> calculate_mlp_dims(50)  # Very large
        [128, 256]
    """
    input_dim = n_features * 2  # values + masks

    # Determine first hidden layer size
    if input_dim < 30:
        first_hidden = min_hidden  # 64
    elif input_dim < 50:
        first_hidden = 128
    else:
        # For very large inputs, use 128 or max_hidden
        first_hidden = min(128, max_hidden)

    return [first_hidden, d_model]


@dataclass
class ModelConfig:
    """Configuration for V1 model architecture"""

    # Embedding dimensions
    d_model: int = 256

    # Transformer architecture
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    dim_feedforward: int = 1024
    activation: str = 'gelu'

    # Modality MLP dimensions
    # Set to None to auto-calculate from feature counts, or provide explicit dimensions
    # Auto-calculation uses: calculate_mlp_dims(n_features, d_model)
    #   - Small inputs (<30): [64, 256]
    #   - Medium inputs (30-50): [128, 256]  
    #   - Large inputs (>50): [128, 256]
    # 
    # Example: To override auto-calculation for static features:
    #   static_mlp_dims = [256, 512]  # Custom larger MLP
    static_mlp_dims: Optional[List[int]] = None
    part1_mlp_dims: Optional[List[int]] = None
    part2_mlp_dims: Optional[List[int]] = None
    part3_mlp_dims: Optional[List[int]] = None
    part4_mlp_dims: Optional[List[int]] = None
    med_mlp_dims: Optional[List[int]] = None
    other_mlp_dims: Optional[List[int]] = None

    # Internal reference to FeatureConfig (set by parent Config during initialization)
    _feature_config: Optional['FeatureConfig'] = None

    def get_mlp_dims(self, feature_config: Optional['FeatureConfig'] = None, modality: str = None) -> List[int]:
        """
        Get MLP dimensions for a modality, auto-calculating if not explicitly set.
        
        Uses self._feature_config (set by parent Config) as the source of truth.
        feature_config parameter is optional for backward compatibility.
        
        Args:
            feature_config: Optional FeatureConfig (deprecated - uses self._feature_config)
            modality: One of 'static', 'part1', 'part2', 'part3', 'part4', 'med', 'other'
        
        Returns:
            List of MLP hidden layer dimensions
        """
        # Use stored feature_config (from parent Config) as single source of truth
        # feature_config parameter kept for backward compatibility only
        if feature_config is None:
            feature_config = self._feature_config

        if feature_config is None:
            raise ValueError("feature_config not available. Must be set in parent Config.")

        if modality is None:
            raise ValueError("modality parameter is required")

        # Map modality names to config attributes and feature lists
        modality_map = {
            'static': ('static_mlp_dims', feature_config.static_features),
            'part1': ('part1_mlp_dims', feature_config.part1_features),
            'part2': ('part2_mlp_dims', feature_config.part2_features),
            'part3': ('part3_mlp_dims', feature_config.part3_features),
            'part4': ('part4_mlp_dims', feature_config.part4_features),
            'med': ('med_mlp_dims', feature_config.medication_features),
            'other': ('other_mlp_dims', feature_config.other_nonmotor_features),
        }

        if modality not in modality_map:
            raise ValueError(f"Unknown modality: {modality}. Must be one of {list(modality_map.keys())}")

        attr_name, feature_list = modality_map[modality]
        explicit_dims = getattr(self, attr_name)

        # If explicitly set, use it
        if explicit_dims is not None:
            return explicit_dims

        # Otherwise, auto-calculate from feature count
        n_features = len(feature_list)
        return calculate_mlp_dims(n_features, self.d_model)

    # Backward compatibility methods (no longer require feature_config)
    def motor_mlp_dims(self, feature_config: Optional['FeatureConfig'] = None) -> List[int]:
        """Get motor MLP dims (uses part3)"""
        # Uses self._feature_config (single source of truth) - feature_config param for backward compat
        if self.part3_mlp_dims is not None:
            return self.part3_mlp_dims
        return self.get_mlp_dims(modality='part3')

    def nonmotor_mlp_dims(self, feature_config: Optional['FeatureConfig'] = None) -> List[int]:
        """Get non-motor MLP dims (uses part1)"""
        # Uses self._feature_config (single source of truth) - feature_config param for backward compat
        if self.part1_mlp_dims is not None:
            return self.part1_mlp_dims
        return self.get_mlp_dims(modality='part1')

    # Prediction targets
    predict_totals: List[str] = field(default_factory=lambda: ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT'])
    # All UPDRS totals: NP1RTOT (non-motor), NP2PTOT (motor ADL), NP3TOT (motor exam), NP4TOT (complications)

    # Prediction heads
    next_visit_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    slope_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])

    # Time encoding
    max_time_months: int = 120  # 10 years

    # Sequence handling
    max_seq_len: int = 20


def get_device() -> str:
    """Dynamically detect and return the best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


@dataclass
class TrainingConfig:
    """Configuration for training"""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100

    # Loss weights
    lambda_slope: float = 0.2  # Weight for slope prediction loss

    # Early stopping
    early_stopping_patience: int = 15

    # Data
    min_visits_for_slope: int = 3
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Logging
    log_every_n_steps: int = 10
    validate_every_n_epochs: int = 1

    # Device (automatically detected: 'cuda', 'mps' for Mac, or 'cpu')
    device: str = field(default_factory=get_device)


@dataclass
class DataConfig:
    """Configuration for data paths"""

    _repo_root = Path(__file__).parent.parent.parent.parent
    base_dir: str = str(_repo_root / "ppmi_pd")

    # Input files
    participant_status: str = "Participant_Status_14Dec2025.csv"
    demographics: str = "Subject_Demographics/Demographics_14Dec2025.csv"
    family_history: str = "Family_History_14Dec2025.csv"
    socio_economic: str = "Subject_Demographics/Socio-Economics_14Dec2025.csv"
    age_at_visit: str = "Subject_Demographics/Age_at_Visit_14Dec2025.csv"

    # Genetics files (relative to base_dir)
    genetic_consensus: str = "Genetic_Status/iu_genetic_consensus_20251025_14Dec2025.csv"
    prs_scores: str = "Genetic_Status/Polygenic_Risk_Scores_14Dec2025.csv"
    prs_pcs: str = "Genetic_Status/PPMI_Project_9001_20250624_14Dec2025.csv"

    # Clinical files (to be added based on your data structure)
    updrs_part1_ques: str = "Motor___MDS-UPDRS/MDS-UPDRS_Part_I_Patient_Questionnaire_14Dec2025.csv"
    updrs_part1: str = "Motor___MDS-UPDRS/MDS-UPDRS_Part_I_14Dec2025.csv"
    updrs_part2: str = "Motor___MDS-UPDRS/MDS_UPDRS_Part_II__Patient_Questionnaire_14Dec2025.csv"
    updrs_part3: str = "Motor___MDS-UPDRS/MDS-UPDRS_Part_III_14Dec2025.csv"
    updrs_part4: str = "Motor___MDS-UPDRS/MDS-UPDRS_Part_IV__Motor_Complications_14Dec2025.csv"

    # Non-motor clinical assessments
    moca: str = "Non-motor_Assessments/Montreal_Cognitive_Assessment__MoCA__14Dec2025.csv"
    ess: str = "Non-motor_Assessments/Epworth_Sleepiness_Scale_14Dec2025.csv"
    scopa_aut: str = "Non-motor_Assessments/SCOPA-AUT_14Dec2025.csv"
    schwab_england: str = "Motor___MDS-UPDRS/Modified_Schwab___England_Activities_of_Daily_Living_14Dec2025.csv"

    # Output paths (relative to V1_implementation directory)
    processed_data_dir: str = "data/processed"
    model_save_dir: str = "models/checkpoints"
    results_dir: str = "results"


@dataclass
class Config:
    """Complete configuration object for V1 model"""
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    def __post_init__(self):
        """Validate configuration after initialization and set up internal references"""
        # Set feature_config reference in model_config so it can access features without passing config around
        self.model._feature_config = self.features
        # Ensure MLP dimensions are compatible
        pass


# Create default configs
def get_default_config() -> Config:
    """Get default configuration for V1 model"""
    return Config(
        features=FeatureConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig()
    )


if __name__ == "__main__":
    # Print configuration
    config = get_default_config()

    print("=" * 80)
    print("V1 Model Configuration")
    print("=" * 80)

    print("\n--- Feature Configuration ---")
    print(f"Static features: {len(config.features.static_features)}")
    print(f"Part I (non-motor): {len(config.features.part1_features)}")
    print(f"Part II (motor ADL): {len(config.features.part2_features)}")
    print(f"Part III (motor exam): {len(config.features.part3_features)}")
    print(f"Part IV (complications): {len(config.features.part4_features)}")
    print(f"Other non-motor: {len(config.features.other_nonmotor_features)}")
    print(f"Medication context: {len(config.features.medication_features)}")
    print(f"UPDRS totals: {', '.join(config.features.all_updrs_totals)}")

    print("\n--- Model Configuration ---")
    print(f"d_model: {config.model.d_model}")
    print(f"n_heads: {config.model.n_heads}")
    print(f"n_layers: {config.model.n_layers}")
    print(f"max_seq_len: {config.model.max_seq_len}")

    print("\n--- MLP Dimensions (Auto-calculated) ---")
    modality_map = {
        'static': 'static_features',
        'part1': 'part1_features',
        'part2': 'part2_features',
        'part3': 'part3_features',
        'part4': 'part4_features',
        'med': 'medication_features',
        'other': 'other_nonmotor_features',
    }
    for mod, attr_name in modality_map.items():
        n_features = len(getattr(config.features, attr_name))
        mlp_dims = config.model.get_mlp_dims(modality=mod)
        print(f"{mod:8s}: {n_features:3d} features → {mlp_dims}")

    print("\n--- Training Configuration ---")
    print(f"batch_size: {config.training.batch_size}")
    print(f"learning_rate: {config.training.learning_rate}")
    print(f"lambda_slope: {config.training.lambda_slope}")
    print(f"max_epochs: {config.training.max_epochs}")

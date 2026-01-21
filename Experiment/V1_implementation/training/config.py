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

    # Age at visit
    age_at_visit_features: List[str] = field(default_factory=lambda: [
        'AGE_AT_VISIT'
    ])

    # Demographics - Socioeconomic Status
    socioeconomic_features: List[str] = field(default_factory=lambda: [
        'EDUCYRS'  # Education years
    ])

    # Demographics - Basic Demographics
    basic_demographics_features: List[str] = field(default_factory=lambda: [
        'SEX', 'HANDED',
        # Descent
        'AFICBERB', 'ASHKJEW', 'BASQUE',
        # Sexuality
        'HOWLIVE', 'GAYLES', 'HETERO', 'BISEXUAL', 'PANSEXUAL', 'ASEXUAL', 'OTHSEXUALITY',
        # Ethnicity/Race
        'HISPLAT', 'RAASIAN', 'RABLACK', 'RAHAWOPI', 'RAINDALS', 'RANOS', 'RAWHITE', 'RAUNKNOWN'
    ])

    # Demographics - Family History
    family_history_features: List[str] = field(default_factory=lambda: [
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
    def demographics_features(self) -> List[str]:
        """Combined demographics features (socioeconomic + basic + family history)"""
        return (
            self.socioeconomic_features +
            self.basic_demographics_features +
            self.family_history_features
        )

    @property
    def static_features(self) -> List[str]:
        """Combined static features (genetics + demographics)"""
        return self.genetics_features + self.demographics_features

    # UPDRS features
    # Part I - Non-motor experiences of daily living
    part1_updrs_features: List[str] = field(default_factory=lambda: [
        'NP1DPRS', 'NP1ANXS', 'NP1APAT',  # Mood and Apathy
        'NP1COG', 'NP1HALL', 'NP1DDS', # Cognitive & Psychosis
        'NP1RTOT'  # Part I total
    ])

    part1_questionnaire_features: List[str] = field(default_factory=lambda: [
        'NP1SLPN', 'NP1SLPD', # Sleep
        'NP1URIN', 'NP1CNST', 'NP1LTHD', # Autonomic
        'NP1PAIN', 'NP1FATG' # Sensory Fatigue
    ])

    @property
    def part1_features(self) -> List[str]:
        """Part I - Non-motor experiences of daily living (combined questionnaire + UPDRS)"""
        return  self.part1_updrs_features + self.part1_questionnaire_features

    # Part II - Motor experiences of daily living (patient-reported)
    part2_features: List[str] = field(default_factory=lambda: [
        'NP2SPCH', 'NP2SALV', 'NP2SWAL', # Bulbar
        'NP2EAT', 'NP2DRES', 'NP2HYGN', 'NP2HWRT', 'NP2HOBB', # Fine Motor
        'NP2TURN', 'NP2RISE', 'NP2WALK', 'NP2FREZ', # Mobility
        'NP2TRMR', # Tremor
        'NP2PTOT'  # Part II total
    ])

    # Part III - Motor examination (clinician-observed)
    part3_features: List[str] = field(default_factory=lambda: [
        #  Bulbar
        'NP3SPCH', 'NP3FACXP',
        # Rigidity
        'NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL',
        # Bradykinesia
        'NP3FTAPR', 'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL',
        'NP3TTAPR', 'NP3TTAPL', 'NP3LGAGR', 'NP3LGAGL',
        # PIGD
        'NP3RISNG', 'NP3GAIT', 'NP3BRADY',
        'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR',
        'NP3PTRMR', 'NP3PTRML',
        # Tremor
        'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL',
        'NP3RTALL','NP3RTALJ', 'NP3RTCON',
        # Hoehn & Yahr Stage
        'NHY',
        'NP3TOT'  # Part III total
    ])

    # Part IV - Motor complications
    part4_features: List[str] = field(default_factory=lambda: [
        # Fluctuations
        'NP4WDYSK', 'NP4DYSKI',
        # Dyskinesias
        'NP4OFF', 'NP4FLCTI', 'NP4FLCTX',
        # Dystonia
        'NP4DYSTN',
        'NP4TOT'  # Part IV total
    ])

    # Additional motor assessment features (supplementary to UPDRS)
    schwab_england_features: List[str] = field(default_factory=lambda: [
        'MSEADLG'  # Modified Schwab & England ADL scale
    ])

    neuro_qol_lower_features: List[str] = field(default_factory=lambda: [
        'NQMOB37', 'NQMOB30', 'NQMOB26', 'NQMOB32', 'NQMOB25', 'NQMOB33', 'NQMOB31', 'NQMOB28'
    ])

    neuro_qol_upper_features: List[str] = field(default_factory=lambda: [
        'NQUEX29', 'NQUEX20', 'NQUEX44', 'NQUEX36', 'NQUEX30', 'NQUEX28', 'NQUEX33', 'NQUEX37'
    ])

    participant_motor_features: List[str] = field(default_factory=lambda: [
        'TRBUPCHR', 'WRTSMLR', 'VOICSFTR', 'POORBAL', 'FTSTUCK', 'LSSXPRSS', 'ARMLGSHK',
        'TRBBUTTN', 'SHUFFLE', 'MVSLOW', 'TOLDPD'
    ])

    # Convenience groupings for backward compatibility
    @property
    def motor_features(self) -> List[str]:
        """Combined motor features (Parts II + III + IV)

        Part II: Motor experiences of daily living (patient-reported)
        Part III: Motor examination (clinician-observed)
        Part IV: Motor complications (dyskinesia, OFF time, fluctuations, dystonia)
        Part I: Non-motor experiences of daily living (for completeness)
        Schwab & England, Neuro QoL, Participant Motor Function are supplementary motor assessments
        """
        return (self.part2_features + self.part3_features + self.part4_features + self.part1_features)
    
    # UPDRS loader supplementary features (used for comprehensive motor assessment)
    @property
    def updrs_supplementary_features(self) -> List[str]:
        """All supplementary features loaded by UPDRS loader (beyond core UPDRS parts)"""
        return (
            self.schwab_england_features +
            self.neuro_qol_lower_features +
            self.neuro_qol_upper_features +
            self.participant_motor_features
        )

    @property
    def all_updrs_totals(self) -> List[str]:
        """All UPDRS total scores"""
        return ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']


    # Non-motor assessment source column names
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

    @property
    def non_motor_features(self) -> List[str]:
        """Additional non-motor assessments (normalized output column names)"""
        return self.moca_features + self.ess_features + self.scopa_aut_features


    # Medication source column names
    # Levodopa equivalent daily dose
    ledd_features: List[str] = field(
        default_factory=lambda: ['LEDTRT', 'STARTDT', 'STOPDT', 'LEDD'])
    
    # Vital signs
    vital_signs_features: List[str] = field(
        default_factory=lambda: ['SYSSUP', 'DIASUP', 'SYSSTND', 'DIASTND', 'HRSUP', 'HRSTND', 'WGTKG', 'HTCM'])  

    # PD diagnosis history
    pd_diagnosis_features: List[str] = field(default_factory=lambda: [
        'PDDXDT', 'SXDT', 'DXTREMOR', 'DXRIGID', 'DXBRADY', 'DOMSIDE'
    ])

    @property
    def medication_features(self) -> List[str]:
        """Combined medication features (LEDD + vital signs + PD diagnosis)"""
        return self.ledd_features + self.vital_signs_features + self.pd_diagnosis_features


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
    activation: str = 'gelu' # this is told better than RELU in Transformer

    # Modality selection: which modalities to include in the model
    # Options: 'static', 'motor', 'nonmotor', 'medication'
    # This enables ablation studies and modality-specific experiments
    enabled_modalities: List[str] = field(default_factory=lambda: [
        'static', 'motor', 'non_motor', 'medication', 'age_at_visit'
    ])

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
    motor_mlp_dims: Optional[List[int]] = None
    non_motor_mlp_dims: Optional[List[int]] = None
    med_mlp_dims: Optional[List[int]] = None
    age_at_mlp_dims: Optional[List[int]] = None
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
            'motor': ('motor_mlp_dims', feature_config.motor_features),
            'med': ('med_mlp_dims', feature_config.medication_features),
            'non_motor': ('non_motor_mlp_dims', feature_config.non_motor_features),
            'age_at_visit': ('age_at_mlp_dims', feature_config.age_at_visit_features),
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

    # Prediction targets:
    # All UPDRS totals: NP1RTOT (non-motor), NP2PTOT (motor ADL), NP3TOT (motor exam), NP4TOT (complications)
    predict_totals: List[str] = field(default_factory=lambda: ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT'])

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

    # Participant Status (master_df)
    participant_status: str = "Participant_Status_14Dec2025.csv"

    # Demographics files
    demographics: str = "Subject_Demographics/Demographics_14Dec2025.csv"
    family_history: str = "Family_History_14Dec2025.csv"
    socio_economic: str = "Subject_Demographics/Socio-Economics_14Dec2025.csv"
    age_at_visit: str = "Subject_Demographics/Age_at_Visit_14Dec2025.csv"

    # Genetics files (relative to base_dir)
    genetic_consensus: str = "Genetic_Status/iu_genetic_consensus_20251025_14Dec2025.csv"
    prs_scores: str = "Genetic_Status/Polygenic_Risk_Scores_14Dec2025.csv"
    prs_pcs: str = "Genetic_Status/PPMI_Project_9001_20250624_14Dec2025.csv"

    # Motor clinical files (to be added based on your data structure)
    updrs_part1_ques: str = "Motor___MDS-UPDRS/MDS-UPDRS_Part_I_Patient_Questionnaire_14Dec2025.csv"
    updrs_part1: str = "Motor___MDS-UPDRS/MDS-UPDRS_Part_I_14Dec2025.csv"
    updrs_part2: str = "Motor___MDS-UPDRS/MDS_UPDRS_Part_II__Patient_Questionnaire_14Dec2025.csv"
    updrs_part3: str = "Motor___MDS-UPDRS/MDS-UPDRS_Part_III_14Dec2025.csv"
    updrs_part4: str = "Motor___MDS-UPDRS/MDS-UPDRS_Part_IV__Motor_Complications_14Dec2025.csv"
    schwab_england: str = "Motor___MDS-UPDRS/Modified_Schwab___England_Activities_of_Daily_Living_14Dec2025.csv"
    neuro_qol_lower: str = "Motor___MDS-UPDRS/Neuro_QoL__Lower_Extremity_Function__Mobility__-_Short_Form_14Dec2025.csv"
    neuro_qol_upper: str = "Motor___MDS-UPDRS/Neuro_QoL__Upper_Extremity_Function_-_Short_Form_14Dec2025.csv"
    participant_motor: str = "Motor___MDS-UPDRS/Participant_Motor_Function_Questionnaire_14Dec2025.csv"

    # Non-motor assessment files
    moca: str = "Non-motor_Assessments/Montreal_Cognitive_Assessment__MoCA__14Dec2025.csv"
    ess: str = "Non-motor_Assessments/Epworth_Sleepiness_Scale_14Dec2025.csv"
    scopa_aut: str = "Non-motor_Assessments/SCOPA-AUT_14Dec2025.csv"

    # Medication files
    ledd: str = "Medical_History/LEDD_Concomitant_Medication_Log_14Dec2025.csv"
    vital_signs: str = "Medical_History/Vital_Signs_14Dec2025.csv"
    pd_diagnosis: str = "Medical_History/PD_Diagnosis_History_14Dec2025.csv"

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
    # When True, loaders and utilities should raise on missing/critical errors
    # (useful during development / CI). When False, loaders may return empty
    # DataFrames for optional files and log errors instead of raising.
    raise_on_error: bool = True

    def __post_init__(self):
        """Validate configuration after initialization and set up internal references"""
        # Set feature_config reference in model_config so it can access features without passing config around
        self.model._feature_config = self.features
        # Ensure MLP dimensions are compatible
        # Make flag visible via model/training if needed in runtime
        try:
            # Attach flag to training for backward compatibility checks
            setattr(self.training, 'raise_on_error', self.raise_on_error)
        except Exception:
            # Non-fatal: only a convenience mapping
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
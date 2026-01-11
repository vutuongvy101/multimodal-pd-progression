# Complete Data Attributes Documentation

This document provides a comprehensive catalog of all data attributes (columns/variables) across the **parkinsons_tele** and **ppmi_pd** datasets.

---

## Table of Contents

1. [Parkinson's Telemonitoring Dataset (parkinsons_tele)](#parkinsons-telemonitoring-dataset)
2. [PPMI PD Dataset (ppmi_pd)](#ppmi-pd-dataset)
   - [Biosample Inventory](#biosample-inventory)
   - [Data   Databases](#data---databases)
   - [Follow Up persons w Neurologic Disease](#follow-up-persons-w-neurologic-disease)
   - [Genetic Status](#genetic-status)
   - [Imaging](#imaging)
   - [Medical History](#medical-history)
   - [Motor   MDS-UPDRS](#motor---mds-updrs)
   - [Non-motor Assessments](#non-motor-assessments)
   - [PPMI Online](#ppmi-online)
   - [PPMI Remote Screening](#ppmi-remote-screening)
   - [Roche Smartphone App](#roche-smartphone-app)
   - [Subject Demographics](#subject-demographics)

---

## Parkinson's Telemonitoring Dataset

**Location:** `Database data/parkinsons_tele/`

### Dataset: `parkinsons_updrs.data`

**Description:** Oxford Parkinson's Disease Telemonitoring Dataset containing biomedical voice measurements from 42 people with early-stage Parkinson's disease. Contains 5,875 voice recordings with 22 attributes.

**Attributes:**

| Attribute Name | Type | Description |
|---------------|------|-------------|
| `subject#` | Integer | Unique identifier for each subject (1-42) |
| `age` | Integer | Subject age in years |
| `sex` | Integer | Subject gender: 0 = male, 1 = female |
| `test_time` | Float | Time since recruitment (days). Integer part = days since recruitment |
| `motor_UPDRS` | Float | Clinician's motor UPDRS score, linearly interpolated |
| `total_UPDRS` | Float | Clinician's total UPDRS score, linearly interpolated |
| `Jitter(%)` | Float | Percentage of variation in fundamental frequency (cycle-to-cycle pitch variation) |
| `Jitter(Abs)` | Float | Absolute jitter measure |
| `Jitter:RAP` | Float | Relative Average Perturbation (RAP) - jitter measure |
| `Jitter:PPQ5` | Float | Pitch Period Perturbation Quotient (5-point) - jitter measure |
| `Jitter:DDP` | Float | Difference of Differences of Periods - jitter measure |
| `Shimmer` | Float | Variation in amplitude (amplitude variation) |
| `Shimmer(dB)` | Float | Shimmer in decibels |
| `Shimmer:APQ3` | Float | Amplitude Perturbation Quotient (3-point) - shimmer measure |
| `Shimmer:APQ5` | Float | Amplitude Perturbation Quotient (5-point) - shimmer measure |
| `Shimmer:APQ11` | Float | Amplitude Perturbation Quotient (11-point) - shimmer measure |
| `Shimmer:DDA` | Float | Difference of Differences of Amplitudes - shimmer measure |
| `NHR` | Float | Noise-to-Harmonics Ratio - measure of ratio of noise to tonal components |
| `HNR` | Float | Harmonics-to-Noise Ratio - measure of ratio of tonal to noise components |
| `RPDE` | Float | Recurrence Period Density Entropy - nonlinear dynamical complexity measure (vocal fold stability) |
| `DFA` | Float | Detrended Fluctuation Analysis - signal fractal scaling exponent (breathiness/turbulent noise) |
| `PPE` | Float | Pitch Period Entropy - nonlinear measure of fundamental frequency variation (impaired pitch control, robust to vibrato) |

**Key Relationships:**
- Each row = one voice recording
- ~200 recordings per patient
- Target variables: `motor_UPDRS` and `total_UPDRS`
- Predictor variables: 16 biomedical voice measures (Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE)

---

## PPMI PD Dataset

**Location:** `Database data/ppmi_pd/`

The PPMI (Parkinson's Progression Markers Initiative) dataset is a comprehensive longitudinal study with multiple data domains. Key identifier: **`PATNO`** (Participant Number) and **`EVENT_ID`** (Visit ID, e.g., BL=Baseline, V01-V24=Visits, R01-R24=Remote visits).

**Updated Structure (December 2025):**
- **198 CSV files** across **13 major categories**
- **5,081 total attributes** across all datasets
- Comprehensive coverage of clinical, imaging, genetic, biosample, and digital health data

---

### Biosample Inventory

**Location:** `ppmi_pd/Biosample_Inventory/`

**Description:** Biological sample catalog and availability.

**Number of Files:** 5

**Key Files:**

- `IUSM_ASSAY_DEV_CATALOG_14Dec2025.csv` - 19 attributes, 5615 rows
- `IUSM_BIOSPECIMEN_CELL_CATALOG_14Dec2025.csv` - 28 attributes, 486 rows
- `IUSM_CATALOG_14Dec2025.csv` - 19 attributes, 131616 rows
- `Whole_Blood_Substudy_14Dec2025.csv` - 65 attributes, 139 rows
- `iPSC_Catalog_Metadata_14Dec2025.csv` - 18 attributes, 151 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `COHORT` | 4 files | Common identifier or metadata field |
| `CLINICAL_EVENT` | 4 files | Common identifier or metadata field |
| `SPECIMEN_NO` | 3 files | Common identifier or metadata field |
| `ALIAS_ID` | 3 files | Common identifier or metadata field |
| `TYPE` | 3 files | Common identifier or metadata field |
| `NUM_AVAILABLE` | 3 files | Common identifier or metadata field |
| `QUANTITY` | 3 files | Common identifier or metadata field |
| `QUANTITY_UNITS` | 3 files | Common identifier or metadata field |
| `ADDTL_STOCK_AVAIL_ON_REQ` | 3 files | Common identifier or metadata field |
| `MASS_UG` | 2 files | Common identifier or metadata field |

---

### Data   Databases

**Location:** `ppmi_pd/Data___Databases/`

**Description:** Data dictionaries and code lists.

**Number of Files:** 5

**Key Files:**

- `Code_List_-_Harmonized_14Dec2025.csv` - 4 attributes, 8840 rows
- `Code_List_-__Annotated__14Dec2025.csv` - 5 attributes, 8840 rows
- `Data_Dictionary_-_Harmonized_14Dec2025.csv` - 9 attributes, 7073 rows
- `Data_Dictionary_-__Annotated__14Dec2025.csv` - 13 attributes, 7073 rows
- `Deprecated_Variables_14Dec2025.csv` - 7 attributes, 917 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `MOD_NAME` | 5 files | Common identifier or metadata field |
| `ITM_NAME` | 5 files | Common identifier or metadata field |
| `MAPPING_NOTES` | 3 files | Common identifier or metadata field |
| `DSCR` | 3 files | Common identifier or metadata field |
| `CODE` | 2 files | Common identifier or metadata field |
| `DECODE` | 2 files | Common identifier or metadata field |
| `PAG_NAME` | 2 files | Common identifier or metadata field |
| `ITM_TYPE` | 2 files | Common identifier or metadata field |
| `FLD_LEN` | 2 files | Common identifier or metadata field |
| `DECML` | 2 files | Common identifier or metadata field |

---

### Follow Up persons w Neurologic Disease

**Location:** `ppmi_pd/Follow_Up_persons_w_Neurologic_Disease/`

**Description:** FOUND study questionnaires and assessments.

**Number of Files:** 10

**Key Files:**

- `FOUND_RFQ_Alcohol_14Dec2025.csv` - 36 attributes, 683 rows
- `FOUND_RFQ_Anti-Inflammatory_Meds_14Dec2025.csv` - 46 attributes, 683 rows
- `FOUND_RFQ_Caffeine_14Dec2025.csv` - 243 attributes, 683 rows
- `FOUND_RFQ_Calcium_Channel_Blockers_14Dec2025.csv` - 150 attributes, 683 rows
- `FOUND_RFQ_Female_Reproductive_Health_14Dec2025.csv` - 38 attributes, 683 rows
- `FOUND_RFQ_Head_Injury_14Dec2025.csv` - 54 attributes, 683 rows
- `FOUND_RFQ_Height___Weight_14Dec2025.csv` - 30 attributes, 683 rows
- `FOUND_RFQ_Physical_Activity_14Dec2025.csv` - 29 attributes, 683 rows
- `FOUND_RFQ_Smoking_History_14Dec2025.csv` - 40 attributes, 683 rows
- `FOUND_Self-Reported_Dx_14Dec2025.csv` - 6 attributes, 17292 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `patno` | 9 files | Common identifier or metadata field |

---

### Genetic Status

**Location:** `ppmi_pd/Genetic_Status/`

**Description:** Genetic data, variants, and polygenic risk scores.

**Number of Files:** 5

**Key Files:**

- `Genotypes_for_Polygenic_Risk_Scores_14Dec2025.csv` - 9 attributes, 214560 rows
- `PPMI_PD_Variants_Genetic_Status_WGS_20180921.csv` - 73 attributes, 960 rows
- `PPMI_Project_9001_20250624_14Dec2025.csv` - 109 attributes, 1319 rows
- `Polygenic_Risk_Scores_14Dec2025.csv` - 17 attributes, 2996 rows
- `iu_genetic_consensus_20251025_14Dec2025.csv` - 21 attributes, 6265 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `PATNO` | 5 files | Participant ID (primary key) |

---

### Imaging

**Location:** `ppmi_pd/Imaging/`

**Description:** Neuroimaging data (MRI, PET, DaTSCAN, DTI, etc.).

**Number of Files:** 27

**Key Files:**

- `AV-133_Prodromal_Substudy_VMAT-2_Imaging_14Dec2025.csv` - 21 attributes, 77 rows
- `C05-05_PET_Imaging_Substudy_Imaging_14Dec2025.csv` - 21 attributes, 10 rows
- `CT_Scan_14Dec2025.csv` - 8 attributes, 341 rows
- `DTI_Regions_of_Interest_14Dec2025.csv` - 14 attributes, 1200 rows
- `DaTscan_Imaging_14Dec2025.csv` - 17 attributes, 13469 rows
- `Dual_PET_AV-133_in_PD_Imaging_Substudy_AV-133_PET_Imaging_14Dec2025.csv` - 21 attributes, 3 rows
- `Dual_PET_AV133_in_PI_Substudy_AV-133_NX_PET_Imaging_14Dec2025.csv` - 9 attributes, 9 rows
- `Early_Imaging_AV-133_Imaging_14Dec2025.csv` - 21 attributes, 207 rows
- `FS7_APARC_CTH_14Dec2025.csv` - 72 attributes, 1716 rows
- `FS7_APARC_SA_14Dec2025.csv` - 72 attributes, 1716 rows
- `FS7_ASEG_VOL_14Dec2025.csv` - 66 attributes, 1713 rows
- `Grey_Matter_Volume_14Dec2025.csv` - 6 attributes, 363 rows
- `MRIQC_14Dec2025.csv` - 58 attributes, 1740 rows
- `Magnetic_Resonance_Imaging__MRI__14Dec2025.csv` - 13 attributes, 9966 rows
- `NX_PI-2620_Tau_Imaging_Substudy_Imaging_14Dec2025.csv` - 21 attributes, 11 rows
- `PET_Acquisition_Metadata_14Dec2025.csv` - 23 attributes, 214 rows
- `PET_SBR_Analysis_14Dec2025.csv` - 28 attributes, 214 rows
- `SV2A_PET_Acquisition_Metadata_14Dec2025.csv` - 17 attributes, 14 rows
- `SV2A_PET_Imaging_Substudy_Imaging_14Dec2025.csv` - 20 attributes, 18 rows
- `SV2A_PET_SUVR_Analysis_14Dec2025.csv` - 9 attributes, 1232 rows
- `TAU_PET_Analysis_14Dec2025.csv` - 9 attributes, 18 rows
- `TAU_PET_Metadata_14Dec2025.csv` - 16 attributes, 18 rows
- `Tau_Substudy_MK-6240_PET_Imaging_14Dec2025.csv` - 20 attributes, 23 rows
- `Xing_Core_Lab_-_DATSCAN_acquisition_metadata_14Dec2025.csv` - 12 attributes, 4385 rows
- `Xing_Core_Lab_-_MRI_acquisition_metadata_14Dec2025.csv` - 10 attributes, 7079 rows
- `Xing_Core_Lab_-_Quant_SBR_14Dec2025.csv` - 42 attributes, 3487 rows
- `Xing_Core_Lab_-_Visual_Read_14Dec2025.csv` - 5 attributes, 1839 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `PATNO` | 27 files | Participant ID (primary key) |
| `EVENT_ID` | 25 files | Visit identifier |
| `PAG_NAME` | 12 files | Common identifier or metadata field |
| `REC_ID` | 11 files | Record ID (unique identifier) |
| `INFODT` | 11 files | Information/assessment date |
| `ORIG_ENTRY` | 11 files | Common identifier or metadata field |
| `LAST_UPDATE` | 11 files | Common identifier or metadata field |
| `PROTOCOL` | 10 files | Common identifier or metadata field |
| `SUB_EVENT_ID` | 9 files | Common identifier or metadata field |
| `INVEVLBF` | 7 files | Common identifier or metadata field |

---

### Medical History

**Location:** `ppmi_pd/Medical_History/`

**Description:** Medical conditions, adverse events, physical exams, and clinical assessments.

**Number of Files:** 52

**Key Files:**

- `AV-133_Prodromal_Substudy_Adverse_Event_In-Clinic_Assessment_14Dec2025.csv` - 11 attributes, 76 rows
- `AV-133_Prodromal_Substudy_Adverse_Event_Telephone_Assessment_14Dec2025.csv` - 12 attributes, 75 rows
- `AV-133_Prodromal_Substudy_Pregnancy_Test_14Dec2025.csv` - 12 attributes, 70 rows
- `Adverse_Event_In-Clinic_Assessment_14Dec2025.csv` - 13 attributes, 11292 rows
- `Adverse_Event_Log_14Dec2025.csv` - 23 attributes, 3235 rows
- `Adverse_Event_Telephone_Assessment_14Dec2025.csv` - 12 attributes, 31789 rows
- `C05-05_PET_Imaging_Substudy_Adverse_Event_In-Clinic_Assessment_14Dec2025.csv` - 11 attributes, 10 rows
- `C05-05_PET_Imaging_Substudy_Adverse_Event_Telephone_Assessment_14Dec2025.csv` - 12 attributes, 10 rows
- `C05-05_PET_Imaging_Substudy_Pregnancy_Test_14Dec2025.csv` - 12 attributes, 10 rows
- `Clinical_Diagnosis_14Dec2025.csv` - 11 attributes, 13292 rows
- `Clinical_Global_Impression__CGI__-_Investigator_14Dec2025.csv` - 8 attributes, 6738 rows
- `Concomitant_Medication_Log_14Dec2025.csv` - 22 attributes, 58373 rows
- `DPA-714_PET_Imaging_Substudy_Adverse_Event_In-Clinic_Assessment_14Dec2025.csv` - 11 attributes, 26 rows
- `DPA-714_PET_Imaging_Substudy_Adverse_Event_Telephone_Assessment_14Dec2025.csv` - 12 attributes, 26 rows
- `DPA-714_PET_Imaging_Substudy_Genetic_Testing_for_TSPO_Gene_14Dec2025.csv` - 11 attributes, 77 rows
- `Determination_of_Freezing_and_Falls_14Dec2025.csv` - 25 attributes, 11381 rows
- `Dual_PET_AV-133_PD_Imaging_Substdy_AE_In-Clinic_Assessment_14Dec2025.csv` - 11 attributes, 3 rows
- `Dual_PET_AV-133_in_PD_Imaging_Substudy_Pregnancy_Test_14Dec2025.csv` - 12 attributes, 2 rows
- `Dual_PET_AV-133_in_PI_Substudy_AE_Telephone_Assessment_14Dec2025.csv` - 12 attributes, 9 rows
- `Dual_PET_AV133_PI_Substdy_AE_In-Clinic_Assessment_14Dec2025.csv` - 11 attributes, 9 rows
- `Dual_PET_AV133_in_PD_Imag_Substudy_AE_Telephone_Assessment_14Dec2025.csv` - 12 attributes, 3 rows
- `Early_Imaging_Adverse_Event_In-Clinic_Assessment_14Dec2025.csv` - 11 attributes, 34 rows
- `Early_Imaging_Adverse_Event_Telephone_Assessment_14Dec2025.csv` - 12 attributes, 118 rows
- `Early_Imaging_Pregnancy_Test_14Dec2025.csv` - 13 attributes, 119 rows
- `Early_Imaging_Report_of_Pregnancy_14Dec2025.csv` - 10 attributes, 0 rows
- `Early_Imaging_Substudy_ECG_14Dec2025.csv` - 18 attributes, 40 rows
- `Features_of_Parkinsonism_14Dec2025.csv` - 12 attributes, 24609 rows
- `Features_of_REM_Behavior_Disorder_14Dec2025.csv` - 13 attributes, 4513 rows
- `Gait_Substudy_Adverse_Event_In-Clinic_Assessment_14Dec2025.csv` - 11 attributes, 359 rows
- `Gait_Substudy_Adverse_Event_Telephone_Assessment_14Dec2025.csv` - 11 attributes, 348 rows
- `General_Physical_Exam_14Dec2025.csv` - 10 attributes, 63241 rows
- `Initiation_of_Dopaminergic_Therapy_14Dec2025.csv` - 8 attributes, 8325 rows
- `LEDD_Concomitant_Medication_Log_14Dec2025.csv` - 15 attributes, 9211 rows
- `Medical_Conditions_Log_14Dec2025.csv` - 15 attributes, 32235 rows
- `NX_PI-2620_Tau_Imaging_Substudy_AE_In-Clinic_Assessment_14Dec2025.csv` - 11 attributes, 11 rows
- `NX_PI-2620_Tau_Imaging_Substudy_AE_Telephone_Assessment_14Dec2025.csv` - 12 attributes, 11 rows
- `NX_PI-2620_Tau_Imaging_Substudy_Pregnancy_Test_14Dec2025.csv` - 12 attributes, 6 rows
- `Neurological_Exam_14Dec2025.csv` - 16 attributes, 18483 rows
- `Other_Clinical_Features_14Dec2025.csv` - 43 attributes, 24676 rows
- `PD_Diagnosis_History_14Dec2025.csv` - 15 attributes, 1992 rows
- `Participant_Global_Impression__PGI__14Dec2025.csv` - 8 attributes, 6719 rows
- `Pregnancy_Test_14Dec2025.csv` - 13 attributes, 876 rows
- `Primary_Clinical_Diagnosis_14Dec2025.csv` - 11 attributes, 27120 rows
- `Procedure_for_PD_Log_14Dec2025.csv` - 11 attributes, 203 rows
- `Report_of_Pregnancy_14Dec2025.csv` - 9 attributes, 6 rows
- `SV2A_PET_Imaging_Substudy_Adverse_Event_In-Clinic_Assessment_14Dec2025.csv` - 11 attributes, 18 rows
- `SV2A_PET_Imaging_Substudy_Adverse_Event_Telephone_Assessment_14Dec2025.csv` - 12 attributes, 18 rows
- `SV2A_PET_Imaging_Substudy_Pregnancy_Test_14Dec2025.csv` - 11 attributes, 15 rows
- `Tau_Substudy_Adverse_Event_Telephone_Assessment_14Dec2025.csv` - 12 attributes, 21 rows
- `Tau_Substudy_Pregnancy_Test_14Dec2025.csv` - 12 attributes, 0 rows
- `Tau_Substudy_Report_of_Pregnancy_14Dec2025.csv` - 10 attributes, 0 rows
- `Vital_Signs_14Dec2025.csv` - 17 attributes, 28723 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `REC_ID` | 52 files | Record ID (unique identifier) |
| `PATNO` | 52 files | Participant ID (primary key) |
| `EVENT_ID` | 52 files | Visit identifier |
| `PAG_NAME` | 52 files | Common identifier or metadata field |
| `ORIG_ENTRY` | 52 files | Common identifier or metadata field |
| `LAST_UPDATE` | 52 files | Common identifier or metadata field |
| `INFODT` | 48 files | Information/assessment date |
| `SUB_EVENT_ID` | 30 files | Common identifier or metadata field |
| `PREGAPPL` | 8 files | Common identifier or metadata field |
| `UPREGPRF` | 5 files | Common identifier or metadata field |

---

### Motor   MDS-UPDRS

**Location:** `ppmi_pd/Motor___MDS-UPDRS/`

**Description:** Motor function assessments including MDS-UPDRS parts I-IV and gait data.

**Number of Files:** 12

**Key Files:**

- `Gait_Data___Arm_swing__Axivity__14Dec2025.csv` - 106 attributes, 100 rows
- `Gait_Data___Arm_swing__Opals__14Dec2025.csv` - 60 attributes, 292 rows
- `Gait_Substudy_Gait_Mobility_Assessment_and_Measurement_14Dec2025.csv` - 26 attributes, 362 rows
- `MDS-UPDRS_Part_III_14Dec2025.csv` - 65 attributes, 35937 rows
- `MDS-UPDRS_Part_IV__Motor_Complications_14Dec2025.csv` - 23 attributes, 10358 rows
- `MDS-UPDRS_Part_I_14Dec2025.csv` - 15 attributes, 30608 rows
- `MDS-UPDRS_Part_I_Patient_Questionnaire_14Dec2025.csv` - 16 attributes, 33034 rows
- `MDS_UPDRS_Part_II__Patient_Questionnaire_14Dec2025.csv` - 22 attributes, 33035 rows
- `Modified_Schwab___England_Activities_of_Daily_Living_14Dec2025.csv` - 8 attributes, 29368 rows
- `Neuro_QoL__Lower_Extremity_Function__Mobility__-_Short_Form_14Dec2025.csv` - 15 attributes, 11362 rows
- `Neuro_QoL__Upper_Extremity_Function_-_Short_Form_14Dec2025.csv` - 15 attributes, 11358 rows
- `Participant_Motor_Function_Questionnaire_14Dec2025.csv` - 19 attributes, 10962 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `PATNO` | 12 files | Participant ID (primary key) |
| `INFODT` | 11 files | Information/assessment date |
| `REC_ID` | 10 files | Record ID (unique identifier) |
| `EVENT_ID` | 10 files | Visit identifier |
| `PAG_NAME` | 10 files | Common identifier or metadata field |
| `ORIG_ENTRY` | 10 files | Common identifier or metadata field |
| `LAST_UPDATE` | 10 files | Common identifier or metadata field |
| `NUPSOURC` | 3 files | Common identifier or metadata field |
| `VISNO` | 2 files | Common identifier or metadata field |

---

### Non-motor Assessments

**Location:** `ppmi_pd/Non-motor_Assessments/`

**Description:** Cognitive, sleep, mood, and other non-motor assessments.

**Number of Files:** 23

**Key Files:**

- `Benton_Judgement_of_Line_Orientation_14Dec2025.csv` - 43 attributes, 17215 rows
- `Clock_Drawing_14Dec2025.csv` - 18 attributes, 10984 rows
- `Cognitive_Categorization_14Dec2025.csv` - 15 attributes, 17901 rows
- `Cognitive_Change_14Dec2025.csv` - 8 attributes, 12452 rows
- `Epworth_Sleepiness_Scale_14Dec2025.csv` - 16 attributes, 19180 rows
- `Geriatric_Depression_Scale__Short_Version__14Dec2025.csv` - 22 attributes, 19460 rows
- `Hopkins_Verbal_Learning_Test_-_Revised_14Dec2025.csv` - 20 attributes, 17265 rows
- `IDEA_Cognitive_Screen_14Dec2025.csv` - 17 attributes, 10 rows
- `Letter_-_Number_Sequencing_14Dec2025.csv` - 31 attributes, 17221 rows
- `Lexical_Fluency_14Dec2025.csv` - 13 attributes, 10738 rows
- `Modified_Boston_Naming_Test_14Dec2025.csv` - 13 attributes, 10472 rows
- `Modified_Semantic_Fluency_14Dec2025.csv` - 13 attributes, 17270 rows
- `Montreal_Cognitive_Assessment__MoCA__14Dec2025.csv` - 35 attributes, 17980 rows
- `Neuro_QoL__Cognition_Function_-_Short_Form_14Dec2025.csv` - 11 attributes, 11367 rows
- `Neuro_QoL__Communication_-_Short_Form_14Dec2025.csv` - 12 attributes, 11368 rows
- `PDAQ-27_14Dec2025.csv` - 34 attributes, 3004 rows
- `QUIP-Current-Short_14Dec2025.csv` - 21 attributes, 19142 rows
- `REM_Sleep_Behavior_Disorder_Questionnaire_14Dec2025.csv` - 29 attributes, 19194 rows
- `SCOPA-AUT_14Dec2025.csv` - 43 attributes, 19160 rows
- `State-Trait_Anxiety_Inventory_14Dec2025.csv` - 47 attributes, 19432 rows
- `Symbol_Digit_Modalities_Test_14Dec2025.csv` - 12 attributes, 17219 rows
- `Trail_Making_A_and_B_14Dec2025.csv` - 16 attributes, 10693 rows
- `University_of_Pennsylvania_Smell_Identification_Test_UPSIT_14Dec2025.csv` - 95 attributes, 8628 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `REC_ID` | 23 files | Record ID (unique identifier) |
| `PATNO` | 23 files | Participant ID (primary key) |
| `EVENT_ID` | 23 files | Visit identifier |
| `PAG_NAME` | 23 files | Common identifier or metadata field |
| `INFODT` | 23 files | Information/assessment date |
| `ORIG_ENTRY` | 23 files | Common identifier or metadata field |
| `LAST_UPDATE` | 23 files | Common identifier or metadata field |
| `PTCGBOTH` | 4 files | Common identifier or metadata field |

---

### PPMI Online

**Location:** `ppmi_pd/PPMI_Online/`

**Description:** Online assessments and questionnaires.

**Number of Files:** 44

**Key Files:**

- `Age_of_Parkinson_s_Disease_Diagnosis__Online__14Dec2025.csv` - 9 attributes, 11324 rows
- `Assessment_of_Constipation__Online__14Dec2025.csv` - 10 attributes, 223985 rows
- `COVID-19_History_Part_2__Online__14Dec2025.csv` - 30 attributes, 49960 rows
- `COVID-19_History__Online__14Dec2025.csv` - 20 attributes, 93049 rows
- `Caffeine_Consumption__Online__14Dec2025.csv` - 38 attributes, 41341 rows
- `Chemical_Exposure__Online__14Dec2025.csv` - 42 attributes, 32540 rows
- `Cognitive_Change__Online__14Dec2025.csv` - 9 attributes, 165264 rows
- `Epworth_Sleepiness_Scale__Online__14Dec2025.csv` - 16 attributes, 62950 rows
- `Family_History_of_PD_1st_Degree_Relatives__Online__14Dec2025.csv` - 26 attributes, 221715 rows
- `Genetic_Testing_Results__Online__14Dec2025.csv` - 14 attributes, 16060 rows
- `Geriatric_Depression_Scale__Online__14Dec2025.csv` - 22 attributes, 164393 rows
- `Head_Injuries__Online__14Dec2025.csv` - 19 attributes, 41352 rows
- `Health_History_Annually__Online__14Dec2025.csv` - 57 attributes, 73502 rows
- `Health_History_Quarterly__Online__14Dec2025.csv` - 21 attributes, 221294 rows
- `High_Interest_Questions_for_Non-PD_Cohort__Online__14Dec2025.csv` - 12 attributes, 28720 rows
- `High_Interest_Questions_for_PD_Cohort__Online__14Dec2025.csv` - 9 attributes, 11933 rows
- `History_of_Falls_Baseline__Online__14Dec2025.csv` - 13 attributes, 37052 rows
- `History_of_Falls_Surveillance__Online__14Dec2025.csv` - 13 attributes, 187225 rows
- `How_you_heard_about_PPMI__Online__14Dec2025.csv` - 44 attributes, 30051 rows
- `Hyposmia_1Qx_from_Remote__Online__14Dec2025.csv` - 9 attributes, 122798 rows
- `MDS-UPDRS_Part_II_Motor_Aspects__Online__14Dec2025.csv` - 21 attributes, 165366 rows
- `MDS-UPDRS_Part_I_Non-Motor_Aspects__Online__14Dec2025.csv` - 15 attributes, 165366 rows
- `Medication_History__Online__14Dec2025.csv` - 48 attributes, 220737 rows
- `Non-Completer_Survey__Online__14Dec2025.csv` - 33 attributes, 391 rows
- `Occupation_and_Military_Service__Online__14Dec2025.csv` - 26 attributes, 15979 rows
- `PD_History_Return_Study_Visit_for_NonPD_Cohort__Online__14Dec2025.csv` - 10 attributes, 131290 rows
- `PD_History_Return_Study_Visit_for_PD_Cohort__Online__14Dec2025.csv` - 10 attributes, 59129 rows
- `PPMI_Online_Codebook_14Dec2025.csv` - 6 attributes, 2733 rows
- `PPMI_Online_Dictionary_14Dec2025.csv` - 11 attributes, 1386 rows
- `PPMI_RBD_Sleep_Questionnaire__Online__14Dec2025.csv` - 15 attributes, 225586 rows
- `Parkinson_Anxiety_Scale__Online__14Dec2025.csv` - 19 attributes, 222602 rows
- `Parkinson_s_Disease_Sleep_Scale_PDSS-2__Online__14Dec2025.csv` - 22 attributes, 73555 rows
- `Participant-Visit_Information__Online__14Dec2025.csv` - 11 attributes, 399058 rows
- `Participant_Enrollment_Status__Online__14Dec2025.csv` - 3 attributes, 46436 rows
- `Participant_Motor_Function_Questionnaire__Online__14Dec2025.csv` - 19 attributes, 181868 rows
- `Penn_Parkinson_s_Daily_Activities_Questionnaire-15__Online__14Dec2025.csv` - 23 attributes, 74167 rows
- `Pesticides_at_Work__Online__14Dec2025.csv` - 481 attributes, 32586 rows
- `Physical_Activity__Online__14Dec2025.csv` - 36 attributes, 28290 rows
- `RBD1Q_Postuma_Acting_out_Dreams__Online__14Dec2025.csv` - 9 attributes, 225586 rows
- `Race_and_Ethnicity__Online__14Dec2025.csv` - 16 attributes, 38350 rows
- `Registration_Information__Online__14Dec2025.csv` - 8 attributes, 46436 rows
- `Residential_Location__Online__14Dec2025.csv` - 9 attributes, 15929 rows
- `Smoking_History__Online__14Dec2025.csv` - 26 attributes, 41330 rows
- `Socioeconomic_Status__Online__14Dec2025.csv` - 9 attributes, 34881 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `PATNO` | 42 files | Participant ID (primary key) |
| `EVENT_ID` | 40 files | Visit identifier |
| `MODIFIED_AT` | 40 files | Common identifier or metadata field |
| `CREATED_AT` | 40 files | Common identifier or metadata field |
| `RESPONDENT_ID` | 39 files | Common identifier or metadata field |
| `MOD_INSTANCE_ID` | 39 files | Common identifier or metadata field |
| `RESPONSE_STATUS` | 39 files | Common identifier or metadata field |
| `NUPSOURC_OL` | 2 files | Common identifier or metadata field |
| `VERSION` | 2 files | Common identifier or metadata field |
| `MOD_NAME` | 2 files | Common identifier or metadata field |

---

### PPMI Remote Screening

**Location:** `ppmi_pd/PPMI_Remote_Screening/`

**Description:** Remote screening data.

**Number of Files:** 6

**Key Files:**

- `Remote_Screening_High_Interest_14Dec2025.csv` - 19 attributes, 146132 rows
- `Remote_Screening_Participant_Progress_14Dec2025.csv` - 23 attributes, 210585 rows
- `Remote_Screening_Screener_14Dec2025.csv` - 28 attributes, 288334 rows
- `Remote_Screening_Smell_Test_Direct_Screener_14Dec2025.csv` - 25 attributes, 200132 rows
- `Remote_Screening_UPSIT_Screening_14Dec2025.csv` - 19 attributes, 7910 rows
- `Remote_University_of_Pennsylvania_Smell_Identification_Test_14Dec2025.csv` - 95 attributes, 108794 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `PATNO` | 6 files | Participant ID (primary key) |
| `EVENT_ID` | 6 files | Visit identifier |
| `PAG_NAME` | 6 files | Common identifier or metadata field |
| `ORIG_ENTRY` | 6 files | Common identifier or metadata field |
| `LAST_UPDATE` | 6 files | Common identifier or metadata field |
| `INFODT` | 5 files | Information/assessment date |
| `COHORT` | 2 files | Common identifier or metadata field |
| `TRACK` | 2 files | Common identifier or metadata field |
| `ORIGIN` | 2 files | Common identifier or metadata field |
| `GENETIC` | 2 files | Common identifier or metadata field |

---

### Roche Smartphone App

**Location:** `ppmi_pd/Roche_Smartphone_App/`

**Description:** Smartphone-based monitoring and questionnaires.

**Number of Files:** 1

**Key Files:**

- `Roche_PD_Monitoring_App_v2_data_14Dec2025.csv` - 22 attributes, 108901 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|

---

### Subject Demographics

**Location:** `ppmi_pd/Subject_Demographics/`

**Description:** Demographic and baseline participant information.

**Number of Files:** 4

**Key Files:**

- `Age_at_visit_14Dec2025.csv` - 3 attributes, 43185 rows
- `Demographics_14Dec2025.csv` - 29 attributes, 7832 rows
- `Socio-Economics_14Dec2025.csv` - 11 attributes, 7776 rows
- `Subject_Cohort_History_14Dec2025.csv` - 3 attributes, 948 rows

**Common Attributes (across files in this category):**

| Attribute | Frequency | Description |
|-----------|-----------|-------------|
| `PATNO` | 4 files | Participant ID (primary key) |
| `EVENT_ID` | 3 files | Visit identifier |
| `REC_ID` | 2 files | Record ID (unique identifier) |
| `PAG_NAME` | 2 files | Common identifier or metadata field |
| `INFODT` | 2 files | Information/assessment date |
| `ORIG_ENTRY` | 2 files | Common identifier or metadata field |
| `LAST_UPDATE` | 2 files | Common identifier or metadata field |

---

## Common Identifiers Across PPMI Datasets

### Primary Keys:
- **`PATNO`** - Participant Number (public identifier, used across most tables)
- **`ALIAS_ID`** - Biorepository internal identifier (used in biosample tables, may need mapping to PATNO)
- **`REC_ID`** - Record ID (unique identifier for each row, often UUID format)
- **`EVENT_ID`** - Visit identifier (BL=Baseline, SC=Screening, V01-V24=Visits, R01-R24=Remote visits, U01-U02=Unscheduled, etc.)
- **`SUB_EVENT_ID`** - Sub-event identifier (for additional visit details)

### Common Visit Codes (EVENT_ID):
- `SC` - Screening
- `BL` - Baseline
- `V01` through `V24` - Scheduled visits
- `R01` through `R24` - Remote visits
- `U01`, `U02` - Unscheduled visits
- `ST` - Symptomatic Therapy
- `PW` - Premature Withdrawal
- `FNL` - Final Visit

---

## Summary Statistics

### parkinsons_tele:
- **1 dataset** with **22 attributes**
- **5,875 rows** (voice recordings)
- **42 participants**

### ppmi_pd:
- **198 datasets** across **13 major categories**
- **5081 total attributes** across all tables
- **Thousands of participants** (varies by table)
- **Longitudinal data** spanning multiple visits per participant

---

*Last Updated: December 2025*
*Data Versions: 14Dec2025 (PPMI), 2009 (Parkinson's Telemonitoring)*

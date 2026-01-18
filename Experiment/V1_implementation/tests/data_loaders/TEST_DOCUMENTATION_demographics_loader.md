# Demographics Loader Test Coverage Documentation

## Overview
Comprehensive test suite for `demographics_loader.py` with 25+ test cases covering unit tests, edge cases, and integration tests.

---

## Test Classes & Methods

### 1. TestDemographicsLoaderFilterValidParticipants (4 tests)
**Purpose:** Test the `__filter_valid_participants__()` method

#### Test 1.1: `test_filter_valid_participants_removes_null_enroll_date`
- **Code Location:** demographics_loader.py, lines 60-65
- **What's Tested:** Filtering removes rows with NULL ENROLL_DATE
- **Example Input:**
  ```
  PATNO:          [1,    2,    3,    4,    5]
  ENROLL_DATE:    ['2020-01-01', '2020-01-02', None, '2020-01-04', None]
  ENROLL_STATUS:  ['Complete'] * 5
  COHORT_DEF:     ['PD'] * 5
  ```
- **Expected Output:**
  ```
  Rows returned: 3 (5 → 3 rows)
  All ENROLL_DATE values: Non-null
  ```

#### Test 1.2: `test_filter_valid_participants_keeps_valid_statuses`
- **Code Location:** demographics_loader.py, lines 67-71
- **What's Tested:** Only valid enrollment statuses retained (Complete, Enrolled, Withdraw Deceased, Withdrew)
- **Example Input:**
  ```
  ENROLL_STATUS: ['Complete', 'InvalidStatus', 'Enrolled', 'Unknown']
  ```
- **Expected Output:**
  ```
  Rows returned: 2 (rows with Complete and Enrolled only)
  All ENROLL_STATUS in: ['Complete', 'Enrolled', 'Withdraw Deceased', 'Withdrew']
  ```

#### Test 1.3: `test_filter_valid_participants_excludes_swedd`
- **Code Location:** demographics_loader.py, lines 73-77
- **What's Tested:** SWEDD cohort excluded; valid cohorts (PD, HC, Prodromal) retained
- **Example Input:**
  ```
  COHORT_DEFINITION: ["Parkinson's Disease", 'Healthy Control', 'Prodromal', 'SWEDD']
  ```
- **Expected Output:**
  ```
  Rows returned: 3 (SWEDD removed)
  'SWEDD' not in result
  ```

#### Test 1.4: `test_filter_valid_participants_returns_required_columns`
- **Code Location:** demographics_loader.py, lines 79-86
- **What's Tested:** Output contains only required columns: [PATNO, COHORT, COHORT_DEFINITION, ENROLL_STATUS, ENROLL_AGE]
- **Example Input:**
  ```
  Input columns: [PATNO, ENROLL_DATE, ENROLL_STATUS, COHORT_DEFINITION, ENROLL_AGE, COHORT, EXTRA_COLUMN]
  ```
- **Expected Output:**
  ```
  Output columns: [PATNO, COHORT, COHORT_DEFINITION, ENROLL_STATUS, ENROLL_AGE]
  Extra columns removed: ENROLL_DATE, EXTRA_COLUMN
  ```

---

### 2. TestDemographicsLoaderMergeData (5 tests)
**Purpose:** Test the `__load_and_merge_data__()` generic merge method

#### Test 2.1: `test_load_and_merge_data_handles_missing_file`
- **Code Location:** demographics_loader.py, lines 89-113
- **What's Tested:** Missing CSV file skipped gracefully with warning
- **Example Input:**
  ```
  file_path: '/nonexistent/file.csv'
  df: Base dataframe with 3 rows
  ```
- **Expected Output:**
  ```
  Result: Original df returned unchanged
  Console: ⚠️  Warning file path not found: /nonexistent/file.csv
  ```

#### Test 2.2: `test_load_and_merge_data_filters_out_patno_from_merge_columns`
- **Code Location:** demographics_loader.py, line 119
- **What's Tested:** PATNO excluded from merge_columns before groupby (prevents "PATNO already exists" error)
- **Example Input:**
  ```
  merge_columns: ['PATNO', 'EDUCYRS']
  CSV data: PATNO [1,1,2,2] with EDUCYRS [16,16,12,12]
  ```
- **Expected Output:**
  ```
  Actual merge columns: ['EDUCYRS'] (PATNO filtered out)
  Result: 2 rows with correct EDUCYRS values
  No error raised
  ```

#### Test 2.3: `test_load_and_merge_data_aggregates_multiple_rows`
- **Code Location:** demographics_loader.py, lines 119-122
- **What's Tested:** Multiple rows per PATNO aggregated using `.first()`
- **Example Input:**
  ```
  CSV: PATNO [1,1,1,2,2] with EDUCYRS [16,16,16,12,12]
  ```
- **Expected Output:**
  ```
  After merge: PATNO [1,2]
  EDUCYRS: [16, 12] (first values taken)
  Rows: 2 (aggregated from 5)
  ```

#### Test 2.4: `test_load_and_merge_data_handles_missing_patno_in_csv`
- **Code Location:** demographics_loader.py, lines 103-107
- **What's Tested:** CSV without PATNO column skipped with warning
- **Example Input:**
  ```
  CSV columns: [ID, EDUCYRS] (no PATNO)
  ```
- **Expected Output:**
  ```
  Result: Original df returned
  Console: ⚠️  Warning: PATNO not found in {file_path}, skipping merge
  ```

#### Test 2.5: `test_load_and_merge_data_selects_only_available_columns`
- **Code Location:** demographics_loader.py, line 119
- **What's Tested:** Only columns that exist in CSV are selected
- **Example Input:**
  ```
  merge_columns requested: ['EDUCYRS', 'NONEXISTENT_COL']
  CSV has: ['PATNO', 'EDUCYRS']
  ```
- **Expected Output:**
  ```
  Actual columns merged: only ['EDUCYRS']
  'NONEXISTENT_COL' not in result
  ```

---

### 3. TestDemographicsLoaderMergeAgeAtVisit (3 tests)
**Purpose:** Test the `__load_and_merge_age_at_visit__()` special handling method

#### Test 3.1: `test_load_and_merge_age_at_visit_no_aggregation`
- **Code Location:** demographics_loader.py, lines 155-167
- **What's Tested:** Age_at_visit rows NOT aggregated; each EVENT_ID creates separate row
- **Example Input:**
  ```
  Demographics: 2 unique PATNO
  age_at_visit: 5 rows
    PATNO [1,1,2,2,2]
    EVENT_ID [ev1,ev2,ev1,ev2,ev3]
  ```
- **Expected Output:**
  ```
  Result rows: 5 (all age_at_visit rows preserved)
  PATNO 1: 2 rows (ev1, ev2)
  PATNO 2: 3 rows (ev1, ev2, ev3)
  Each row has all demographics columns + EVENT_ID + AGE_AT_VISIT
  ```

#### Test 3.2: `test_load_and_merge_age_at_visit_keeps_event_id`
- **Code Location:** demographics_loader.py, lines 150-152
- **What's Tested:** EVENT_ID and AGE_AT_VISIT columns preserved
- **Example Input:**
  ```
  age_at_visit columns: [PATNO, EVENT_ID, AGE_AT_VISIT]
  ```
- **Expected Output:**
  ```
  Result has columns: ['PATNO', 'EVENT_ID', 'AGE_AT_VISIT', ...demographics...]
  EVENT_ID values accessible for grouping by visit
  ```

#### Test 3.3: `test_load_and_merge_age_at_visit_handles_missing_file`
- **Code Location:** demographics_loader.py, lines 135-143
- **What's Tested:** Missing age_at_visit file skipped gracefully
- **Example Input:**
  ```
  age_at_visit file: /nonexistent/file.csv
  ```
- **Expected Output:**
  ```
  Result: Original df returned unchanged
  Console: ⚠️  Warning: age_at_visit file not found: ...
  ```

---

### 4. TestDemographicsLoaderValidation (2 tests)
**Purpose:** Test the `validate()` method

#### Test 4.1: `test_validate_checks_patno_column_exists`
- **Code Location:** demographics_loader.py, lines 231-235
- **What's Tested:** ValueError raised if PATNO column missing
- **Example Input:**
  ```
  df: DataFrame without PATNO
  ```
- **Expected Output:**
  ```
  Exception: ValueError("Missing PATNO column")
  ```

#### Test 4.2: `test_validate_counts_unique_patients`
- **Code Location:** demographics_loader.py, lines 236-238
- **What's Tested:** Correctly counts unique patients; handles multi-row data
- **Example Input:**
  ```
  5 rows with PATNO [1,1,2,2,3]
  ```
- **Expected Output:**
  ```
  Console: ✓ Validation passed: 3 unique patients, 5 total records
  Return value: True
  ```

---

### 5. TestDemographicsLoaderNumericConversion (2 tests)
**Purpose:** Test numeric type conversion with `pd.to_numeric()`

#### Test 5.1: `test_numeric_conversion_coerces_invalid_values_to_nan`
- **Code Location:** demographics_loader.py, lines 215-230
- **What's Tested:** Invalid numeric strings converted to NaN with errors='coerce'
- **Example Input:**
  ```
  Series: ['16', '12', 'NA', '14']
  ```
- **Expected Output:**
  ```
  Result: [16.0, 12.0, NaN, 14.0]
  Invalid string 'NA' → NaN (no error raised)
  ```

#### Test 5.2: `test_numeric_conversion_handles_nan_values`
- **Code Location:** demographics_loader.py, lines 215-230
- **What's Tested:** Existing NaN values preserved
- **Example Input:**
  ```
  Series: [1.0, 2.0, NaN, 3.0]
  ```
- **Expected Output:**
  ```
  Result: [1.0, 2.0, NaN, 3.0] (unchanged)
  NaN positions preserved
  ```

---

### 6. TestDemographicsLoaderEdgeCases (3 tests)
**Purpose:** Test edge cases and boundary conditions

#### Test 6.1: `test_loader_handles_empty_dataframe`
- **Code Location:** demographics_loader.py, lines 55-86
- **What's Tested:** Empty dataframe after filtering handled gracefully
- **Example Input:**
  ```
  All participants filtered out
  Empty df with columns but 0 rows
  ```
- **Expected Output:**
  ```
  Result: 0 rows
  Columns preserved: [PATNO, COHORT, ...]
  No exception raised
  ```

#### Test 6.2: `test_loader_handles_all_nan_column`
- **Code Location:** demographics_loader.py, lines 215-230
- **What's Tested:** Column of all NaN values handled gracefully
- **Example Input:**
  ```
  EDUCYRS: [NaN, NaN, NaN]
  ```
- **Expected Output:**
  ```
  Column preserved with all NaN
  No conversion errors
  ```

#### Test 6.3: `test_loader_maintains_data_types`
- **Code Location:** demographics_loader.py, lines 215-230
- **What's Tested:** Data types maintained correctly
- **Example Input:**
  ```
  PATNO: [1, 2, 3]
  COHORT_DEFINITION: ['PD', 'HC', 'PD']
  ```
- **Expected Output:**
  ```
  PATNO dtype: int64
  COHORT_DEFINITION dtype: object
  ```

---

### 7. TestDemographicsLoaderIntegration (5 tests)
**Purpose:** Integration tests for complete workflow

#### Test 7.1: `test_demographics_loader_load_complete`
- **Code Location:** demographics_loader.py, lines 170-204
- **What's Tested:** Complete load workflow from actual CSV files
- **Example Output:**
  ```
  ✓ Loaded demographics: 423 unique patients, 89 columns
  ✓ Validation passed: 423 unique patients, 12345 total records
  ```

#### Test 7.2: `test_demographics_loader_multiple_rows_per_patient`
- **Code Location:** demographics_loader.py, lines 155-167
- **What's Tested:** Age_at_visit creates multiple rows per patient correctly
- **Example Output:**
  ```
  Unique patients: 423
  Total rows: 1200 (multiple EVENT_IDs per PATNO)
  Multiple rows per patient confirmed
  ```

#### Test 7.3: `test_demographics_loader_required_columns_present`
- **Code Location:** demographics_loader.py, lines 206-211
- **What's Tested:** All required columns present in output
- **Example Input:**
  ```
  get_required_columns(): [PATNO, ENROLL_AGE, SEX, COHORT, COHORT_DEFINITION, ENROLL_STATUS]
  ```
- **Expected Output:**
  ```
  All 6 required columns in loaded dataframe
  ```

#### Test 7.4: `test_demographics_loader_age_values_reasonable`
- **Code Location:** demographics_loader.py, lines 213-214
- **What's Tested:** Age values within reasonable range
- **Example Output:**
  ```
  ENROLL_AGE min: 25
  ENROLL_AGE max: 95
  All values between 0-150
  ```

#### Test 7.5: `test_demographics_loader_cohort_distribution`
- **Code Location:** demographics_loader.py, lines 213-214
- **What's Tested:** Cohort distribution is reasonable and non-empty
- **Example Output:**
  ```
  Cohort distribution (by unique patient):
    Healthy Control:      195 (46%)
    Parkinson's Disease:  228 (54%)
    Prodromal:              0 (0%)
  Total unique patients: 423
  ```

---

## Summary of Test Coverage

| Category | Test Class | #Tests | Code Lines Covered |
|----------|-----------|--------|-------------------|
| **Filtering** | TestDemographicsLoaderFilterValidParticipants | 4 | 60-86 |
| **Merging** | TestDemographicsLoaderMergeData | 5 | 89-130 |
| **Age at Visit** | TestDemographicsLoaderMergeAgeAtVisit | 3 | 135-167 |
| **Validation** | TestDemographicsLoaderValidation | 2 | 231-238 |
| **Numeric Conversion** | TestDemographicsLoaderNumericConversion | 2 | 215-230 |
| **Edge Cases** | TestDemographicsLoaderEdgeCases | 3 | Multiple |
| **Integration** | TestDemographicsLoaderIntegration | 5 | Full workflow |
| | **TOTAL** | **24** | **Core logic** |

---

## Running the Tests

### Run all demographics loader tests:
```bash
pytest tests/data_loaders/test_demographics_loader.py -v
```

### Run specific test class:
```bash
pytest tests/data_loaders/test_demographics_loader.py::TestDemographicsLoaderFilterValidParticipants -v
```

### Run specific test method:
```bash
pytest tests/data_loaders/test_demographics_loader.py::TestDemographicsLoaderFilterValidParticipants::test_filter_valid_participants_removes_null_enroll_date -v
```

### Run only integration tests (requires data):
```bash
pytest tests/data_loaders/test_demographics_loader.py::TestDemographicsLoaderIntegration -v
```

### Run with data requirement:
```bash
pytest tests/data_loaders/test_demographics_loader.py -v -m requires_data
```

---

## Test Markers Used

- `@pytest.mark.requires_data` - Integration tests requiring actual PPMI CSV files
- `@pytest.mark.slow` - Longer-running tests (can skip with `-m "not slow"`)

---

## Key Testing Patterns

### 1. Private Method Access
Tests access private methods using name mangling:
```python
loader._DemographicsLoader__filter_valid_participants__(df)
loader._DemographicsLoader__load_and_merge_data__(df, path, cols)
```

### 2. Mock File Creation
Tests use `tmp_path` fixture to create temporary test CSVs:
```python
csv_path = tmp_path / "test_data.csv"
test_data = pd.DataFrame({...})
test_data.to_csv(csv_path, index=False)
```

### 3. Captured Output Verification
Tests check console output using `capsys`:
```python
captured = capsys.readouterr()
assert '423 unique patients' in captured.out
```

### 4. Exception Testing
Tests verify error handling:
```python
with pytest.raises(ValueError, match="Missing PATNO column"):
    loader.validate(df)
```

---

## Test Fixtures (from conftest.py)

- `test_config`: Configuration object with data paths
- `skip_if_no_data`: Skips test if CSV files not found
- `capsys`: Captures stdout/stderr for assertion

---

## Code Coverage

These 24 tests cover:
- ✅ All filtering logic (null handling, status validation, cohort filtering)
- ✅ All merging logic (missing files, column selection, aggregation, PATNO handling)
- ✅ Age_at_visit special handling (no aggregation, multi-row preservation)
- ✅ Numeric type conversion (invalid values, NaN handling)
- ✅ Validation (PATNO check, unique patient counting)
- ✅ Edge cases (empty dataframes, all NaN columns, data types)
- ✅ Integration workflow (full load, required columns, reasonable values)

**Note:** `test_aggregate_by_patient` was removed as the `__aggregate_by_patient__()` method is not needed (aggregation now done inline in `__load_and_merge_data__`).

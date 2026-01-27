"""
Example usage of PatientPredictor for disease stage prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from .predictor import PatientPredictor, DiseaseStage
import json


def create_sample_patient_data():
    """
    Create sample patient data with 4 visits for demonstration.
    In practice, this would come from your database.
    """
    # Sample data structure - adjust column names to match your data
    visits_data = {
        'visit_id': [1, 2, 3, 4],
        'visit_date': pd.date_range('2022-01-15', periods=4, freq='M'),
        'months_since_baseline': [0, 1, 2, 3],
        
        # UPDRS Scores (motor and non-motor)
        'NP1RTOT': [5, 6, 7, 8],          # Non-motor experiences
        'NP2PTOT': [3, 3, 4, 4],          # Non-motor examination
        'NP3TOT': [25, 28, 32, 35],       # Motor examination (severity indicator)
        'NP4TOT': [1, 1, 2, 2],           # Motor complications
        
        # Additional features (example)
        'age': [65, 65, 65, 65],
        'medication_count': [2, 2, 3, 3],
        'tremor_score': [2, 3, 3, 4],
        'rigidity_score': [1, 2, 2, 3],
    }
    
    return pd.DataFrame(visits_data)


def predict_single_patient(
    checkpoint_path: str,
    patient_visits: pd.DataFrame,
    device: str = "cuda"
) -> None:
    """
    Example: Predict disease stage for a single patient.
    """
    print("=" * 80)
    print("DISEASE STAGE PREDICTION - SINGLE PATIENT")
    print("=" * 80)
    
    # Initialize predictor with trained model
    predictor = PatientPredictor(
        checkpoint_path=checkpoint_path,
        config_version="v1",
        device=device
    )
    
    # Make prediction
    print(f"\nPatient History ({len(patient_visits)} visits):")
    print(patient_visits[['visit_id', 'NP3TOT', 'NP1RTOT', 'NP2PTOT', 'NP4TOT']])
    
    try:
        result = predictor.predict_next_visit(
            patient_visits,
            return_confidence=True,
            return_stage=True
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("PREDICTION FOR NEXT VISIT")
        print("=" * 80)
        
        # Current severity
        current = result['current_severity']
        print(f"\nCurrent Severity:")
        print(f"  Stage: {current['stage']}")
        print(f"  Motor Score (NP3TOT): {current['motor_score']:.1f}")
        
        # Predicted UPDRS scores
        print(f"\nPredicted UPDRS Scores (Next Visit):")
        for name, score in result['predicted_updrs'].items():
            print(f"  {name}: {score:.2f}")
        
        # Predicted stage
        predicted_stage = result['predicted_stage']
        print(f"\nPredicted Stage: {predicted_stage}")
        print(f"Stage Thresholds (NP3TOT):")
        for stage, (min_val, max_val) in result['stage_thresholds'].items():
            print(f"  {stage:20s}: {min_val:3d} - {max_val:3d}")
        
        # Disease progression
        print(f"\nDisease Progression:")
        print(f"  Estimated Slope: {result['slope']:.4f} (points/visit)")
        if result['slope'] > 0:
            print(f"  Direction: Deteriorating (↑ higher scores = worse)")
        elif result['slope'] < 0:
            print(f"  Direction: Improving (↓ lower scores = better)")
        else:
            print(f"  Direction: Stable")
        
        # Confidence intervals
        if 'confidence' in result:
            print(f"\nConfidence Intervals (95%):")
            for name, (lower, upper) in result['confidence'].items():
                pred = result['predicted_updrs'][name]
                print(f"  {name}: {pred:.2f} [95% CI: {lower:.2f} - {upper:.2f}]")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()


def predict_multiple_patients(
    checkpoint_path: str,
    patient_data_file: str,
    output_file: str = "predictions.json",
    device: str = "cuda"
) -> None:
    """
    Example: Predict disease stage for multiple patients from CSV file.
    
    Expected CSV columns: patient_id, visit_id, NP1RTOT, NP2PTOT, NP3TOT, NP4TOT, ...
    """
    print("=" * 80)
    print("DISEASE STAGE PREDICTION - BATCH PROCESSING")
    print("=" * 80)
    
    from .predictor import BatchPredictor
    
    # Load patient data
    print(f"\nLoading patient data from {patient_data_file}")
    df = pd.read_csv(patient_data_file)
    print(f"Total records: {len(df)}")
    
    # Group by patient
    patient_groups = {}
    for patient_id, group in df.groupby('patient_id'):
        patient_groups[patient_id] = group.reset_index(drop=True)
    
    print(f"Unique patients: {len(patient_groups)}")
    
    # Initialize batch predictor
    batch_predictor = BatchPredictor(
        checkpoint_path=checkpoint_path,
        config_version="v1",
        device=device
    )
    
    # Make predictions
    print(f"\nMaking predictions...")
    results = batch_predictor.predict_patients(patient_groups, output_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results.values() if 'error' not in r)
    print(f"Total patients: {len(results)}")
    print(f"Successful predictions: {successful}")
    print(f"Failed predictions: {len(results) - successful}")
    
    # Show sample results
    if successful > 0:
        print(f"\nSample Result (First Successful Patient):")
        for patient_id, result in results.items():
            if 'error' not in result:
                print(f"\nPatient: {patient_id}")
                print(f"  Predicted Stage: {result['predicted_stage']}")
                print(f"  Predicted NP3TOT: {result['predicted_updrs']['NP3TOT']:.2f}")
                print(f"  Disease Slope: {result['slope']:.4f}")
                break


def interactive_prediction(checkpoint_path: str, device: str = "cuda") -> None:
    """
    Interactive mode: Enter patient data manually and get prediction.
    """
    print("=" * 80)
    print("INTERACTIVE DISEASE STAGE PREDICTION")
    print("=" * 80)
    
    predictor = PatientPredictor(
        checkpoint_path=checkpoint_path,
        config_version="v1",
        device=device
    )
    
    # Get number of visits
    while True:
        try:
            n_visits = int(input("\nEnter number of visits (4-10): "))
            if 4 <= n_visits <= 10:
                break
        except ValueError:
            pass
        print("Please enter a number between 4 and 10")
    
    # Build patient data
    visits_data = {
        'NP1RTOT': [],
        'NP2PTOT': [],
        'NP3TOT': [],
        'NP4TOT': [],
        'months_since_baseline': []
    }
    
    print(f"\nEnter UPDRS scores for each visit:")
    for visit_idx in range(n_visits):
        print(f"\n--- Visit {visit_idx + 1} ---")
        try:
            visits_data['NP1RTOT'].append(float(input("NP1RTOT (Non-motor experiences): ")))
            visits_data['NP2PTOT'].append(float(input("NP2PTOT (Non-motor examination): ")))
            visits_data['NP3TOT'].append(float(input("NP3TOT (Motor - indicates severity): ")))
            visits_data['NP4TOT'].append(float(input("NP4TOT (Motor complications): ")))
            visits_data['months_since_baseline'].append(float(visit_idx))
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            return
    
    visits_df = pd.DataFrame(visits_data)
    
    # Make prediction
    try:
        result = predictor.predict_next_visit(visits_df)
        
        print("\n" + "=" * 80)
        print("PREDICTION RESULTS")
        print("=" * 80)
        
        print(f"\nCurrent Stage: {result['current_severity']['stage']}")
        print(f"Predicted Next Stage: {result['predicted_stage']}")
        print(f"\nPredicted UPDRS Scores:")
        for name, score in result['predicted_updrs'].items():
            print(f"  {name}: {score:.2f}")
        print(f"\nDisease Progression Rate: {result['slope']:.4f} points/visit")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict patient disease stage")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (e.g., models/checkpoints/fold_1/best_checkpoint.pt)")
    parser.add_argument("--mode", type=str, default="sample", 
                       choices=["sample", "batch", "interactive"],
                       help="Prediction mode")
    parser.add_argument("--patient-file", type=str, help="CSV file with patient data (for batch mode)")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    if args.mode == "sample":
        # Demonstrate with sample data
        sample_data = create_sample_patient_data()
        predict_single_patient(args.checkpoint, sample_data, args.device)
    
    elif args.mode == "batch":
        if not args.patient_file:
            print("Error: --patient-file required for batch mode")
            exit(1)
        predict_multiple_patients(args.checkpoint, args.patient_file, args.output, args.device)
    
    elif args.mode == "interactive":
        interactive_prediction(args.checkpoint, args.device)

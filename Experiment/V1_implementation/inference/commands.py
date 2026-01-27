"""
Inference/prediction command for the V1 model.
Add this to main.py as an alternative mode.
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Optional


def add_inference_args(subparsers):
    """Add inference subcommand to argument parser"""
    inference_parser = subparsers.add_parser(
        'predict',
        help='Make disease stage predictions for patients'
    )
    
    inference_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    inference_parser.add_argument(
        '--patient-data',
        type=str,
        help='CSV file with patient visit data (patient_id, visit_id, NP1RTOT, NP2PTOT, NP3TOT, NP4TOT, ...)'
    )
    
    inference_parser.add_argument(
        '--patient-id',
        type=str,
        help='Specific patient ID to predict (optional, if not provided predicts all)'
    )
    
    inference_parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Output file for predictions'
    )
    
    inference_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda, mps, cpu)'
    )
    
    inference_parser.add_argument(
        '--config-version',
        choices=['v1', 'v2'],
        default='v1',
        help='Config version used during training'
    )
    
    inference_parser.add_argument(
        '--mode',
        choices=['sample', 'batch', 'interactive'],
        default='sample',
        help='Prediction mode'
    )
    
    return inference_parser


def run_inference(args):
    """
    Run inference/prediction based on arguments.
    Call this from main() when args.command == 'predict'
    """
    from inference.example_usage import (
        predict_single_patient,
        predict_multiple_patients,
        interactive_prediction,
        create_sample_patient_data
    )
    
    if args.mode == 'sample':
        # Demo with sample data
        print("\n" + "=" * 80)
        print("RUNNING SAMPLE PREDICTION")
        print("=" * 80)
        sample_data = create_sample_patient_data()
        predict_single_patient(args.checkpoint, sample_data, args.device)
    
    elif args.mode == 'batch':
        # Batch prediction from CSV
        if not args.patient_data:
            print("Error: --patient-data required for batch mode")
            return
        
        if not Path(args.patient_data).exists():
            print(f"Error: Patient data file not found: {args.patient_data}")
            return
        
        predict_multiple_patients(
            args.checkpoint,
            args.patient_data,
            args.output,
            args.device
        )
    
    elif args.mode == 'interactive':
        # Interactive input
        interactive_prediction(args.checkpoint, args.device)

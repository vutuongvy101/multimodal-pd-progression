"""
Standalone script for generating single-patient UPDRS visualizations.

This script is reusable and allows you to:
- Generate plots for ANY patient in cached predictions
- Batch generate plots for multiple patients
- Customize output directory
- No need to re-run full evaluation

Usage:
    python plot_single_patient.py --patno 3003 --npz visualization/plotters/preds_fold7_correct.npz
    python plot_single_patient.py --patno 3003 236541 235339 --npz ... --output-dir plots/
    python plot_single_patient.py --patno-file patients.txt --npz ... --output-dir plots/
"""

# Fix for OpenMP conflict (must be before importing numpy/matplotlib)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.plotters.evaluation import EvaluationPlotter
from visualization.io import PlotIO


def load_predictions_from_npz(npz_path: str):
    """Load predictions cache from .npz file."""
    cache = np.load(npz_path, allow_pickle=False)
    
    predictions = torch.from_numpy(cache['predictions']).float()
    actuals = torch.from_numpy(cache['actuals']).float()
    time_months = torch.from_numpy(cache['time_months']).float()
    patno = cache['patno']
    
    return predictions, actuals, time_months, patno


def get_available_patnos(patno_array: np.ndarray):
    """Get unique PATNOs from the cached data."""
    return sorted(np.unique(patno_array).astype(int))


def generate_single_patient_plots(
    patno_list: list,
    npz_path: str,
    output_dir: str = "plots",
    plot_components: bool = True,
    plot_total: bool = True,
    skip_missing: bool = True
):
    """
    Generate single-patient visualization plots.
    
    Args:
        patno_list: List of PATNOs to plot
        npz_path: Path to cached predictions .npz file
        output_dir: Directory to save plots
        plot_components: Generate 4-component subplot
        plot_total: Generate total UPDRS plot
        skip_missing: Skip patients not in cache (default: True)
    """
    print(f"Loading predictions from: {npz_path}")
    predictions, actuals, time_months, patno = load_predictions_from_npz(npz_path)
    
    available_patnos = get_available_patnos(patno)
    print(f"Available patients in cache: {len(available_patnos)}")
    
    # Create output directory
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    plotter = EvaluationPlotter()
    io = PlotIO(str(output_path), show=False, dpi=150)
    
    print(f"\nGenerating plots for {len(patno_list)} patients...")
    print("=" * 80)
    
    successful = 0
    skipped = 0
    
    for target_patno in patno_list:
        if target_patno not in available_patnos:
            if skip_missing:
                print(f"⊘ PATNO {target_patno}: Not found in cache (skipped)")
                skipped += 1
                continue
            else:
                print(f"✗ PATNO {target_patno}: Not found in cache (ERROR)")
                continue
        
        try:
            if plot_components:
                print(f"→ PATNO {target_patno}: Generating 4-component plot...", end="", flush=True)
                plotter.plot_single_patient_prediction(
                    predictions=predictions,
                    actuals=actuals,
                    time_months=time_months,
                    patno=patno,
                    target_patno=target_patno,
                    io=io
                )
                print(" ✓")
            
            if plot_total:
                print(f"→ PATNO {target_patno}: Generating total UPDRS plot...", end="", flush=True)
                plotter.plot_single_patient_total_updrs(
                    predictions=predictions,
                    actuals=actuals,
                    time_months=time_months,
                    patno=patno,
                    target_patno=target_patno,
                    io=io
                )
                print(" ✓")
            
            successful += 1
            
        except Exception as e:
            print(f" ✗ ERROR: {str(e)}")
    
    print("=" * 80)
    print(f"\nResults:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ⊘ Skipped: {skipped}")
    print(f"  Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-patient UPDRS visualization plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single patient
  python plot_single_patient.py --patno 3003 --npz visualization/plotters/preds_fold7_correct.npz
  
  # Multiple patients
  python plot_single_patient.py --patno 3003 236541 235339 --npz ... --output-dir plots/
  
  # From file (one PATNO per line)
  python plot_single_patient.py --patno-file patients.txt --npz ... --output-dir plots/
  
  # List available patients
  python plot_single_patient.py --list-patnos --npz visualization/plotters/preds_fold7_correct.npz
  
  # Top N lowest UPDRS patients
  python plot_single_patient.py --top-lowest-updrs 10 --npz ... --output-dir plots/
        """)
    
    parser.add_argument(
        "--patno",
        type=int,
        nargs='+',
        default=None,
        help="Patient ID(s) to plot"
    )
    parser.add_argument(
        "--patno-file",
        type=str,
        default=None,
        help="File with PATNOs (one per line)"
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="visualization/plotters/preds_fold7_correct.npz",
        help="Path to cached predictions .npz file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--list-patnos",
        action="store_true",
        help="List all available PATNOs and exit"
    )
    parser.add_argument(
        "--top-lowest-updrs",
        type=int,
        default=None,
        help="Generate plots for top N patients with lowest mean UPDRS (requires patno_updrs_scores.csv)"
    )
    parser.add_argument(
        "--skip-components",
        action="store_true",
        help="Skip 4-component plots (only generate total UPDRS)"
    )
    parser.add_argument(
        "--skip-total",
        action="store_true",
        help="Skip total UPDRS plots (only generate component plots)"
    )
    
    args = parser.parse_args()
    
    # Load predictions to get available PATNOs
    print(f"Loading predictions from: {args.npz}")
    predictions, actuals, time_months, patno = load_predictions_from_npz(args.npz)
    available_patnos = get_available_patnos(patno)
    
    # Handle --list-patnos
    if args.list_patnos:
        print(f"\nAvailable PATNOs in {args.npz}:")
        print(f"Total: {len(available_patnos)} patients\n")
        print(", ".join(str(p) for p in available_patnos))
        return
    
    # Determine which patients to plot
    patnos_to_plot = []
    
    if args.top_lowest_updrs:
        # Load UPDRS scores and find top N with lowest mean
        import pandas as pd
        try:
            df_updrs = pd.read_csv('data/patno_updrs_scores.csv')
            lowest_summary = (
                df_updrs[df_updrs['PATNO'].isin(available_patnos)]
                .groupby('PATNO')['Total_UPDRS']
                .mean()
                .nsmallest(args.top_lowest_updrs)
            )
            patnos_to_plot = lowest_summary.index.astype(int).tolist()
            print(f"\nTop {args.top_lowest_updrs} patients with lowest mean UPDRS:")
            for p, score in lowest_summary.items():
                print(f"  PATNO {p}: {score:.2f}")
        except FileNotFoundError:
            print("Error: Could not find data/patno_updrs_scores.csv")
            print("Run: python -m training.extract_patno_updrs --output data/patno_updrs_scores.csv")
            return
    
    elif args.patno_file:
        # Load from file
        try:
            with open(args.patno_file, 'r') as f:
                patnos_to_plot = [int(line.strip()) for line in f if line.strip()]
            print(f"Loaded {len(patnos_to_plot)} PATNOs from {args.patno_file}")
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    
    elif args.patno:
        patnos_to_plot = args.patno
    
    else:
        print("Error: Must specify --patno, --patno-file, --top-lowest-updrs, or --list-patnos")
        parser.print_help()
        return
    
    if not patnos_to_plot:
        print("Error: No patients specified")
        return
    
    # Generate plots
    generate_single_patient_plots(
        patno_list=patnos_to_plot,
        npz_path=args.npz,
        output_dir=args.output_dir,
        plot_components=not args.skip_components,
        plot_total=not args.skip_total,
        skip_missing=True
    )


if __name__ == "__main__":
    main()

import argparse
import os

from visualization.config import VizConfig
from visualization.visualizers import SingleRunVisualizer, KFoldVisualizer, CompareVisualizer


def main():
    p = argparse.ArgumentParser(description="Minimal training_history visualizer (next_visit + slope).")
    p.add_argument("path", nargs="?", help="Single mode: training_history.json. Kfold: base dir.")

    p.add_argument("--output-dir", default="./plots")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--target", default="NP3TOT")
    p.add_argument("--dpi", type=int, default=150)

    p.add_argument("--kfold", action="store_true")
    p.add_argument("--pattern", default="fold_*/training_history.json")

    p.add_argument("--compare", action="store_true")
    p.add_argument("--path1", default=None)
    p.add_argument("--path2", default=None)
    p.add_argument("--name1", default="Model1")
    p.add_argument("--name2", default="Model2")

    args = p.parse_args()
    cfg = VizConfig(target=args.target, dpi=args.dpi)
    show = not args.no_show

    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare:
        if not args.path1 or not args.path2:
            raise ValueError("Compare mode requires --path1 and --path2")
        CompareVisualizer(cfg).run(args.path1, args.path2, args.output_dir, show=show, name1=args.name1, name2=args.name2)
        return

    if args.kfold:
        if not args.path:
            raise ValueError("K-fold mode requires base directory as positional 'path'")
        title = os.path.basename(args.path.rstrip("/")) or "kfold"
        KFoldVisualizer(cfg).run(args.path, args.pattern, args.output_dir, show=show, title=title)
        return

    if not args.path:
        raise ValueError("Single mode requires a training_history.json path")
    title = os.path.basename(os.path.dirname(args.path)) or "run"
    SingleRunVisualizer(cfg).run(args.path, args.output_dir, show=show, title=title)

if __name__ == "__main__":
    main()

# Stand at V1_implementation and run below scripts:

# -- Single fold --
#   python -m visualization.main \
#   models/checkpoints/modalities_static/fold_1/training_history.json \
#   --output-dir ./visualization/plots \
#   --target NP3TOT

# -- K-fold --
#   python -m visualization.main \
#   models/checkpoints/modalities_static \
#   --kfold \
#   --pattern "fold_*/training_history.json" \
#   --output-dir ./visualization/kfold_plots \
#   --target NP3TOT

# -- Compare 2 histories --
#   python -m visualization.main --compare \
#   --path1 models/checkpoints/modalities_static/fold_1/training_history.json \
#   --path2 models/checkpoints/modalities_static/fold_2/training_history.json \
#   --name1 "static_1" --name2 "static_2" \
#   --output-dir ./visualization/compare_plots \
#   --target NP3TOT

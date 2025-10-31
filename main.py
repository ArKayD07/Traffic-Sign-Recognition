import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_ORDER = [
    "generate_synthetic.py",
    "convert_annotations.py",
    "augmentation.py",
    "train_and_eval.py",
    "evaluate_and_sanity_check.py",
    "evaluate_and_save_named_preds.py",
    "compare_models.py",
]

DEFAULT_ARGS = {
    #Leave empty to use each script's internal defaults; override in CLI if needed
    "generate_synthetic.py": [],
    "convert_annotations.py": ["--dataset_dir", "dataset", "--output_dir", "converted_dataset"],
    "augmentation.py": [],
    "train_and_eval.py": ["--dataset_dir", "dataset", "--output_dir", "results"],
    "evaluate_and_sanity_check.py": ["--dataset_dir", "dataset", "--model_path", "results/best_model.pth", "--output_dir", "results/eval"],
    "evaluate_and_save_named_preds.py": ["--dataset_dir", "dataset", "--model_path", "results/best_model.pth", "--model_name", "modelA", "--output_dir", "results/eval"],
}


def find_scripts():
    py_files = [f for f in os.listdir(ROOT) if f.lower().endswith('.py') and f != os.path.basename(__file__)]
    return sorted(py_files)


def run_script(script, extra_args=None, python_exe=None, dry_run=False):
    path = os.path.join(ROOT, script)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Script not found: {script}")
    python_exe = python_exe or sys.executable
    cmd = [python_exe, path] + (extra_args or [])
    print("\n--- Running:", ' '.join(cmd))
    if dry_run:
        return 0
    try:
        res = subprocess.run(cmd, check=True)
        return res.returncode
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Script {script} exited with code {e.returncode}")
        raise


def parse_steps_arg(val):
    parts = [p.strip() for p in val.split(',') if p.strip()]
    return parts


def main():
    p = argparse.ArgumentParser(description="Run project scripts in a convenient order")
    p.add_argument("--all", action='store_true', help="Run default full pipeline (default order)")
    p.add_argument("--steps", type=str, help="Comma-separated script basenames to run (e.g. generate_synthetic,train_and_eval)")
    p.add_argument("--list", action='store_true', help="List available scripts in this folder")
    p.add_argument("--continue_on_error", action='store_true', help="If set, keep running next steps even if one fails")
    p.add_argument("--dry_run", action='store_true', help="Print the commands but do not execute them")
    p.add_argument("--python", type=str, default=sys.executable, help="Path to python executable to use")
    args, unknown = p.parse_known_args()

    available = find_scripts()
    if args.list:
        print("Available scripts:")
        for s in available:
            print(" -", s)
        return

    if args.steps:
        steps = parse_steps_arg(args.steps)
        normalized = []
        for s in steps:
            if not s.lower().endswith('.py'):
                s = s + '.py'
            if s not in available:
                print(f"[WARN] Requested step not found: {s}")
            else:
                normalized.append(s)
        steps = normalized
    elif args.all:
        steps = [s for s in DEFAULT_ORDER if s in available]
    else:
        p.print_help()
        return

    if len(steps) == 0:
        print("No valid steps to run. Use --list to see available scripts.")
        return

    print(f"Will run steps: {steps}")

    for script in steps:
        extra = DEFAULT_ARGS.get(script, [])
        try:
            run_script(script, extra_args=extra, python_exe=args.python, dry_run=args.dry_run)
        except Exception as e:
            print(f"[ERROR] Step failed: {script} -> {e}")
            if not args.continue_on_error:
                print("Aborting remaining steps.")
                raise
            else:
                print("Continuing to next step due to --continue_on_error")

    print("\nAll requested steps completed (or attempted).")


if __name__ == '__main__':
    main()

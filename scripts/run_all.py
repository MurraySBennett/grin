import subprocess
import sys
import time
from pathlib import Path

# Define the sequence of scripts with their respective command-line arguments.
# Each entry is a list representing the command split by spaces.
PIPELINE_STEPS = [
    # 1. Core Data Generation, Training & Evaluation
    # # ["python", "scripts/generate_data.py", "--report"],
    # ["python", "scripts/train.py"],
    # ["python", "scripts/evaluate.py"], # need to be rerun / extended (for RT, too)
    ["python", "scripts/make_figures.py"],

    # 2. Response Time (RT) Expansion
    # # ["python", "scripts/generate_data.py", "--rt"],
    # ["python", "scripts/train_rt.py"],
    ["python", "scripts/evaluate_rt.py"],
    ["python", "scripts/make_figures_rt.py"],

    # 3. R Baseline Benchmarks & Comparison
    ["python", "scripts/export_for_r.py", "--n", "600"],
    # ["Rscript", "scripts/R/fit_baselines.R"],
    # ["python", "scripts/compare_to_r.py"],
    # ["Rscript", "scripts/R/fit_real_data.R"],

    # 4. Model Export & Validation Runs
    ["python", "scripts/export_onnx.py"],
    ["python", "scripts/export_onnx.py", "--rt"],
    ["python", "validation/run_all.py"],
    # ["python", "validation/sweeps.py"],
    # ["python", "scripts/build_benchmarks.py"]
]

def run_pipeline():
    total_start = time.time()
    print("======================================================================")
    print(f"🚀 Starting Project-Wide ML & Validation Pipeline")
    print(f"   Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("======================================================================\n")

    for idx, cmd in enumerate(PIPELINE_STEPS, 1):
        # Human-readable string representation of the command
        cmd_str = " ".join(cmd)
        print(f"🔹 [{idx}/{len(PIPELINE_STEPS)}] Running: {cmd_str}")
        
        # Verify target file exists before trying to run it
        target_file = Path(cmd[1])
        if not target_file.exists():
            print(f"❌ Error: Script file '{target_file}' does not exist! Aborting.")
            sys.exit(1)

        step_start = time.time()
        
        try:
            # subprocess.run waits for execution and handles streams safely
            subprocess.run(cmd, check=True)
            
            elapsed = time.time() - step_start
            print(f"✅ Success | Elapsed: {elapsed/60:.1f}m ({elapsed:.1f}s)\n")
            
        except subprocess.CalledProcessError as e:
            print("\n" + "="*70)
            print(f"❌ PIPELINE FAILURE AT STEP {idx}: {cmd_str}")
            print(f"Exit code: {e.returncode}")
            print("Aborting remaining pipeline steps to prevent corrupted evaluations.")
            print("="*70)
            sys.exit(e.returncode)
            
        except KeyboardInterrupt:
            print("\n🛑 Pipeline manually interrupted by user. Exiting safely.")
            sys.exit(130)

    total_elapsed = time.time() - total_start
    print("======================================================================")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   Total Execution Time: {total_elapsed/60:.2f} minutes")
    print("======================================================================")

if __name__ == "__main__":
    run_pipeline()

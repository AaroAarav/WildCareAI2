import subprocess
import sys

def run_step(script_name):
    print(f"\n{'='*50}\nRUNNING {script_name}\n{'='*50}")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"ERROR: {script_name} failed. Aborting pipeline.")
        sys.exit(1)

def main():
    steps = [
        "step2_class_mapping.py",
        "step3_convert_and_clean.py",
        "step4_balance.py",
        "step5_split.py",
        "step6_generate_configs.py",
        "step7_validate_and_report.py"
    ]
    
    for step in steps:
        run_step(step)
        
    print("\nPipeline execution complete! Dataset ready.")

if __name__ == "__main__":
    main()
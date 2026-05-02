import os
import subprocess
import sys

def run_phase(phase_name, script_path):
    print("\n" + "="*80)
    print(f" EXECUTING {phase_name.upper()}")
    print("="*80 + "\n")
    
    if not os.path.exists(script_path):
        print(f"[ERROR] Could not find script at: {script_path}")
        sys.exit(1)
        
    # Run the script using the current Python executable
    # Set the current working directory to the script's directory so imports work correctly
    script_dir = os.path.dirname(script_path)
    result = subprocess.run([sys.executable, script_path], cwd=script_dir)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {phase_name} encountered an error.")
        sys.exit(result.returncode)

if __name__ == '__main__':
    # Get the directory where this main.py is located
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths to the central printing/main scripts for each phase
    phase1_script = os.path.join(root_dir, 'Phase-1', 'main.py')
    phase2_script = os.path.join(root_dir, 'Phase-2', 'Phase2_Printing.py')
    phase3_script = os.path.join(root_dir, 'Phase-3', 'Phase3_Printing.py')
    phase4_script = os.path.join(root_dir, 'Phase-4', 'Phase4_Printing.py')
    
    # Execute them in order
    run_phase("Phase 1", phase1_script)
    run_phase("Phase 2", phase2_script)
    run_phase("Phase 3", phase3_script)
    run_phase("Phase 4", phase4_script)
    
    print("\n" + "="*80)
    print(" ALL PHASES EXECUTED SUCCESSFULLY! ")
    print("="*80 + "\n")

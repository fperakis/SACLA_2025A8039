# process.py
import subprocess
import re
import argparse
import time
from pathlib import Path

'''
Processes the img data of a run and save an h5 file. 
Example of use from terminal: 

python process.py \
  --data_path /xustrg0/2024B8049 \
  --run_number 222753 \
  --mask_path /UserData/fperakis/test_data_2025/utilities/empty_mask.npy \
  --output_path /UserData/fperakis/test_data_2025/processed \
  --poni_file /UserData/fperakis/test_data_2025/utilities/geometry_test.poni \
  --nbins 150 \
  --n_phi 36 \
  --n_chunks 10

'''

def submit_array_job(config):
    env_vars = ",".join([
        f"DATA_PATH={config.data_path}",
        f"RUN_NUMBER={config.run_number}",
        f"MASK_PATH={config.mask_path}",
        f"OUTPUT_PATH={config.output_path}",
        f"PONI_FILE={config.poni_file}",
        f"NBINS={config.nbins}",
        f"N_PHI={config.n_phi}",
        f"N_CHUNKS={config.n_chunks}"
    ])

    submit_cmd = [
        "qsub", "-J", f"0-{config.n_chunks - 1}",
        "-v", env_vars,
        "submit_chunk.pbs"
    ]

    result = subprocess.run(submit_cmd, capture_output=True, text=True)
    match = re.search(r"(\d+)", result.stdout)

    if not match:
        raise RuntimeError("Failed to parse array job ID. Output:\n" + result.stdout)

    job_id = match.group(1)
    print(f"Array job submitted with ID: {job_id}")
    return job_id

def submit_merge_job(array_job_id, config):
    output_file = Path(config.output_path) / f"Iq_{config.run_number}_combined.h5"
    job_id_base = f"{array_job_id}[]"  # cluster-dependent, may need adjustment

    merge_cmd = [
        "qsub",
        "-W", f"depend=afterok:{job_id_base}",
        "-v", f"INPUT_DIR={config.output_path},RUN_NUMBER={config.run_number},OUTPUT_FILE={output_file}, DATA_PATH={config.data_path}",
        "combine_chunks.pbs"
    ]

    result = subprocess.run(merge_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        merge_job_id = result.stdout.strip()
        print(f"Merge job submitted successfully with job ID: {merge_job_id}")
    else:
        print("Failed to submit merge job.")
        print(result.stderr)

def main():
    parser = argparse.ArgumentParser(description="Submit array and merge PBS jobs for processing chunks.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--run_number", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--poni_file", required=True)
    parser.add_argument("--nbins", type=int, default=150)
    parser.add_argument("--n_phi", type=int, default=36)
    parser.add_argument("--n_chunks", type=int, default=10)

    config = parser.parse_args()

    array_job_id = submit_array_job(config)
    submit_merge_job(array_job_id, config)

if __name__ == "__main__":
    main()

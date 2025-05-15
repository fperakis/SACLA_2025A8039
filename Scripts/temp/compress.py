# to run this:
#$ python compress.py -r run -n 10

import os
import sys
import argparse
import glob

# - parsers
parser = argparse.ArgumentParser(description='Compress the run and analyse')
parser.add_argument('-r', '--run', type=str, required=True, help='run number to process')
parser.add_argument('-n', '--nsplits', type=int, required=False, default=10, help='number of jobs to submit')

args = parser.parse_args()
run = args.run
n_splits = args.nsplits
# n_splits = 10

directory = f"/xustrg0/2024B8049/{run}/"  
N_shots = len(glob.glob1(directory,"*.img"))

print("N_shots: ", N_shots)
print("Run: ", run)

for i in range(n_splits):
    os.system(f'qsub -v run={run},tagStart={int(N_shots/n_splits*i)},tagEnd={int(N_shots/n_splits*i+N_shots/n_splits)} job_all.sh')
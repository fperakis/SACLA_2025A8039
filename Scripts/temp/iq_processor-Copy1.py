import os
import argparse
import glob
import time
import numpy as np
import h5py
import pyFAI
import matplotlib.image as mpimg

def main(data_path, run_number, mask_path, output_path,
         poni_file, nbins, n_phi):

    # Determine chunk index and total number of chunks
    chunk_index = int(os.environ.get("PBS_ARRAY_INDEX", 0))
    n_chunks = int(os.environ["N_CHUNKS"])

    # Load mask and integrator
    mask = np.load(mask_path)
    ai = pyFAI.load(poni_file)

    # Build list of .img files
    run_dir = os.path.join(data_path, run_number)
    file_list = sorted(glob.glob(os.path.join(run_dir, "data_*.img")))
    total_shots = len(file_list)

    if total_shots == 0:
        print(f"No .img files found in {run_dir}")
        return

    # Calculate chunk range
    chunk_size = total_shots // n_chunks
    remainder = total_shots % n_chunks

    start = chunk_index * chunk_size + min(chunk_index, remainder)
    end = start + chunk_size + (1 if chunk_index < remainder else 0)

    print(f"Processing chunk {chunk_index+1}/{n_chunks}: shots {start} to {end-1}")
    print(f"Total shots: {total_shots}")

    # Allocate output array
    I = np.zeros((end - start, n_phi, nbins))

    for i, shot in enumerate(range(start, end)):
        image_path = file_list[shot]
        img = mpimg.imread(image_path).astype("int16")

        start_time = time.time()
        I[i], q, phi = ai.integrate2d_ng(
            img, nbins, n_phi, mask=mask,
            correctSolidAngle=False, unit="q_nm^-1"
        )
        print(f"Shot {shot}: {1 / (time.time() - start_time):.2f} Hz")

    # Save results
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"Iq_{run_number}_chunk{chunk_index:02d}.h5")

    with h5py.File(out_file, "w") as f:
        f.create_dataset("I", data=I)
        f.create_dataset("q", data=q/10) ##! in angstrom
        f.create_dataset("phi", data=phi)

    print(f"Iq saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--run_number", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--poni_file", required=True)
    parser.add_argument("--nbins", type=int, default=150)
    parser.add_argument("--n_phi", type=int, default=72)

    args = parser.parse_args()

    main(args.data_path, args.run_number, args.mask_path,
         args.output_path, args.poni_file, args.nbins, args.n_phi)

import os
import argparse
import glob
import time
import numpy as np
import h5py
import pyFAI
import pyFAI.azimuthalIntegrator
import pyFAI.load
import matplotlib.image as mpimg

'''
python iq_processor.py \
  --data_path /xustrg0/2024B8049 \
  --run_number 222753 \
  --mask_path /UserData/fperakis/test_data_2025/utilities/empty_mask.npy \
  --output_path /UserData/fperakis/test_data_2025/processed \
  --poni_file /UserData/fperakis/test_data_2025/utilities/geometry_test.poni \
  --nbins 150 \
  --n_phi 72 \
  --n_chunks 10 \
  --chunk_index 0

'''

def main(data_path, run_number, mask_path, output_path,
         poni_file, nbins, n_phi, n_chunks, chunk_index):

    # Load mask
    mask = np.load(mask_path)

    # Load integrator from .poni file
    ai = pyFAI.load(poni_file)

    # Prepare file list
    run_dir = os.path.join(data_path, run_number)
    file_list = sorted(glob.glob(os.path.join(run_dir, "data_*.img")))
    total_shots = len(file_list)

    if total_shots == 0:
        print(f"No .img files found in {run_dir}")
        return

    # Compute chunk range
    chunk_size = total_shots // n_chunks
    remainder = total_shots % n_chunks

    start = chunk_index * chunk_size + min(chunk_index, remainder)
    end = start + chunk_size + (1 if chunk_index < remainder else 0)

    print(f"Processing chunk {chunk_index+1}/{n_chunks}: shots {start} to {end-1}")
    print(f"Total shots in run: {total_shots}")

    # Allocate output array
    I = np.zeros([end - start, n_phi, nbins])

    for i, shot in enumerate(range(start, end)):
        image_path = file_list[shot]
        img = mpimg.imread(image_path).astype("int16")

        start_time = time.time()
        I[i, :, :], q, phi = ai.integrate2d_ng(
            img, nbins, n_phi, mask=mask,
            correctSolidAngle=True, unit="q_nm^-1"
        )
        print(f"Shot {shot} | {1 / (time.time() - start_time):.2f} Hz")

    # Save result
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"Iq_{run_number}_chunk{chunk_index:02d}.h5")

    with h5py.File(output_file, "w") as f:
        f.create_dataset("I", data=I)
        f.create_dataset("q", data=q)
        f.create_dataset("phi", data=phi)

    print(f"Iq saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Azimuthal integration over run chunks using .poni geometry")

    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--run_number", required=True, type=str)
    parser.add_argument("--mask_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--poni_file", required=True, type=str)

    parser.add_argument("--nbins", default=150, type=int)
    parser.add_argument("--n_phi", default=72, type=int)
    parser.add_argument("--n_chunks", default=1, type=int)
    parser.add_argument("--chunk_index", default=0, type=int)

    args = parser.parse_args()

    main(args.data_path, args.run_number, args.mask_path, args.output_path,
         args.poni_file, args.nbins, args.n_phi, args.n_chunks, args.chunk_index)

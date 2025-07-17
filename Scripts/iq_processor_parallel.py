#!/usr/bin/env python3

import os
import argparse
import glob
import time
import re
import numpy as np
import h5py
import pyFAI
import matplotlib.image as mpimg
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_image(args):
    image_path, mask_path, poni_file, nbins, n_phi = args
    try:
        img = mpimg.imread(image_path).astype("int16")
        ai = pyFAI.load(poni_file)
        mask = np.load(mask_path)
        polarization = ai.guess_polarization(img, npt_rad=nbins, npt_azim=n_phi, unit='q_A^-1')
        I, q, phi = ai.integrate2d_ng(
            img, nbins, n_phi, mask=mask,
            correctSolidAngle=False, unit="q_A^-1", polarization_factor=polarization
        )
        match = re.search(r"data_(\d+)\.img", os.path.basename(image_path))
        image_id = int(match.group(1)) if match else -1
        return (I, q, phi, image_id, image_path, None)
    except Exception as e:
        return (None, None, None, None, image_path, str(e))

def main(data_path, run_number, mask_path, output_path, poni_file, nbins, n_phi, n_workers, batch_size):
    run_dir = os.path.join(data_path, run_number)
    file_list = sorted(glob.glob(os.path.join(run_dir, "data_*.img")))
    total_shots = len(file_list)

    if total_shots == 0:
        print(f"No .img files found in {run_dir}")
        return

    print(f"Processing {total_shots} images with {n_workers} workers, batch size {batch_size}...")

    # Prepare output file and extendable datasets
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"Iq_{run_number}.h5")
    with h5py.File(out_file, "w") as f:
        maxshape = (None, n_phi, nbins)
        I_ds = f.create_dataset("I", shape=(0, n_phi, nbins), maxshape=maxshape, dtype=np.float32, chunks=True)
        image_ids_ds = f.create_dataset("image_id", shape=(0,), maxshape=(None,), dtype=int, chunks=True)
        q_ds = None
        phi_ds = None
        batch_start = 0
        for batch_idx in range(0, total_shots, batch_size):
            batch_files = file_list[batch_idx:batch_idx+batch_size]
            args_list = [(image_path, mask_path, poni_file, nbins, n_phi) for image_path in batch_files]
            batch_len = len(batch_files)
            I_batch = np.zeros((batch_len, n_phi, nbins), dtype=np.float32)
            image_ids_batch = np.zeros((batch_len,), dtype=int)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_image, args): idx for idx, args in enumerate(args_list)}
                for i, future in enumerate(as_completed(futures)):
                    idx = futures[future]
                    I, q, phi, image_id, image_path, error = future.result()
                    if error:
                        print(f"Error processing {image_path}: {error}")
                        continue
                    I_batch[idx] = I
                    image_ids_batch[idx] = image_id
                    if q_ds is None and phi_ds is None:
                        # Save q and phi only once
                        q_ds = f.create_dataset("q", data=q)
                        phi_ds = f.create_dataset("phi", data=phi)
                    print(f"Processed {batch_start + i + 1}/{total_shots}: {image_path}")
            # Append batch results to datasets
            I_ds.resize(I_ds.shape[0] + batch_len, axis=0)
            I_ds[-batch_len:] = I_batch
            image_ids_ds.resize(image_ids_ds.shape[0] + batch_len, axis=0)
            image_ids_ds[-batch_len:] = image_ids_batch
            batch_start += batch_len
    print(f"Iq saved to {out_file}")

if __name__ == "__main__":
    from multiprocessing import cpu_count
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--run_number", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--poni_file", required=True)
    parser.add_argument("--nbins", type=int, default=250)
    parser.add_argument("--n_phi", type=int, default=36)
    parser.add_argument("--n_workers", type=int, default=cpu_count(), help="Number of parallel workers")
    parser.add_argument("--n_chunks", type=int, default=100, help="Number of images per batch")

    args = parser.parse_args()

    main(args.data_path, args.run_number, args.mask_path,
         args.output_path, args.poni_file, args.nbins, args.n_phi, args.n_workers, args.n_chunks)

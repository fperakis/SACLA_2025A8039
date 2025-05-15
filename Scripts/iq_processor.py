import argparse
import glob
import time
import numpy as np
import h5py
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import detector_factory
from concurrent.futures import ThreadPoolExecutor
from matplotlib import image as mpimg  # Replace with fabio if needed

'''
Example of usage: 

python iq_processor.py \
  --data_path /xustrg0/2024B8049 \
  --run_number 222753 \
  --mask_path /UserData/fperakis/test_data_2025/utilities/empty_mask.npy \
  --output_path /UserData/fperakis/test_data_2025/processed/ \
  --detector_name RayonixMx225hs \
  --dist 0.095 \
  --wavelength 0.887e-10 \
  --poni1 0.09736 \
  --poni2 0.10823 \
  --nbins 150 \
  --n_phi 72

'''


def process_shot(filename, ai, nbins, n_phi, mask):
    img = mpimg.imread(filename).astype('int16')
    I2d, _, _ = ai.integrate2d_ng(img, nbins, n_phi, mask=mask, correctSolidAngle=True, unit='q_nm^-1')
    return I2d

def main(data_path, run_number, mask_path, output_path, detector_name,
         dist, wavelength, poni1, poni2, nbins=100, n_phi=36):

    # Load detector
    detector = detector_factory(detector_name)

    # Set up AzimuthalIntegrator
    ai = AzimuthalIntegrator(dist=dist,
                             detector=detector,
                             wavelength=wavelength,
                             poni1=poni1,
                             poni2=poni2)

    # Load mask
    mask = np.load(mask_path)

    # Image files
    run_dir = f"{data_path}/{run_number}"
    file_list = sorted(glob.glob(f"{run_dir}/*.img"))
    N_shots = len(file_list)

    print(f"Processing {N_shots} shots from run {run_number}...")

    I_all = np.zeros((N_shots, n_phi, nbins))

    start_time = time.time()

    # Parallel processing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda fname: process_shot(fname, ai, nbins, n_phi, mask), file_list))

    for i, I in enumerate(results):
        I_all[i, :, :] = I

    print(f"Processed in {time.time() - start_time:.2f} seconds")

    # Save output
    output_file = f"{output_path}/Iq_{run_number}.h5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("Iq", data=I_all)
        f.create_dataset("q", data=ai.qArray((n_phi, nbins)))
        f.create_dataset("phi", data=ai.chiArray((n_phi, nbins)))

    print(f"Iq saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate I(q, φ) from SACLA detector images and save to HDF5.")

    # File and path arguments
    parser.add_argument("--data_path", type=str, required=True, help="Base path to data directory")
    parser.add_argument("--run_number", type=str, required=True, help="Run number or folder name (e.g., 0023)")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to .npy mask file")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save output HDF5 file")

    # Detector and integration parameters
    parser.add_argument("--detector_name", type=str, required=True,
                        help="Detector name for pyFAI, e.g., 'RayonixMx225hs'")
    parser.add_argument("--dist", type=float, required=True, help="Sample-detector distance in meters")
    parser.add_argument("--wavelength", type=float, required=True, help="Wavelength in meters")
    parser.add_argument("--poni1", type=float, required=True, help="PONI1 (vertical) in meters")
    parser.add_argument("--poni2", type=float, required=True, help="PONI2 (horizontal) in meters")

    # Optional integration parameters
    parser.add_argument("--nbins", type=int, default=100, help="Number of q bins (radial)")
    parser.add_argument("--n_phi", type=int, default=36, help="Number of azimuthal bins")

    args = parser.parse_args()

    main(args.data_path, args.run_number, args.mask_path, args.output_path,
         args.detector_name, args.dist, args.wavelength, args.poni1, args.poni2,
         args.nbins, args.n_phi)

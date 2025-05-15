import numpy as np
import h5py 
import pyFAI, pyFAI.detectors, pyFAI.azimuthalIntegrator
import sys
import time
import argparse
import matplotlib.image as mpimg
import glob as glob

# - parsers
parser = argparse.ArgumentParser(description='Analyse run')
parser.add_argument('-r', '--run', type=str, required=True, help='run number to process')
parser.add_argument('-s', '--start', type=str, required=True, help='start tag')
parser.add_argument('-e', '--end', type=str, required=True, default=10, help='end tag')

args = parser.parse_args()

run = args.run
n_start = int(args.start)
n_end = int(args.end)



def Iq_calculator(run,start_img,end_img):
    """
    Calculate the Iq and Iphi of an entire run and generates an h5 file with the Iqs for each shot
    args:
        run : run number without extensions
    """
    # detector params
    path = '/UserData/girelli/sacla2024/'
    ponifile='/utilities/geometry_final.poni'
    
    det=pyFAI.detectors.RayonixMx225hs()

    #define an integrator
    sample_det_distance=0.095
    wavelength=0.887e-10
    posx=det.pixel1*970
    posy=det.pixel2*930
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist = sample_det_distance, detector = det, wavelength = wavelength, poni1 = posx, poni2 = posy)

#    sample_det_distance = 0.095
#    wavelength = 0.887e-10
#    Pixel1, Pixel2 = 50e-6, 50e-6
#    posx, posy = Pixel1*1200, Pixel2*1150
#    det = pyFAI.detectors.Detector(pixel1=Pixel1, pixel2=Pixel2, max_shape=[2399,2399])
#    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist=sample_det_distance, detector=det, wavelength=wavelength, poni1=posx, poni2=posy)
    #ai=pyFAI.load(f'{path}{ponifile}')
    # integration params
    nbins = 300
    n_phi = 36
    N_shots = n_end-n_start
    #radial_range = [0,27]
    
    mask = np.load(f'{path}/utilities/test_mask.npy')
    I=np.zeros([N_shots, n_phi,nbins])
    
    for i,shot in enumerate(range(n_start,n_end)):
        I[i,:,:], phi, q =  ai.integrate2d_ng(mpimg.imread(f"/xustrg0/2024B8049/{run}/data_{shot+1:06}.img").astype('int16'), nbins, n_phi, mask=mask, correctSolidAngle=True, unit='q_nm^-1')
    
    print(f"End of Iq and Iphi calculation from image {n_start} to {n_end}")
    hf = h5py.File(f'{path}/processed/IqPhi_{run}_{n_start}_{n_end}.h5', 'w')
    hf.create_dataset('q', data=q/10) ##! in angstrom
    hf.create_dataset('I', data=I)
    hf.create_dataset('phi', data=phi)
    #hf.create_dataset('PulseEnergy', data=f[f'/run_{run}/event_info/bl_3/oh_2/bm_1_pulse_energy_in_joule'])
    #hf.create_dataset('PhotonEnergy', data=f[f'/run_{run}/event_info/bl_3/oh_2/photon_energy_in_eV'])
    #hf.create_dataset('tags', data=f[f'/run_{run}/event_info/tag_number_list'])
    #hf.create_dataset('shutter', data=f[f'/run_{run}/event_info/bl_3/eh_1/xfel_pulse_selector_status'])
    hf.close()

    #print(f"Analysis saved in {path}/processed/IqPhi_{run}.h5")

print("run: ", args.run,n_start,n_end)

Iq_calculator(args.run,n_start,n_end)

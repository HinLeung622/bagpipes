from __future__ import print_function,  division,  absolute_import

import numpy as np

from astropy.io import fits

from .utils import *
from .models.making import igm_inoue2014
from . import config

""" The standard file names of grids """
stellar_grids = {
    'bc03_miles': "bc03_miles_stellar_grids.fits",               # default
    'knowles23_smiles': "knowles23_smiles_stellar_grids.fits",
    'BPASS_v2.2.1': "bpass_2.2.1_bin_imf135_300_stellar_grids.fits",
    'BPASS_v2.3': "bpass_2.3_bin_imf135_300_stellar_grids.fits"
}
neb_grids = {
    "bc03_miles_extended": ["bc03_miles_nebular_cont_grids_extended.fits", "bc03_miles_nebular_line_grids_extended.fits"]
}
dust_grids = {
    "dl07": ["dl07_grids_umin_only.fits", "dl07_grids_umin_umax.fits"]
}
igm_grids = {
    "inoue14": "d_igm_grid_inoue14.fits"
}


def list_grids():
    # check if each grid exists in file and print lists of available grids
    print('Available stellar grids:')
    for g in stellar_grids:
        if stellar_grids[g] in os.listdir(grid_dir):
            print(g)

    print('')
    print('Available nebular grids:')
    for g in neb_grids:
        if neb_grids[g][0] in os.listdir(grid_dir):
            print(g)
    
    print('')
    print('Available dust grids:')
    for g in dust_grids:
        if dust_grids[g][0] in os.listdir(grid_dir):
            print(g)

    print('')
    print('Available IGM grids:')
    for g in igm_grids:
        if igm_grids[g] in os.listdir(grid_dir):
            print(g)

def which_grids():
    # print currently selected grids
    print('stellar grid:', config.stellar_grid_name)
    print('nebular grid:', config.neb_grid_name)
    print('dust grid:', config.dust_grid_name)
    print('IGM grid:', config.igm_grid_name)

def _change_stellar_grid(stellar_grid_name):
    if stellar_grid_name not in stellar_grids.keys():
        raise ValueError(f"Invalid requested new stellar grid. Only accepting {list(stellar_grids.keys())}")
    
    config.stellar_grid_name = stellar_grid_name
    fail_message = "Failed to update stellar grids, these should be placed in the bagpipes/models/grids/ directory."
    if stellar_grid_name == 'bc03_miles':
        try:
            # Name of the fits file storing the stellar models
            config.stellar_file = stellar_grids[stellar_grid_name]

            # The metallicities of the stellar grids in units of Z_Solar
            config.metallicities = np.array([0.005, 0.02, 0.2, 0.4, 1., 2.5, 5.])

            # The wavelengths of the grid points in Angstroms
            config.wavelengths = fits.open(grid_dir + "/" + config.stellar_file)[-1].data

            # The ages of the grid points in Gyr
            config.raw_stellar_ages = fits.open(grid_dir + "/" + config.stellar_file)[-2].data

            # The alpha enhancement of the grid points in [alpha/Fe] (i.e. log10(alpha/Fe)* - log10(alpha/Fe)sol)
            config.alpha_Fe = np.array([0.0])

            # The fraction of stellar mass still living (1 - return fraction).
            # Axis 0 runs over alpha/Fe, axis 1 runs over metallicity, axis 2 runs over age.
            config.live_frac = np.expand_dims(fits.open(grid_dir + "/" + config.stellar_file)[-3].data[:, 1:].T, axis=0)

            # The raw stellar grids, stored as a FITS HDUList.
            # The different HDUs are the grids at different metallicities.
            # Axis 0 of each grid runs over wavelength, axis 1 over age.
            config.raw_stellar_grid = np.expand_dims(
                np.array([hdu.data for hdu in fits.open(grid_dir + "/" + config.stellar_file)[1:8]]),
                axis=0
            )
            #raw_stellar_grid = fits.open(grid_dir + "/" + stellar_file)[1:8]

            # Set up edge positions for metallicity bins for stellar models.
            config.metallicity_bins = make_bins(config.metallicities, make_rhs=True)[0]
            config.metallicity_bins[0] = 0.
            config.metallicity_bins[-1] = 10.

            # set up edge positions for alpha/Fe bins for stellar models.
            config.alpha_Fe_bins = np.array([0.0, 0.0])

        except IOError:
            print(fail_message)
            
    elif stellar_grid_name == 'knowles23_smiles':
        try:
            # Name of the fits file storing the stellar models
            config.stellar_file = stellar_grids[stellar_grid_name]

            # The metallicities of the stellar grids in units of Z_Solar
            config.metallicities = fits.open(grid_dir + "/" + config.stellar_file)[-2].data

            # The wavelengths of the grid points in Angstroms
            config.wavelengths = fits.open(grid_dir + "/" + config.stellar_file)[-1].data

            # The ages of the grid points in Gyr
            config.raw_stellar_ages = fits.open(grid_dir + "/" + config.stellar_file)[-3].data

            # The alpha enhancement of the grid points in [alpha/Fe] (i.e. log10(alpha/Fe)* - log10(alpha/Fe)sol)
            config.alpha_Fe = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])

            # The fraction of stellar mass still living (1 - return fraction).
            # Axis 0 runs over alpha/Fe, axis 1 runs over metallicity, axis 2 runs over age.
            config.live_frac = np.tile(fits.open(grid_dir + "/" + config.stellar_file)[-4].data, (len(config.alpha_Fe), 1, 1))

            # The raw stellar grids, stored as a numpy array.
            # Axis 0 runs over alpha/Fe, axis 1 over metallicity, axis 2 runs over age, axis 3 runs over wavelength.
            config.raw_stellar_grid = np.array([hdu.data for hdu in fits.open(grid_dir + "/" + config.stellar_file)[1:6]])

            # Set up edge positions for metallicity bins for stellar models.
            config.metallicity_bins = make_bins(config.metallicities, make_rhs=True)[0]
            config.metallicity_bins[0] = 0.
            config.metallicity_bins[-1] = 2.5

            # set up edge positions for alpha/Fe bins for stellar models.
            config.alpha_Fe_bins = make_bins(config.alpha_Fe, make_rhs=True)[0]

        except IOError:
            print(fail_message)
            
    elif stellar_grid_name == 'BPASS_v2.2.1':
        try:
            # Name of the fits file storing the stellar models
            config.stellar_file = stellar_grids[stellar_grid_name]

            # The metallicities of the stellar grids in units of Z_Solar
            config.metallicities = np.array([10**-5, 10**-4, 0.001, 0.002, 0.003, 0.004,
                                            0.006, 0.008, 0.010, 0.014, 0.020, 0.030,
                                            0.040])/0.02

            # The wavelengths of the grid points in Angstroms
            config.wavelengths = fits.open(grid_dir + "/" + config.stellar_file)[-1].data

            # The ages of the grid points in Gyr
            config.raw_stellar_ages = fits.open(grid_dir + "/" + config.stellar_file)[-2].data

            # The alpha enhancement of the grid points in [alpha/Fe] (i.e. log10(alpha/Fe)* - log10(alpha/Fe)sol)
            config.alpha_Fe = np.array([0.0])

            # The fraction of stellar mass still living (1 - return fraction).
            # Axis 0 runs over alpha/Fe, axis 1 runs over metallicity, axis 2 runs over age.
            config.live_frac = np.expand_dims(fits.open(grid_dir + "/" + config.stellar_file)[-3].data.T, axis=0)

            # The raw stellar grids, stored as a FITS HDUList.
            # The different HDUs are the grids at different metallicities.
            # Axis 0 of each grid runs over wavelength, axis 1 over age.
            config.raw_stellar_grid = np.expand_dims(
                np.array([hdu.data for hdu in fits.open(grid_dir + "/" + config.stellar_file)[1:14]]),
                axis=0
            )

            # Set up edge positions for metallicity bins for stellar models.
            config.metallicity_bins = make_bins(config.metallicities, make_rhs=True)[0]
            config.metallicity_bins[0] = 0.
            config.metallicity_bins[-1] = 2.5

            # set up edge positions for alpha/Fe bins for stellar models.
            config.alpha_Fe_bins = np.array([0.0, 0.0])

        except IOError:
            print(fail_message)

    elif stellar_grid_name == 'BPASS_v2.3':
        try:
            # Name of the fits file storing the stellar models
            config.stellar_file = stellar_grids[stellar_grid_name]

            # The metallicities of the stellar grids in units of Z_Solar
            config.metallicities = fits.open(grid_dir + "/" + config.stellar_file)[-2].data


            # The wavelengths of the grid points in Angstroms
            config.wavelengths = fits.open(grid_dir + "/" + config.stellar_file)[-1].data

            # The ages of the grid points in Gyr
            config.raw_stellar_ages = fits.open(grid_dir + "/" + config.stellar_file)[-3].data

            # The alpha enhancement of the grid points in [alpha/Fe] (i.e. log10(alpha/Fe)* - log10(alpha/Fe)sol)
            config.alpha_Fe = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])

            # The fraction of stellar mass still living (1 - return fraction).
            # Axis 0 runs over alpha/Fe, axis 1 runs over metallicity, axis 2 runs over age.
            config.live_frac = fits.open(grid_dir + "/" + config.stellar_file)[-4].data

            # The raw stellar grids, stored as a FITS HDUList.
            # The different HDUs are the grids at different metallicities.
            # Axis 0 of each grid runs over wavelength, axis 1 over age.
            config.raw_stellar_grid = np.array([hdu.data for hdu in fits.open(grid_dir + "/" + config.stellar_file)[1:6]])

            # Set up edge positions for metallicity bins for stellar models.
            config.metallicity_bins = make_bins(config.metallicities, make_rhs=True)[0]
            config.metallicity_bins[0] = 0.
            config.metallicity_bins[-1] = 3.

            # set up edge positions for alpha/Fe bins for stellar models.
            config.alpha_Fe_bins = make_bins(config.alpha_Fe, make_rhs=True)[0]

        except IOError:
            print(fail_message)

def _change_neb_grid(nebular_grid_name):
    pass

def _change_dust_grid(dust_grid_name):
    pass

def _change_igm_grid(IGM_grid_name):
    pass

def change_grid(stellar_grid_name=None, 
                neb_grid_name=None, 
                dust_grid_name=None,
                igm_grid_name=None
                ):
    
    if stellar_grid_name is not None:
        _change_stellar_grid(stellar_grid_name)

    if neb_grid_name is not None:
        _change_neb_grid(neb_grid_name)

    if dust_grid_name is not None:
        _change_dust_grid(dust_grid_name)

    if igm_grid_name is not None:
        _change_igm_grid(igm_grid_name)

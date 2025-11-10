from __future__ import print_function, division, absolute_import

import numpy as np

from .. import config
from .. import utils


class nebular(object):
    """ Allows access to and maniuplation of nebular emission models.
    These must be pre-computed using Cloudy and the relevant set of
    stellar emission models. This has already been done for the default
    stellar models.

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values desired for the stellar models.
    """

    def __init__(self, wavelengths, velshift):
        self.wavelengths = wavelengths
        self.velshift = velshift
        self.combined_grid, self.line_grid = self._setup_grids()

    def _setup_grids(self):
        """ Loads Cloudy nebular continuum grid and resamples to the
        input wavelengths. Loads nebular line grids and adds line fluxes
        to the correct pixels in order to create a combined grid. """

        line_grid = config.line_grid.copy().transpose(4,0,1,2,3)

        comb_grid = np.zeros((self.wavelengths.shape[0],
                              config.neb_alpha_Fe.shape[0],
                              config.neb_metallicities.shape[0],
                              config.logU.shape[0],
                              config.neb_ages.shape[0]))

        for i in range(config.neb_alpha_Fe.shape[0]):
            for j in range(config.neb_metallicities.shape[0]):
                for k in range(config.logU.shape[0]):
                    for l in range(config.neb_ages.shape[0]):
                        comb_grid[:, i, j, k, l] = np.interp(
                            self.wavelengths, config.neb_wavs, config.cont_grid[i,j,k,l,:],
                            left=0, right=0
                        )

        # Add the nebular lines to the resampled nebular continuum grid.
        for i in range(config.line_wavs.shape[0]):
            line_wav_shift = config.line_wavs[i]*(1+(self.velshift/(3*10**5)))
            ind = np.abs(self.wavelengths - line_wav_shift).argmin()
            if ind != 0 and ind != self.wavelengths.shape[0]-1:
                width = (self.wavelengths[ind+1] - self.wavelengths[ind-1])/2
                comb_grid[ind, :, :, :, :] += line_grid[i, :, :, :, :]/width

        # Interpolate both grids to match the stellar grid
        comb_grid, line_grid = self._interpolate_neb_grid_to_stellar_grid(comb_grid, line_grid)

        return comb_grid, line_grid

    def _interpolate_neb_grid_to_stellar_grid(self, comb_grid, line_grid):
        """ Interpolates the nebular grids (combined and line) from the
        nebular alpha/Fe, age and metallicity to the stellar alpha/Fe, 
        age and metallicity. 
        config.neb_alpha_Fe -> config.alpha_Fe
        config.neb_metallicities -> config.metallicities
        config.neb_ages -> config.age_sampling
        For each interpolation step, we first check if the nebular grid points
        match those in the stellar grid point. If match, no need to interpolate
        """

        # age interpolation
        comb_grid = self._resample_in_age(comb_grid)
        line_grid = self._resample_in_age(line_grid)

        # metallicity interpolation
        comb_grid = self._resample_in_metallicity(comb_grid)
        line_grid = self._resample_in_metallicity(line_grid)

        # alpha/Fe interpolation
        comb_grid = self._resample_in_alpha_Fe(comb_grid)
        line_grid = self._resample_in_alpha_Fe(line_grid)

        return comb_grid, line_grid

    def _resample_in_age(self, in_grid):
        """ Perform interpolation in age """
        # check if the age grids are the same
        # the nebular age grid can be much shorter than the stellar age grid, but still work
        # so only check the first N items where N is the length of the neb age grid
        if np.all(
            np.log10(config.neb_ages) == np.log10(config.age_sampling[:len(config.neb_ages)])
            ):
            #print('age interp skipped')
            return in_grid
        
        # main interpolation
        out_grid = np.zeros((in_grid.shape[0],
                             in_grid.shape[1],
                             in_grid.shape[2],
                             in_grid.shape[3],
                             config.age_sampling.shape[0]))

        raw_age_lhs, raw_age_widths = utils.make_bins(config.neb_ages,
                                                      make_rhs=True)

        # Force raw ages to span full range from 0 to age of Universe.
        raw_age_widths[0] += raw_age_lhs[0]
        raw_age_lhs[0] = 0.

        if raw_age_lhs[-1] < config.age_bins[-1]:
            raw_age_widths[-1] += config.age_bins[-1] - raw_age_lhs[-1]
            raw_age_lhs[-1] = config.age_bins[-1]

        start = 0
        stop = 0

        # Loop over the new age bins
        for j in range(config.age_bins.shape[0] - 1):

            # Find the first raw bin partially covered by the new bin
            while raw_age_lhs[start + 1] <= config.age_bins[j]:
                start += 1

            # Find the last raw bin partially covered by the new bin
            while raw_age_lhs[stop+1] < config.age_bins[j + 1]:
                stop += 1

            # If new bin falls completely within one raw bin
            if stop == start:
                out_grid[:, :, :, :, j] = in_grid[:, :, :, :, start]

            # If new bin has contributions from more than one raw bin
            else:
                start_fact = ((raw_age_lhs[start + 1] - config.age_bins[j])
                              / (raw_age_lhs[start + 1] - raw_age_lhs[start]))

                end_fact = ((config.age_bins[j + 1] - raw_age_lhs[stop])
                            / (raw_age_lhs[stop + 1] - raw_age_lhs[stop]))

                raw_age_widths[start] *= start_fact
                raw_age_widths[stop] *= end_fact

                width_slice = raw_age_widths[start:stop + 1]

                summed = np.sum(width_slice[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
                                * in_grid[:, :, :, :, start:stop + 1], axis=4)

                out_grid[:, :, :, :, j] = summed/np.sum(width_slice)

                raw_age_widths[start] /= start_fact
                raw_age_widths[stop] /= end_fact

        return out_grid

    def _resample_in_metallicity(self, in_grid):
        """ Perform interpolation in metallicity """
        # check if the metallicity grids are the same
        if len(config.neb_metallicities) == len(config.metallicities):
            if np.all(config.neb_metallicities == config.metallicities):
                #print('metallicity interp skipped')
                return in_grid

        # main interpolation
        out_grid = np.zeros((in_grid.shape[0],
                             in_grid.shape[1],
                             config.metallicities.shape[0],
                             in_grid.shape[3],
                             in_grid.shape[4]))

        raw_zmet_lhs, raw_zmet_widths = utils.make_bins(config.neb_metallicities,
                                                        make_rhs=True)
        zmet_lhs, zmet_widths = utils.make_bins(config.metallicities,
                                                make_rhs=True)

        # Force raw metallicities to span the full range of the stellar metallicity grid
        if raw_zmet_lhs[0] > zmet_lhs[0]:
            raw_zmet_lhs[0] = zmet_lhs[0]
            raw_zmet_widths[0] = raw_zmet_lhs[1] - raw_zmet_lhs[0]
        if raw_zmet_lhs[-1] < zmet_lhs[-1]:
            raw_zmet_lhs[-1] = zmet_lhs[-1]
            raw_zmet_widths[-1] = raw_zmet_lhs[-1] - raw_zmet_lhs[-2]
        
        start = 0
        stop = 0

        # Loop over the new metallicity bins
        for j in range(zmet_lhs.shape[0] - 1):

            # Find the first raw bin partially covered by the new bin
            while raw_zmet_lhs[start + 1] <= zmet_lhs[j]:
                start += 1

            # Find the last raw bin partially covered by the new bin
            while raw_zmet_lhs[stop+1] < zmet_lhs[j + 1]:
                stop += 1

            # If new bin falls completely within one raw bin
            if stop == start:
                out_grid[:, :, j, :, :] = in_grid[:, :, start, :, :]

            # If new bin has contributions from more than one raw bin
            else:
                start_fact = ((raw_zmet_lhs[start + 1] - zmet_lhs[j])
                              / (raw_zmet_lhs[start + 1] - raw_zmet_lhs[start]))

                end_fact = ((zmet_lhs[j + 1] - raw_zmet_lhs[stop])
                            / (raw_zmet_lhs[stop + 1] - raw_zmet_lhs[stop]))

                raw_zmet_widths[start] *= start_fact
                raw_zmet_widths[stop] *= end_fact

                width_slice = raw_zmet_widths[start:stop + 1]

                summed = np.sum(width_slice[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                                * in_grid[:, :, start:stop + 1, :, :], axis=2)

                out_grid[:, :, j, :, :] = summed/np.sum(width_slice)

                raw_zmet_widths[start] /= start_fact
                raw_zmet_widths[stop] /= end_fact

        return out_grid

    def _resample_in_alpha_Fe(self, in_grid):
        """ Perform interpolation in alpha/Fe """
        # check if the alpha/Fe grids are the same
        if len(config.neb_alpha_Fe) == len(config.alpha_Fe):
            if np.all(config.alpha_Fe == config.alpha_Fe):
                #print('alpha interp skipped')
                return in_grid

        # check if the nebular grid only has one alpha/Fe value
        # if yes, apply the nebular grids to all stellar alpha/Fe values
        if len(config.neb_alpha_Fe) == 1:
            #print('alpha populating single neb grid to all alpha values')
            out_grid = np.tile(in_grid, (1, len(config.alpha_Fe), 1, 1, 1))
            return out_grid

        # check if the destination grid has only one value. 
        if len(config.alpha_Fe) == 1:
            # check if the target grid point is identical to one of the raw grid points
            for i,a in enumerate(config.neb_alpha_Fe):
                if a == config.alpha_Fe[0]:
                    #print('alpha exact value found skipped')
                    out_grid = np.expand_dims(in_grid[:, i, :, :, :], axis=1)
                    return out_grid

            #print('alpha two bin interp')
            # weigh the contributions from neighbouring bins in the raw by proximity
            left_ind = np.where(config.neb_alpha_Fe > config.alpha_Fe[0])[0][-1]
            right_ind = left_ind+1
            start_fact = ((config.neb_alpha_Fe[right_ind] - config.alpha_Fe[0]) / 
                          (config.neb_alpha_Fe[right_ind] - config.neb_alpha_Fe[left_ind]))
            end_fact = ((config.alpha_Fe[0] - config.neb_alpha_Fe[left_ind]) / 
                        (config.neb_alpha_Fe[right_ind] - config.neb_alpha_Fe[left_ind]))

            if start_fact + end_fact != 1:
                raise ValueError("sum of factors not equal 1 in alpha/Fe interpolation!")

            out_grid = np.expand_dims(
                in_grid[:, left_ind, :, :, :] * start_fact + in_grid[:, right_ind, :, :, :] * end_fact,
                axis=1
            )

            return out_grid

        #print('alpha interp')
        # main interpolation
        out_grid = np.zeros((in_grid.shape[0],
                             config.alpha_Fe.shape[0],
                             in_grid.shape[2],
                             in_grid.shape[3],
                             in_grid.shape[4]))

        raw_a_lhs, raw_a_widths = utils.make_bins(config.neb_alpha_Fe,
                                                  make_rhs=True)
        a_lhs, a_widths = utils.make_bins(config.alpha_Fe,
                                          make_rhs=True)

        # Force raw metallicities to span the full range of the stellar metallicity grid
        if raw_a_lhs[0] > a_lhs[0]:
            raw_a_lhs[0] = a_lhs[0]
            raw_a_widths[0] = raw_a_lhs[1] - raw_a_lhs[0]
        if raw_a_lhs[-1] < a_lhs[-1]:
            raw_a_lhs[-1] = a_lhs[-1]
            raw_a_widths[-1] = raw_a_lhs[-1] - raw_a_lhs[-2]
        
        start = 0
        stop = 0

        # Loop over the new metallicity bins
        for j in range(a_lhs.shape[0] - 1):

            # Find the first raw bin partially covered by the new bin
            while raw_a_lhs[start + 1] <= a_lhs[j]:
                start += 1

            # Find the last raw bin partially covered by the new bin
            while raw_a_lhs[stop+1] < a_lhs[j + 1]:
                stop += 1

            # If new bin falls completely within one raw bin
            if stop == start:
                out_grid[:, j, :, :, :] = in_grid[:, start, :, :, :]

            # If new bin has contributions from more than one raw bin
            else:
                start_fact = ((raw_a_lhs[start + 1] - a_lhs[j])
                              / (raw_a_lhs[start + 1] - raw_a_lhs[start]))

                end_fact = ((a_lhs[j + 1] - raw_a_lhs[stop])
                            / (raw_a_lhs[stop + 1] - raw_a_lhs[stop]))

                raw_a_widths[start] *= start_fact
                raw_a_widths[stop] *= end_fact

                width_slice = raw_a_widths[start:stop + 1]

                summed = np.sum(width_slice[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
                                * in_grid[:, start:stop + 1, :, :, :], axis=2)

                out_grid[:, j, :, :, :] = summed/np.sum(width_slice)

                raw_a_widths[start] /= start_fact
                raw_a_widths[stop] /= end_fact

        return out_grid

    def spectrum(self, sfh_ceh, t_bc, logU):
        """ Obtain a 1D spectrum for a given star-formation and
        chemical enrichment history, ionization parameter and t_bc.

        parameters
        ----------

        sfh_ceh : numpy.ndarray
            2D array containing the desired star-formation and
            chemical evolution history.

        logU : float
            Log10 of the ionization parameter.

        t_bc : float
            The maximum age at which to include nebular emission.
        """

        return self._interpolate_grid(self.combined_grid, sfh_ceh, t_bc, logU)

    def line_fluxes(self, sfh_ceh, t_bc, logU):
        """ Obtain line fluxes for a given star-formation and
        chemical enrichment history, ionization parameter and t_bc.

        parameters
        ----------

        sfh_ceh : numpy.ndarray
            2D array containing the desired star-formation and
            chemical evolution history.

        logU : float
            Log10 of the ionization parameter.

        t_bc : float
            The maximum age at which to include nebular emission.
        """

        return self._interpolate_grid(self.line_grid, sfh_ceh, t_bc, logU)

    def _interpolate_grid(self, grid, sfh_ceh, t_bc, logU):
        """ Interpolates a chosen grid in logU and collapses over star-
        formation and chemical enrichment history to get 1D models. """

        t_bc *= 10**9

        if logU == config.logU[0]:
            logU += 10**-10

        spectrum_low_logU = np.zeros_like(grid[:, 0, 0, 0, 0])
        spectrum_high_logU = np.zeros_like(grid[:, 0, 0, 0, 0])

        logU_ind = config.logU[config.logU < logU].shape[0]
        logU_weight = ((config.logU[logU_ind] - logU)
                       / (config.logU[logU_ind] - config.logU[logU_ind-1]))

        index = config.age_bins[config.age_bins < t_bc].shape[0]
        weight = 1 - (config.age_bins[index] - t_bc)/config.age_widths[index-1]

        for i in range(config.alpha_Fe.shape[0]):
            for j in range(config.metallicities.shape[0]):
                if sfh_ceh[i, j, :index].sum() > 0.:
                    sfh_ceh[:, :, index-1] *= weight

                    spectrum_low_logU += np.sum(grid[:, i, j, logU_ind-1, :index]
                                                * sfh_ceh[i, j, :index], axis=1)

                    spectrum_high_logU += np.sum(grid[:, i, j, logU_ind, :index]
                                                * sfh_ceh[i, j, :index], axis=1)

                    sfh_ceh[:, :, index-1] /= weight

        spectrum = (spectrum_high_logU*(1 - logU_weight)
                    + spectrum_low_logU*logU_weight)

        return spectrum

from __future__ import print_function, division, absolute_import

import os
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM


def make_dirs(run="."):
    """ Make local Bagpipes directory structure in working dir. """

    if not os.path.exists(working_dir + "/pipes"):
        os.mkdir(working_dir + "/pipes")

    if not os.path.exists(working_dir + "/pipes/plots"):
        os.mkdir(working_dir + "/pipes/plots")

    if not os.path.exists(working_dir + "/pipes/posterior"):
        os.mkdir(working_dir + "/pipes/posterior")

    if not os.path.exists(working_dir + "/pipes/cats"):
        os.mkdir(working_dir + "/pipes/cats")

    if run != ".":
        if not os.path.exists("pipes/posterior/" + run):
            os.mkdir("pipes/posterior/" + run)

        if not os.path.exists("pipes/plots/" + run):
            os.mkdir("pipes/plots/" + run)


def make_bins(midpoints, make_rhs=False):
    """ A general function for turning an array of bin midpoints into an
    array of bin left hand side positions and bin widths. Splits the
    distance between bin midpoints equally in linear space.

    Parameters
    ----------

    midpoints : numpy.ndarray
        Array of bin midpoint positions

    make_rhs : bool
        Whether to add the position of the right hand side of the final
        bin to bin_lhs, defaults to false.
    """

    bin_widths = np.zeros_like(midpoints)

    if make_rhs:
        bin_lhs = np.zeros(midpoints.shape[0]+1)
        bin_lhs[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
        bin_widths[-1] = (midpoints[-1] - midpoints[-2])
        bin_lhs[-1] = midpoints[-1] + (midpoints[-1]-midpoints[-2])/2
        bin_lhs[1:-1] = (midpoints[1:] + midpoints[:-1])/2
        bin_widths[:-1] = bin_lhs[1:-1]-bin_lhs[:-2]

    else:
        bin_lhs = np.zeros_like(midpoints)
        bin_lhs[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
        bin_widths[-1] = (midpoints[-1] - midpoints[-2])
        bin_lhs[1:] = (midpoints[1:] + midpoints[:-1])/2
        bin_widths[:-1] = bin_lhs[1:]-bin_lhs[:-1]

    return bin_lhs, bin_widths

def convert_deepdish_group(group):
    """Convert a deepdish h5py group to a Python dictionary.
    
    Parameters
    ----------
    group : h5py.Group
        The h5py group to convert
        
    Returns
    -------
    dict
        The converted dictionary
    """
    result = {}
    
    # List of deepdish-specific attributes to ignore
    deepdish_attrs = {'CLASS', 'TITLE', 'VERSION', 'DEEPDISH_IO_VERSION'}
    
    for key in group.keys():
        item = group[key]
        
        if isinstance(item, h5py.Group):
            # Check if this is an empty group that should be a tuple
            if len(item.keys()) == 0 and 'i0' in item.attrs and 'i1' in item.attrs:
                result[key] = (item.attrs['i0'], item.attrs['i1'])
            else:
                # Regular group, recursively convert
                sub_result = {}
                for subkey in item.keys():
                    subitem = item[subkey]
                    if isinstance(subitem, h5py.Group):
                        # Check if this is a tuple group (has i0 and i1 attributes)
                        if 'i0' in subitem.attrs and 'i1' in subitem.attrs:
                            sub_result[subkey] = (subitem.attrs['i0'], subitem.attrs['i1'])
                        else:
                            # Recursively convert nested groups
                            nested_result = convert_deepdish_group(subitem)
                            if nested_result:
                                sub_result[subkey] = nested_result
                    elif isinstance(subitem, h5py.Dataset):
                        value = subitem[()]
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        sub_result[subkey] = value
                
                # Add any prior information from group attributes
                for attr_key, attr_value in item.attrs.items():
                    # Skip deepdish-specific attributes
                    if attr_key in deepdish_attrs:
                        continue
                    if attr_key.endswith('_prior'):
                        param_name = attr_key[:-6]  # Remove '_prior' suffix
                        if param_name in sub_result:
                            sub_result[attr_key] = attr_value.decode('utf-8') if isinstance(attr_value, bytes) else attr_value
                    # Handle string attributes (like 'type')
                    elif isinstance(attr_value, (bytes, str)):
                        sub_result[attr_key] = attr_value.decode('utf-8') if isinstance(attr_value, bytes) else attr_value
                    # Handle scalar attributes (like 'Q')
                    elif isinstance(attr_value, (int, float, np.number)):
                        sub_result[attr_key] = float(attr_value) if isinstance(attr_value, np.number) else attr_value
                
                if sub_result:  # Only add non-empty results
                    result[key] = sub_result
        elif isinstance(item, h5py.Dataset):
            # Convert datasets to Python scalars
            try:
                value = item[()]
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                result[key] = value
            except Exception as e:
                print(f"Error reading dataset {key}:", e)
                
    return result


# Set up necessary variables for cosmological calculations.
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
z_array = np.arange(0., 100., 0.01)
age_at_z = cosmo.age(z_array).value
ldist_at_z = cosmo.luminosity_distance(z_array).value

install_dir = os.path.dirname(os.path.realpath(__file__))
grid_dir = install_dir + "/models/grids"
working_dir = os.getcwd()

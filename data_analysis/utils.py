import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.ndimage import gaussian_filter

from scipy.ndimage import maximum_filter, label, find_objects


import scipy.signal as sig
# from MW import mw_generic as gene
# import matplotlib.pylab as plt
import lmfit

path_to_data = 'C:\\Users\\gusarov\\Raman_USC_project\\data\\'
path_to_arrays = 'C:\\Users\\gusarov\\Raman_USC_project\\data_arrays\\'

# FileName = '2024_10_11_Raman_14-1_Device3_Coil_2.hdf5'
FileName = '2024_10_11_Raman_14-1_Device3_Coil_4.hdf5'

DataType = 'Coil_VNA'

ExpName = FileName[:-5]


def LoadDataLabber(FileName, DataType, verbose=False):

    ''' Load Labber file. Returns x, y, z
    Datatypes: 'xdata_ydata'
        * 'VNA_Freq': 1D data
            * frequency along x
        * 'Coil_VNA': 
            * coil parameter along x (voltage)
            * VNA freq along y
        * 'VNA_Power':
            * VNA frequency along x
            * VNA power along y
    '''
    import Labber
    
    file = Labber.LogFile(FileName)

    if verbose==True:
            print(f'Opening file:\n   {FileName}')
            print(file.getEntry())

    if DataType == 'Coil_VNA':
        S21 = file.getData()
        V_flux = file.getStepChannels()[0]['values']
        freq = file.getTraceXY()[0]
        return V_flux, freq, S21, file
    
    elif DataType == 'VNA_Power':
        S21 = file.getData()
        VNApower = file.getStepChannels()[0]['values']
        freq = file.getTraceXY()[0]
        return freq, VNApower, np.transpose(S21), file
    
    elif DataType == 'VNA_Freq':
        S21 = file.getData()
        freq = file.getTraceXY()[0]
        return freq, S21, file
    


def get_sliced_data(S21_angle, V_flux_list, freq_list, V_flux_bounds=None, freq_bounds=None, return_grid = False, sigma=40):
    # Unpack the bounds
    if V_flux_bounds!=None:
        min_V_flux, max_V_flux = V_flux_bounds
    else:
        min_V_flux, max_V_flux = min(V_flux_list), max(V_flux_list)
    
    if freq_bounds!=None:
        min_freq, max_freq = freq_bounds
    else:
        min_freq, max_freq = min(freq_list), max(freq_list)

    # Get the indices that correspond to the boundaries for V_flux_list and freq_list
    V_flux_indices = np.where((V_flux_list >= min_V_flux) & (V_flux_list <= max_V_flux))[0]
    freq_indices = np.where((freq_list >= min_freq) & (freq_list <= max_freq))[0]

    # Slice the V_flux_list, freq_list, and S21_angle accordingly
    sliced_V_flux = V_flux_list[V_flux_indices]
    sliced_freq = freq_list[freq_indices]
    sliced_S21_angle = S21_angle[np.ix_(V_flux_indices, freq_indices)]

    if sigma!=None:
        # Apply Gaussian smoothing on S21_angle
        # for i in range(V_flux_list.shape[0]):
        sliced_S21_angle = sig.savgol_filter(sliced_S21_angle, sigma, polyorder=2, axis=1)
        # sliced_S21_angle = gaussian_filter(sliced_S21_angle, sigma=sigma)

    if return_grid:
        sliced_freq, sliced_V_flux = np.meshgrid(sliced_freq, sliced_V_flux)
        return sliced_S21_angle, sliced_V_flux, sliced_freq

    return sliced_S21_angle, sliced_V_flux, sliced_freq


# %%
import import_ipynb
import numpy as np
import pandas as pd
import itertools

from os import listdir, mkdir, stat

from scipy.signal import fftconvolve, convolve2d

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm as lognorm

from astropy.io import fits
from astropy.wcs import WCS

from glob import glob
# %%
def binary_array(num):
    variants = [[0,1],]*num
    out = np.zeros((2**num,num),bool)
    for idx, level in enumerate(itertools.product(*variants)):
        out[idx] = np.array(level,dtype=bool)
    return out
def create_array(filename,mode='Sky'):
    temp = fits.getdata(filename,1)
    if mode == 'Sky':
        return np.histogram2d(temp['Y'],temp['X'],1000,[[0, 1000], [0, 1000]])[0]
    if mode == 'Det':
        return np.histogram2d(temp['DET1Y'],temp['DET1X'],360,[[0, 360], [0, 360]])[0]
def get_wcs(file):
    header = file[1].header
    wcs = WCS({
        'CTYPE1': header['TCTYP38'], 'CTYPE2': header['TCTYP39'],
        'CUNIT1': header['TCUNI38'], 'CUNIT2': header['TCUNI39'],
        'CDELT1': header['TCDLT38'], 'CDELT2': header['TCDLT39'],
        'CRPIX1': header['TCRPX38'], 'CRPIX2': header['TCRPX39'],
        'CRVAL1': header['TCRVL38'], 'CRVAL2': header['TCRVL39'],
        'NAXIS1': header['TLMAX38'], 'NAXIS2': header['TLMAX39']
    })
    return wcs
def get_link_list(folder, sort_list=False):
    links = glob(f'{folder}\\**\\*_cl.evt',recursive=True)
    sorted_list = sorted(links, key=lambda x: stat(x).st_size) 
    return np.array(sorted_list)
def atrous(level = 0, resize = False, max_size = 1001):
    base = 1/256*np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1],
    ])
    size = 2**level * (base.shape[0]-1)+1
    output = np.zeros((size,size))
    output[::2**level, ::2**level] = base
    if resize:
        output = np.pad(output, pad_width=2**(level+1))
    if output.shape[0]>max_size:
        return output[(size-1)//2-(max_size-1)//2:(size-1)//2+(max_size-1)//2+1, (size-1)//2-(max_size-1)//2:(size-1)//2+(max_size-1)//2+1]
    return output
def gauss(level=0, resize=False, max_size = 1000):
    size = min(5*2**(level+1)+1, max_size)
    sigma = 2**(level)
    A = 1/(2*np.pi*sigma**2)**0.5
    x = A*np.exp( (-(np.arange(size)-(size-1)//2)**2)/(2*sigma**2))
    out = np.multiply.outer(x,x)
    return out
def adjecent(array):
    grid = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ])
    output = fftconvolve(array,grid,mode='same') >= 0.5
    try:
        output = np.logical_and(np.logical_and(output, np.logical_not(array)),np.logical_not(array.mask))
    except AttributeError:
        output = np.logical_and(output, np.logical_not(array))
    output = np.argwhere(output == True)
    return output[:,0], output[:,1]
def add_borders(array,middle=True):
        # array_blurred = convolve2d(array,np.ones((5,5)),mode='same')
        mask = np.zeros(array.shape)
        datax, datay = np.any(array>0,0), np.any(array>0,1)
        # datax, datay = np.any(array_blurred>0,0), np.any(array_blurred>0,1)
        #Add border masks
        x_min, y_min = np.argmax(datax), np.argmax(datay)
        x_max, y_max = len(datax) - np.argmax(datax[::-1]), len(datay) - np.argmax(datay[::-1])
        # x_mid_min, y_mid_min = x_min+10+np.argmin(datax[x_min+10:]), y_min+10+np.argmin(datay[y_min+10:])
        # x_mid_max, y_mid_max = x_max-10-np.argmin(datax[x_max-11::-1]), y_max-10-np.argmin(datay[y_max-11::-1])
        mask[y_min:y_max,x_min:x_max] = True
        if middle is True:
            mask[176:191,:] = False
            mask[:,176:191] = False
            # mask[y_mid_min:y_mid_max,:] = False
            # mask[:,x_mid_min:x_mid_max] = False
        mask = np.logical_not(mask)
        return mask
def fill_poisson(array, size_input=32):
    if not(isinstance(array,np.ma.MaskedArray)):
        print('No mask found')
        return array
    size = size_input
    output = array.data.copy()
    mask = array.mask.copy()
    while mask.sum()>1:
        kernel = np.ones((size,size))/size**2
        # coeff = fftconvolve(np.logical_not(mask), kernel, mode='same')
        coeff = fftconvolve(np.logical_not(mask),kernel,mode='same')
        mean = fftconvolve(output,kernel,mode='same')
        idx = np.where(np.logical_and(mask,coeff>0.1))
        output[idx] = np.random.poisson(np.abs(mean[idx]/coeff[idx]))
        mask[idx] = False
        size *= 2
    return output
def fill_mean(array,size_input=3):
    size = size_input
    if not(isinstance(array,np.ma.MaskedArray)):
        print('No mask found')
        return array
    output = array.filled(0)
    for i,j in zip(*np.where(array.mask)):
        output[i,j] = array[max(0,i-size):min(array.shape[0]-1,i+size+1),max(0,j-size):min(array.shape[1]-1,j+size+1)].mean()
    while np.isnan(output).any():
        size += 5
        for i,j in zip(*np.where(np.isnan(output))):
            output[i,j] = array[max(0,i-size):min(array.shape[0]-1,i+size+1),max(0,j-size):min(array.shape[1]-1,j+size+1)].mean()
    return output
def mirror(array):
    size = array.shape[0]
    output = np.tile(array,(3,3))
    output[0:size] = np.flipud(output[0:size])
    output[2*size:3*size] = np.flipud(output[2*size:3*size])
    output[:,0:size] = np.fliplr(output[:,0:size])
    output[:,2*size:3*size] = np.fliplr(output[:,2*size:3*size])
    return output
class Observation:
    def __init__(self, file_name, E_borders=[3,20]):
        self.filename = file_name
        self.name = file_name[file_name.find('nu'):].replace('_cl.evt','')
        with fits.open(file_name) as file:
            self.obs_id = file[0].header['OBS_ID']
            self.ra = file[0].header['RA_NOM']
            self.dec = file[0].header['DEC_NOM']
            self.time_start = file[0].header['TSTART']
            self.header = file[0].header
            self.det = self.header['INSTRUME'][-1]
            self.wcs = get_wcs(file)
            self.data = np.ma.masked_array(*self.get_data(file,E_borders))
            self.hard_mask = add_borders(self.data.data,middle=False)
        shift = shift = int((360-64)/2)
        self.model = (fits.getdata(f'D:\Programms\Jupyter\Science\Source_mask\Model/det1_fpm{self.det}.cxb.fits',0)
                *(180/np.pi/10150)**2)[shift:360+shift,shift:360+shift]
        self.exposure = self.header['EXPOSURE']
    def get_coeff(self):
        coeff = np.array([0.977,0.861,1.163,1.05]) if self.det=='A' else np.array([1.004,0.997,1.025,0.979])
        resized_coeff = (coeff).reshape(2,2).repeat(180,0).repeat(180,1)
        return resized_coeff
    def get_data(self, file, E_borders=[3,20]):
        PI_min, PI_max = (np.array(E_borders)-1.6)/0.04
        data = file[1].data.copy()
        idx_mask = (data['STATUS'].sum(1) == 0)        
        idx_output = np.logical_and(idx_mask,np.logical_and((data['PI']>PI_min),(data['PI']<PI_max)))
        data_output = data[idx_output]
        data_mask = data[np.logical_not(idx_mask)]
        build_hist = lambda array: np.histogram2d(array['DET1Y'],array['DET1X'],360,[[0,360],[0,360]])[0]
        output = build_hist(data_output)
        mask = build_hist(data_mask)
        mask = np.logical_or(mask,add_borders(output))
        mask = np.logical_or(mask, self.get_bad_pix(file))
        return output, mask
    def get_bad_pix(self, file):
        output = np.zeros((360,360))
        kernel = np.ones((5,5))
        for i in range(4):
            pixpos_file = fits.getdata(f'D:\\Programms\\Jupyter\\Science\\Source_mask\\Pixpos\\nu{self.det}pixpos20100101v007.fits',i+1)
            bad_pix_file = file[3+i].data.copy()
            temp = np.zeros(len(pixpos_file),dtype=bool)
            for x,y in zip(bad_pix_file['rawx'],bad_pix_file['rawy']):    
                temp = np.logical_or(temp, np.equal(pixpos_file['rawx'],x)*np.equal(pixpos_file['rawy'],y))
            temp = pixpos_file[temp] 
            output += np.histogram2d(temp['REF_DET1Y'],temp['REF_DET1X'], 360, [[0,360],[0,360]])[0]
        output = convolve2d(output, kernel, mode='same') > 0
        return output
    def wavdecomp(self, mode = 'atrous', thresh=False,occ_coeff = False):
        #THRESHOLD
        if type(thresh) is int: thresh_max, thresh_add = thresh, thresh/2
        elif type(thresh) is tuple: thresh_max, thresh_add = thresh
        #INIT NEEDED VARIABLES
        wavelet = globals()[mode]
        max_level = 8
        conv_out = np.zeros((max_level+1,self.data.shape[0],self.data.shape[1]))
        size = self.data.shape[0]
        #PREPARE ORIGINAL DATA FOR ANALYSIS: FILL THE HOLES + MIRROR + DETECTOR CORRECTION
        data = fill_poisson(self.data)
        if occ_coeff: data = data*self.get_coeff()
        data = mirror(data)
        data_bkg = data.copy()
        #ITERATIVELY CONDUCT WAVLET DECOMPOSITION
        for i in range(max_level):
            conv = fftconvolve(data,wavelet(i),mode='same')
            temp_out = data-conv
            #ERRORMAP CALCULATION
            if thresh_max != 0:
                sig = sigma(mode, i)
                bkg = fftconvolve(data_bkg, wavelet(i),mode='same')
                bkg[bkg<0] = 0
                # err = (1+np.sqrt(bkg/sig**2 + 0.75))*sig**3
                err = (1+np.sqrt(bkg+0.75))*sig
                significant = (np.abs(temp_out)> thresh_max*err)[size:2*size,size:2*size]
                # significant = (temp_out > thresh_max*err)[size:2*size,size:2*size]
                if thresh_add != 0:
                    add_significant = (np.abs(temp_out)> thresh_add*err)[size:2*size,size:2*size]
                    # add_significant = (temp_out > thresh_add*err)[size:2*size,size:2*size]
                    adj = adjecent(significant)
                    add_condition = np.logical_and(add_significant[adj[0],adj[1]],np.logical_not(significant[adj[0],adj[1]]))
                    while (add_condition).any():
                        to_add = adj[0][add_condition], adj[1][add_condition] 
                        significant[to_add[0],to_add[1]] = True
                        adj = adjecent(significant)
                        add_condition = np.logical_and(add_significant[adj[0],adj[1]],np.logical_not(significant[adj[0],adj[1]]))
                        # add_condition = np.logical_and(np.abs(temp_out)[adj[0],adj[1]] >= thresh_add*err[adj[0],adj[1]], np.logical_not(significant)[adj[0],adj[1]])
                temp_out[size:2*size,size:2*size][np.logical_not(significant)] = 0
            #WRITING THE WAVELET DECOMP LAYER
            if temp_out[size:2*size,size:2*size].sum() == 0: break
            conv_out[i] = +temp_out[size:2*size,size:2*size]
            conv_out[i][conv_out[i]<0]=0 #leave only positive data to prevent problems while summing layers
            data = conv
        conv_out[max_level] = conv[size:2*size,size:2*size]
        return conv_out
# %%

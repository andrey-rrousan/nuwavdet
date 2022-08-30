# %%
from get_region_pack import *
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import multiprocessing
from multiprocessing import get_context
import warnings
import time
import os
warnings.filterwarnings('ignore')
# %%
def ellipse(array):
    grid = np.indices(array.shape)[::-1]
    center = [((curr_axis*array).sum()/array.sum()) for curr_axis in grid]
    y,x = np.where(array)
    return center, np.abs(x-center[0]).max()*2, np.abs(y-center[1]).max()*2
def mask_ellipse(array):
    (center_x, center_y), len_x, len_y = ellipse(array)
    x,y = np.indices(array.shape)[::-1]
    radius = ((x-center_x)/(0.5*len_x))**2+((y-center_y)/(0.5*len_y))**2
    return radius <= 1
def poisson_divider(array):
    sub_arrays = np.zeros((4,180,180))
    for i in range(2):
        for j in range(2):
            sub_arrays[i+2*j] = array[180*i:180*(i+1),180*j:180*(j+1)].filled(-1000)
    pix_sum = 0
    for layer in sub_arrays:
        _masked_array = np.ma.masked_less(layer,0)
        pix_sum += ((_masked_array-_masked_array.mean())**2/_masked_array.mean()).sum()
    pix_sum /= np.logical_not(array.mask).sum()
    return pix_sum
def process(argument):
    idx, obs_name = argument
    bin_num = 6
    try:
        obs = Observation(obs_name)
        sky_coord = SkyCoord(ra=obs.ra*u.deg,dec=obs.dec*u.deg,frame='fk5').transform_to('galactic')
        lon, lat = sky_coord.l.value, sky_coord.b.value
        rem_signal, rem_area, poiss_comp, rms = np.zeros((4,2**bin_num))
        region = np.zeros(obs.data.shape, dtype=bool)
        rem_region = np.logical_and(region, np.logical_not(obs.data.mask))
        masked_obs = np.ma.masked_array(obs.data, mask = region)
        good_lvl = np.zeros(bin_num,dtype=bool)
        good_idx = 0
        if obs.exposure > 1000:
            wav_obs = obs.wavdecomp('gauss',(5,3),occ_coeff=True)
            occ_coeff = obs.get_coeff()
            for idx, lvl in enumerate(binary_array(bin_num)):
                try:
                    # region = np.array([layer>0 for layer in wav_obs[2:-1][lvl]]).sum(0).astype(bool)
                    region = wav_obs[2:-1][lvl].sum(0)>0
                except ValueError:
                    region = np.zeros(obs.data.shape,dtype=bool)
                masked_obs = np.ma.masked_array(obs.data, mask = region)*occ_coeff
                rem_region = np.logical_and(region, np.logical_not(obs.data.mask))
                rem_signal[idx] = 1-obs.data[region].sum()/obs.data.sum()
                rem_area[idx] = 1 - rem_region.sum()/np.logical_not(obs.data.mask).sum()
                # poiss_comp[idx] = poisson_divider(masked_obs)
                poiss_comp[idx] = np.mean((masked_obs-masked_obs.mean())**2/masked_obs.mean())
                rms[idx] = np.sqrt(((masked_obs-masked_obs.mean())**2).mean())
                parameter = lambda idx: ((poiss_comp[idx])**2+((1-rem_area[idx])*0.5)**2)
                if (parameter(idx)<parameter(good_idx)):
                    good_idx = idx
                    good_lvl = lvl
            try:
                # region = np.array([layer>0 for layer in wav_obs[2:-1][lvl]]).sum(0).astype(bool)
                region = wav_obs[2:-1][good_lvl].sum(0)>0
            except ValueError:
                region = np.zeros(obs.data.shape,dtype=bool)
            masked_obs = np.ma.masked_array(obs.data, mask = region)
            rem_region = np.logical_and(region, np.logical_not(obs.data.mask))
            to_table = [obs.obs_id,
                        obs.det,
                        obs.ra,
                        obs.dec,
                        lon,
                        lat,
                        obs.time_start, 
                        obs.exposure,
                        masked_obs.mean()/obs.exposure, #count rate
                        1 - rem_region.sum()/np.logical_not(obs.data.mask).sum(), #rem_area
                        poiss_comp[good_idx],
                        poiss_comp[0],
                        rms[good_idx]
                        ]
        else:
            to_table = [obs.obs_id,
                        obs.det,
                        obs.ra,
                        obs.dec,
                        lon,
                        lat,
                        obs.time_start, 
                        obs.exposure,
                        -1,
                        -1, #rem_signal
                        -1, #rem_area
                        -1,
                        -1,
                        -1
                        ]
        return to_table, region.astype(int)
    except TypeError:
        return obs_name, np.zeros((360,360))
#%%  
if __name__ == '__main__':
    start = time.perf_counter()
    processing = True
    start_new = True
    group_size = 50
    fits_folder = 'D:\Programms\Jupyter\Science\Source_mask\\\Archive\Processing_v8'
    region_folder = f'{fits_folder}\\Region'
    if not os.path.exists(fits_folder):
        os.makedirs(fits_folder)
        os.makedirs(region_folder)
    obs_list = get_link_list('E:\\Archive\\0[0-9]\\[0-9]',sort_list = True)[:]
    #FILTERING BY THE FILE SIZE
    print(f'Finished scanning folders. Found {len(obs_list)} observations.')
    table = {
        'obs_id':[], 'detector':[], 'ra':[], 'dec':[], 'lon':[], 'lat':[], 't_start':[], 'exposure':[],
        'count_rate':[], 'remaining_area':[], 'poisson_chi2':[], 'poisson_chi2_full':[], 'rms':[]
        }
    if start_new:
        out_table = pd.DataFrame(table)
        out_table.to_csv(f'{fits_folder}\\test.csv')
        # out_table.to_csv(f'{fits_folder}\\test_skipped.csv')
        os.system(f'copy D:\Programms\Jupyter\Science\Source_mask\Archive\Processing_v3\\test_skipped.csv {fits_folder}')
    #REMOVING PROCESSED OBSERVATIONS
    # already_processed = fits.getdata(f'{fits_folder}\\test.fits')['obs_name']
    already_processed_list = pd.read_csv(f'{fits_folder}\\test.csv',index_col=0,dtype={'obs_id':str})
    already_skipped_list = pd.read_csv(f'{fits_folder}\\test_skipped.csv',index_col=0,dtype={'obs_id':str})
    already_processed = (already_processed_list['obs_id'].astype(str)+already_processed_list['detector']).values
    already_skipped = (already_skipped_list['obs_id'].astype(str)+already_skipped_list['detector']).values
    obs_list_names = [curr[curr.index('nu')+2:curr.index('_cl.evt')-2] for curr in obs_list]
    not_processed = np.array([(curr not in already_processed) for curr in obs_list_names])
    not_skipped = np.array([(curr not in already_skipped) for curr in obs_list_names])
    obs_list = obs_list[np.logical_and(not_processed,not_skipped)]
    print(f'Removed already processed observations. {len(obs_list)} observations remain.')
    if processing:
        print('Started processing...')
        num = 0
        for group_idx in range(len(obs_list)//group_size+1):
            print(f'Started group {group_idx}')
            group_list = obs_list[group_size*group_idx:min(group_size*(group_idx+1),len(obs_list))]
            max_size = np.array([stat(file).st_size/2**20 for file in group_list]).max()
            process_num = 10 if max_size<50 else (5 if max_size<200 else (2 if max_size<1000 else 1))
            print(f"Max file size in group is {max_size:.2f}Mb, create {process_num} processes")
            with get_context('spawn').Pool(processes=process_num) as pool:
                for result,region in pool.imap(process,enumerate(group_list)):
                    if type(result) is np.str_:
                        obs_id = result[result.index('nu'):result.index('_cl.evt')]
                        print(f'{num:>3} is skipped. File {obs_id}')
                        num +=1
                        continue
                    for key,value in zip(table.keys(),result):
                        table[key] = [value]
                    if table['exposure'][0] < 1000:
                        print(f'{num:>3} {str(result[0])+result[1]} is skipped. Exposure < 1000')
                        pd.DataFrame(table).to_csv(f'{fits_folder}\\test_skipped.csv',mode='a',header=False)
                        num +=1
                        continue
                    pd.DataFrame(table).to_csv(f'{fits_folder}\\test.csv',mode='a',header=False)
                    fits.writeto(f'{region_folder}\\{str(result[0])+result[1]}_region.fits', region, overwrite= True)
                    print(f'{num:>3} {str(result[0])+result[1]} is written.')
                    num +=1
            print('Converting generated csv to fits file...')
            print(f'Current time in: {time.perf_counter()-start}')
            print(f'Processed {num/len(obs_list)*100:.2f} percent')
            csv_file = pd.read_csv(f'{fits_folder}\\test.csv',index_col=0,dtype={'obs_id':str})
            Table.from_pandas(csv_file).write(f'{fits_folder}\\test.fits',overwrite=True)
        print(f'Finished writing: {time.perf_counter()-start}')
# %%

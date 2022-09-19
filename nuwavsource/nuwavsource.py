# %%
import numpy as np
import itertools
from pandas import DataFrame, read_csv
from astropy.table import Table, unique
from astropy.coordinates import SkyCoord
from astropy import units as u
from multiprocessing import get_context, cpu_count
from time import perf_counter
from os import stat, makedirs
from os.path import dirname
from scipy.signal import fftconvolve, convolve2d
from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
from warnings import filterwarnings
filterwarnings('ignore')


def get_link_list(folder, sort_list=True):
    links = glob(f'{folder}\\**\\*_cl.evt', recursive=True)
    if sort_list:
        sorted_list = sorted(links, key=lambda x: stat(x).st_size)
        return np.array(sorted_list)
    else:
        return np.array(links)


def binary_array(num):
    variants = [[0, 1], ]*num
    out = np.zeros((2**num, num), bool)
    for idx, level in enumerate(itertools.product(*variants)):
        out[idx] = np.array(level, dtype=bool)
    return out


def create_array(filename, mode='Sky'):
    temp = fits.getdata(filename, 1)
    if mode == 'Sky':
        return np.histogram2d(temp['Y'],
                              temp['X'],
                              1000,
                              [[0, 1000], [0, 1000]])[0]
    if mode == 'Det':
        return np.histogram2d(temp['DET1Y'],
                              temp['DET1X'],
                              360,
                              [[0, 360], [0, 360]])[0]


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


def atrous(level=0, max_size=1001):
    base = 1/256*np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1],
    ])
    size = 2**level * (base.shape[0]-1)+1
    output = np.zeros((size, size))
    output[::2**level, ::2**level] = base
    if output.shape[0] > max_size:
        return output[(size-1)//2-(max_size-1)//2:(size-1)//2+(max_size-1)//2+1,
                      (size-1)//2-(max_size-1)//2:(size-1)//2+(max_size-1)//2+1]
    return output


def gauss(level=0, max_size=1000):
    size = min(5*2**(level+1)+1, max_size)
    sigma = 2**(level)
    A = 1/(2*np.pi*sigma**2)**0.5
    x = A*np.exp((-(np.arange(size)-(size-1)//2)**2)/(2*sigma**2))
    out = np.multiply.outer(x, x)
    return out


def adjecent(array):
    grid = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    output = fftconvolve(array, grid, mode='same') >= 0.5
    try:
        output = np.logical_and(np.logical_and(output, np.logical_not(array)),
                                np.logical_not(array.mask))
    except AttributeError:
        output = np.logical_and(output, np.logical_not(array))
    output = np.argwhere(output == True)
    return output[:, 0], output[:, 1]


def add_borders(array, middle=True):
    mask = np.zeros(array.shape)
    datax, datay = np.any(array > 0, 0), np.any(array > 0, 1)
    # Add border masks
    x_min, y_min = np.argmax(datax), np.argmax(datay)
    x_max = len(datax) - np.argmax(datax[::-1])
    y_max = len(datay) - np.argmax(datay[::-1])
    mask[y_min:y_max, x_min:x_max] = True
    if middle is True:
        mask[176:191, :] = False
        mask[:, 176:191] = False
    mask = np.logical_not(mask)
    return mask


def fill_poisson(array, size_input=32):
    if not (isinstance(array, np.ma.MaskedArray)):
        print('No mask found')
        return array
    size = size_input
    output = array.data.copy()
    mask = array.mask.copy()
    while mask.sum() > 1:
        kernel = np.ones((size, size))/size**2
        coeff = fftconvolve(np.logical_not(mask), kernel, mode='same')
        mean = fftconvolve(output, kernel, mode='same')
        idx = np.where(np.logical_and(mask, coeff > 0.1))
        output[idx] = np.random.poisson(np.abs(mean[idx]/coeff[idx]))
        mask[idx] = False
        size *= 2
    return output


def mirror(array):
    size = array.shape[0]
    output = np.tile(array, (3, 3))
    output[0:size] = np.flipud(output[0:size])
    output[2*size:3*size] = np.flipud(output[2*size:3*size])
    output[:, 0:size] = np.fliplr(output[:, 0:size])
    output[:, 2*size:3*size] = np.fliplr(output[:, 2*size:3*size])
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
            self.data = np.ma.masked_array(*self.get_data(file, E_borders))
            self.hard_mask = add_borders(self.data.data, middle=False)
        self.exposure = self.header['EXPOSURE']

    def get_coeff(self):
        coeff = np.array([0.977, 0.861, 1.163, 1.05]) if self.det == 'A' else np.array([1.004, 0.997, 1.025, 0.979])
        resized_coeff = (coeff).reshape(2, 2).repeat(180, 0).repeat(180, 1)
        return resized_coeff

    def get_data(self, file, E_borders=[3, 20]):
        PI_min, PI_max = (np.array(E_borders)-1.6)/0.04
        data = file[1].data.copy()
        idx_mask = (data['STATUS'].sum(1) == 0)        
        idx_output = np.logical_and(idx_mask, np.logical_and((data['PI'] > PI_min), (data['PI'] < PI_max)))
        data_output = data[idx_output]
        data_mask = data[np.logical_not(idx_mask)]
        build_hist = lambda array: np.histogram2d(array['DET1Y'], array['DET1X'], 360, [[0, 360], [0, 360]])[0]
        output = build_hist(data_output)
        mask = build_hist(data_mask)
        mask = np.logical_or(mask, add_borders(output))
        mask = np.logical_or(mask, self.get_bad_pix(file))
        return output, mask

    def get_bad_pix(self, file):
        output = np.zeros((360, 360))
        kernel = np.ones((5, 5))
        for i in range(4):
            current_dir = dirname(__file__)
            pixpos_file = fits.getdata(f'{current_dir}\\pixpos\\nu{self.det}pixpos20100101v007.fits',i+1)
            bad_pix_file = file[3+i].data.copy()
            temp = np.zeros(len(pixpos_file), dtype=bool)
            for x, y in zip(bad_pix_file['rawx'], bad_pix_file['rawy']):    
                temp = np.logical_or(temp, np.equal(pixpos_file['rawx'], x)*np.equal(pixpos_file['rawy'], y))
            temp = pixpos_file[temp] 
            output += np.histogram2d(temp['REF_DET1Y'], temp['REF_DET1X'], 360, [[0, 360],[0, 360]])[0]
        output = convolve2d(output, kernel, mode='same') > 0
        return output

    def wavdecomp(self, mode='gauss', thresh=False, occ_coeff=False):
        # THRESHOLD
        if type(thresh) is int:
            thresh_max, thresh_add = thresh, thresh/2
        elif type(thresh) is tuple:
            thresh_max, thresh_add = thresh
        # INIT NEEDED VARIABLES
        wavelet = globals()[mode]
        max_level = 8
        conv_out = np.zeros((max_level+1, self.data.shape[0], self.data.shape[1]))
        size = self.data.shape[0]
        # PREPARE ORIGINAL DATA FOR ANALYSIS: FILL THE HOLES + MIRROR + DETECTOR CORRECTION
        data = fill_poisson(self.data)
        if occ_coeff:
            data = data*self.get_coeff()
        data = mirror(data)
        data_bkg = data.copy()
        # ITERATIVELY CONDUCT WAVLET DECOMPOSITION
        for i in range(max_level):
            conv = fftconvolve(data, wavelet(i), mode='same')
            temp_out = data-conv
            # ERRORMAP CALCULATION
            if thresh_max != 0:
                sig = ((wavelet(i)**2).sum())**0.5
                bkg = fftconvolve(data_bkg, wavelet(i), mode='same')
                bkg[bkg < 0] = 0
                err = (1+np.sqrt(bkg+0.75))*sig
                significant = (temp_out > thresh_max*err)[size:2*size, size:2*size]
                if thresh_add != 0:
                    add_significant = (temp_out > thresh_add*err)[size:2*size, size:2*size]
                    adj = adjecent(significant)
                    add_condition = np.logical_and(add_significant[adj[0], adj[1]],
                                                   np.logical_not(significant[adj[0], adj[1]]))
                    while (add_condition).any():
                        to_add = adj[0][add_condition], adj[1][add_condition] 
                        significant[to_add[0], to_add[1]] = True
                        adj = adjecent(significant)
                        add_condition = np.logical_and(add_significant[adj[0], adj[1]],
                                                       np.logical_not(significant[adj[0],adj[1]]))
                temp_out[size:2*size, size:2*size][np.logical_not(significant)] = 0
            # WRITING THE WAVELET DECOMP LAYER
            conv_out[i] = +temp_out[size:2*size, size:2*size]
            # DISCARDING NEGATIVE COMPONENTS OF WAVELETS TO MAKE MASK BY SUMMING WAVELET LAYERS
            conv_out[i][conv_out[i] < 0] = 0 
            data = conv
        conv_out[max_level] = conv[size:2*size, size:2*size]
        return conv_out

    def region_to_raw(self, region):
        x_region, y_region = np.where(region)
        tables = []
        for i in range(4):
            current_dir = dirname(__file__)
            pixpos = Table(fits.getdata(f'{current_dir}\\pixpos\\nu{self.det}pixpos20100101v007.fits', i+1))
            pixpos = pixpos[pixpos['REF_DET1X'] != -1]
            test = np.zeros(len(pixpos['REF_DET1X']), dtype=bool)
            for idx, (x, y) in enumerate(zip(pixpos['REF_DET1X'], pixpos['REF_DET1Y'])):
                test[idx] = np.logical_and(np.equal(x, x_region), np.equal(y, y_region)).any()
            table = Table({'RAWX': pixpos['RAWX'][test], 'RAWY': pixpos['RAWY'][test]})
            if not table:
                tables.append(table)
            else:
                tables.append(unique(table))
        hdu_list = fits.HDUList([
            fits.PrimaryHDU(),
            fits.table_to_hdu(tables[0]),
            fits.table_to_hdu(tables[1]),
            fits.table_to_hdu(tables[2]),
            fits.table_to_hdu(tables[3]),
        ])
        return hdu_list


def process(args):
    """
    Creates a mask using wavelet decomposition and produces some statistical and metadata about the passed observation.
    args must contain two arguments: path to the file of interest and threshold, e.g. ('D:\Data\obs_cl.evt',(5,2)) 
    """
    obs_path, thresh = args
    bin_num = 6
    try:
        obs = Observation(obs_path)
        sky_coord = SkyCoord(ra=obs.ra*u.deg, dec=obs.dec*u.deg, frame='fk5').transform_to('galactic')
        lon, lat = sky_coord.l.value, sky_coord.b.value
        rem_signal, rem_area, poiss_comp, rms = np.zeros((4, 2**bin_num))
        region = np.zeros(obs.data.shape, dtype=bool)
        region_raw = -1
        rem_region = np.logical_and(region, np.logical_not(obs.data.mask))
        masked_obs = np.ma.masked_array(obs.data, mask=region)
        good_lvl = np.zeros(bin_num, dtype=bool)
        good_idx = 0
        if obs.exposure > 1000:
            wav_obs = obs.wavdecomp('gauss', thresh, occ_coeff=True)
            occ_coeff = obs.get_coeff()
            for idx, lvl in enumerate(binary_array(bin_num)):
                try:
                    region = wav_obs[2:-1][lvl].sum(0) > 0
                except ValueError:
                    region = np.zeros(obs.data.shape, dtype=bool)
                masked_obs = np.ma.masked_array(obs.data, mask=region)*occ_coeff
                rem_region = np.logical_and(region, np.logical_not(obs.data.mask))
                rem_signal[idx] = 1-obs.data[region].sum()/obs.data.sum()
                rem_area[idx] = 1 - rem_region.sum()/np.logical_not(obs.data.mask).sum()
                poiss_comp[idx] = np.mean((masked_obs-masked_obs.mean())**2/masked_obs.mean())
                rms[idx] = np.sqrt(((masked_obs-masked_obs.mean())**2).mean())
                parameter = lambda idx: ((poiss_comp[idx])**2+((1-rem_area[idx])*0.5)**2)
                if (parameter(idx) < parameter(good_idx)):
                    good_idx = idx
                    good_lvl = lvl
            try:
                region = wav_obs[2:-1][good_lvl].sum(0) > 0
                if region.sum() > 0:
                    region_raw = obs.region_to_raw(region.astype(int))
            except ValueError:
                region = np.zeros(obs.data.shape, dtype=bool)
            masked_obs = np.ma.masked_array(obs.data, mask=region)
            rem_region = np.logical_and(region, np.logical_not(obs.data.mask))
            to_table = [obs.obs_id,
                        obs.det,
                        obs.ra,
                        obs.dec,
                        lon,
                        lat,
                        obs.time_start, 
                        obs.exposure,
                        masked_obs.mean()/obs.exposure,  # count rate
                        1 - rem_region.sum()/np.logical_not(obs.data.mask).sum(),  # rem_area
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
                        -1,  # rem_signal
                        -1,  # rem_area
                        -1,
                        -1,
                        -1
                        ]
        return to_table, region.astype(int), region_raw
    except TypeError:
        return obs_path, -1, -1


def process_folder(input_folder=None, start_new_file=None, fits_folder=None, thresh=None):
    # DIALOGUE
    if not (input_folder):
        print('Enter path to the input folder')
        input_folder = input()
    if not (start_new_file):
        print('Create new file for this processing? y/n')
        start_new_file = input()
    if start_new_file == 'y':
        start_new = True
    elif start_new_file == 'n':
        start_new = False
    else:
        print('Cannot interprete input, closing script')
        raise SystemExit(0)
    if not (fits_folder):
        print(f'Enter path to the output folder')
        fits_folder = input()
    region_folder = f'{fits_folder}\\Region'
    region_raw_folder = f'{fits_folder}\\Region_raw'
    if not thresh:
        print('Enter threshold values for wavelet decomposition:')
        print('General threshold:')
        _thresh_max = float(input())
        print('Additional threshold:')
        _thresh_add = float(input())
        thresh = (_thresh_max, _thresh_add)
    # CREATE ALL NECESSARY FILES AND VARIBALES
    obs_list = get_link_list(input_folder, sort_list=True)
    start = perf_counter()
    group_size = 50
    makedirs(region_folder, exist_ok=True)
    makedirs(region_raw_folder, exist_ok=True)
    # FILTERING BY THE FILE SIZE
    print(f'Finished scanning folders. Found {len(obs_list)} observations.')
    table = {
        'obs_id': [], 'detector': [], 'ra': [], 'dec': [],
        'lon': [], 'lat': [], 't_start': [], 'exposure': [],
        'count_rate': [], 'remaining_area': [], 'poisson_chi2': [],
        'poisson_chi2_full': [], 'rms': []
        }
    if start_new:
        out_table = DataFrame(table)
        out_table.to_csv(f'{fits_folder}\\test.csv')
        out_table.to_csv(f'{fits_folder}\\test_skipped.csv')
    # FILTERING OUT PROCESSED OBSERVATIONS
    already_processed_list = read_csv(f'{fits_folder}\\test.csv', index_col=0, dtype={'obs_id':str})
    already_skipped_list = read_csv(f'{fits_folder}\\test_skipped.csv', index_col=0, dtype={'obs_id':str})
    already_processed = (already_processed_list['obs_id'].astype(str)+already_processed_list['detector']).values
    already_skipped = (already_skipped_list['obs_id'].astype(str)+already_skipped_list['detector']).values
    obs_list_names = [curr[curr.index('nu')+2:curr.index('_cl.evt')-2] for curr in obs_list]
    not_processed = np.array([(curr not in already_processed) for curr in obs_list_names])
    not_skipped = np.array([(curr not in already_skipped) for curr in obs_list_names])
    obs_list = obs_list[np.logical_and(not_processed, not_skipped)]
    print(f'Removed already processed observations. {len(obs_list)} observations remain.')
    # START PROCESSING
    print('Started processing...')
    num = 0
    for group_idx in range(len(obs_list)//group_size+1):
        print(f'Started group {group_idx}')
        group_list = obs_list[group_size*group_idx:min(group_size*(group_idx+1), len(obs_list))]
        max_size = np.array([stat(file).st_size/2**20 for file in group_list]).max()
        process_num = cpu_count() if max_size < 50 else (cpu_count()//2 if max_size < 200 else (cpu_count()//4 if max_size < 1000 else 1))
        print(f"Max file size in group is {max_size:.2f}Mb, create {process_num} processes")
        with get_context('spawn').Pool(processes=process_num) as pool:
            packed_args = map(lambda _: (_, thresh), group_list)
            for result, region, region_raw in pool.imap(process, packed_args):
                if type(result) is np.str_:
                    obs_id = result[result.index('nu'):result.index('_cl.evt')]
                    print(f'{num:>3} is skipped. File {obs_id}')
                    num += 1
                    continue
                for key, value in zip(table.keys(), result):
                    table[key] = [value]
                if table['exposure'][0] < 1000:
                    print(f'{num:>3} {str(result[0])+result[1]} is skipped. Exposure < 1000')
                    DataFrame(table).to_csv(f'{fits_folder}\\test_skipped.csv', mode='a', header=False)
                    num +=1
                    continue
                DataFrame(table).to_csv(f'{fits_folder}\\test.csv', mode='a', header=False)
                fits.writeto(f'{region_folder}\\{str(result[0])+result[1]}_region.fits', region, overwrite=True)
                if region_raw != -1:
                    region_raw.writeto(f'{region_raw_folder}\\{str(result[0])+result[1]}_reg_raw.fits', overwrite=True)
                print(f'{num:>3} {str(result[0])+result[1]} is written.')
                num +=1
        print('Converting generated csv to fits file...')
        print(f'Current time in: {(perf_counter()-start):.2f}')
        print(f'Processed {num/len(obs_list)*100:.2f} percent')
        csv_file = read_csv(f'{fits_folder}\\test.csv', index_col=0, dtype={'obs_id': str})
        Table.from_pandas(csv_file).write(f'{fits_folder}\\test.fits', overwrite=True)
    print(f'Finished writing: {perf_counter()-start}')

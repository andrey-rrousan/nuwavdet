import itertools
import numpy as np
import os

from pandas import DataFrame, read_csv
from scipy.signal import fftconvolve

from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from time import perf_counter
from multiprocessing import get_context, cpu_count
from glob import glob
from warnings import filterwarnings
filterwarnings('ignore')


def get_link_list(folder: str, sort_list: bool = True) -> list[str]:
    """
    Returns array of paths to all *_cl.evt files in the directory recursively. 
    """
    links = glob(os.path.join(folder, '**', '*_cl.evt'), recursive=True)
    if sort_list:
        sorted_list = sorted(links, key=lambda x: os.stat(x).st_size)
        return np.array(sorted_list)
    else:
        return np.array(links)


def binary_array(num: int) -> list[list[bool]]:
    """
    Returns list of all possible combinations of num of bool values.
    """
    variants = [[0, 1], ]*num
    out = np.zeros((2**num, num), bool)
    for idx, level in enumerate(itertools.product(*variants)):
        out[idx] = np.array(level, dtype=bool)
    return out


def create_array(file_path: str, mode: str = 'Sky') -> list[int]:
    """
    Returns a 2d array of counts for given observation file.
    Modes 'Sky' and 'Det' return arrays in (X,Y) and DET1 respectively.
    """
    temp = fits.getdata(file_path, 1)
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
    """
    Returns WCS for given observation.
    Note that argument here is an opened fits file, not a path.
    """
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


def atrous(level: int = 0, max_size: int = 1001) -> list[list[float]]:
    """
    Returns a trou kernel with the size 2**level and corresponding shape.
    """
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


def atrous_sig(level: int = 0) -> float:
    # sig_values = [0.8908, 0.20066, 0.08551, 0.04122, 0.02042]
    sig_values = [0.8725, 0.1893, 0.0946, 0.0473, 0.0237]
    if level < 5:
        return sig_values[level]
    else:
        return sig_values[4]/2**(level-4)


def gauss(level: int = 0, max_size: int = 1000) -> list[list[float]]:
    """
    Returns gaussian kernel with sigma = 2**level
    """
    size = min(5*2**(level+1)+1, max_size)
    sigma = 2**(level)
    A = 1/(2*np.pi*sigma**2)**0.5
    x = A*np.exp((-(np.arange(size)-(size-1)//2)**2)/(2*sigma**2))
    out = np.multiply.outer(x, x)
    return out


def adjecent(array):
    """
    Returns two lists of indices of cells adjecent or diagonal to non-zero cells of given array
    """
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
    return output


def add_borders(array, middle=True):
    """
    Returns border mask for an DET1 observation array
    """
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


def fill_poisson(array, size_input=15):
    """
    Fills all masked elements of an array with poisson signal with local expected value.
    """
    if not (isinstance(array, np.ma.MaskedArray)):
        print('No mask found')
        return array
    size = size_input
    output = array.data.copy()
    mask = array.mask.copy()
    mask_full = np.ones(mask.shape)
    while mask.sum() > 1:
        kernel = np.ones((size, size))/size**2
        coeff_full = fftconvolve(mask_full, kernel, mode='same')
        coeff = fftconvolve(np.logical_not(mask), kernel, mode='same') / coeff_full
        mean = fftconvolve(output, kernel, mode='same')
        idx = np.where(np.logical_and(mask, coeff > 0.7))
        output[idx] = np.random.poisson(np.abs(mean[idx]/coeff[idx]))
        mask[idx] = False
        size += size_input
        size += (1 - size % 2)
    return output


def count_binning(array, count_per_bin: int = 2):
    _array = (array[array.mask == False]
              if hasattr(array, 'mask') else array)
    _array = (_array.flatten()
              if array.ndim != 1 else _array)

    bin_sum, bin_size = [], []
    _sum, _count = 0, 0

    for el in _array:
        _sum += el
        _count += 1
        if _sum >= count_per_bin:
            bin_sum.append(_sum)
            bin_size.append(_count)
            _sum, _count = 0, 0

    return np.array(bin_sum), np.array(bin_size)


def cstat(expected, data: list, count_per_bin: int = 2) -> float:
    _data = data.flatten()
    _data = _data[_data.mask == False]
    _expected = expected
    c_stat = 0
    bin_sum_array, bin_count_array = count_binning(_data, count_per_bin)
    bin_exp_array = bin_count_array * _expected
    c_stat = 2 * (bin_exp_array - bin_sum_array + bin_sum_array * (np.log(bin_sum_array / bin_exp_array))).mean()
    return c_stat


class Observation:
    """
    Main class, contains information about the observation given.
    """
    def __init__(self, file_name, E_borders=[3, 20]):
        self.filename = file_name
        self.name = file_name[file_name.find('nu'):].replace('_cl.evt', '')
        with fits.open(file_name) as file:
            self.obs_id = file[0].header['OBS_ID']
            self.ra = file[0].header['RA_NOM']
            self.dec = file[0].header['DEC_NOM']
            self.time_start = file[0].header['TSTART']
            self.header = file[0].header
            self.exposure = self.header['EXPOSURE']
            self.det = self.header['INSTRUME'][-1]
            self.wcs = get_wcs(file)
            self.data = np.ma.masked_array(*self.get_data(file, E_borders))

    def get_coeff(self):
        """
        Returns normalalizing coefficients for different chips of the observation detector.
        Coefficients are obtained from stacked observations in OCC mode.
        """
        coeff = np.array([0.977, 0.861, 1.163, 1.05]) if self.det == 'A' else np.array([1.004, 0.997, 1.025, 0.979])
        resized_coeff = (coeff).reshape(2, 2).repeat(180, 0).repeat(180, 1)
        return resized_coeff

    def get_data(self, file, E_borders=[3, 20], generate_mask=True):
        """
        Returns masked array with DET1 image data for given energy band.
        Mask is created from observations badpix tables and to mask the border and gaps. 
        """
        PI_min, PI_max = (np.array(E_borders)-1.6)/0.04
        data = file[1].data.copy()
        idx_mask = (data['STATUS'].sum(1) == 0)        
        idx_output = np.logical_and(idx_mask, np.logical_and((data['PI'] > PI_min), (data['PI'] < PI_max)))
        data_output = data[idx_output]
        data_mask = data[np.logical_not(idx_mask)]
        build_hist = lambda array: np.histogram2d(array['DET1Y'], array['DET1X'], 360, [[0, 360], [0, 360]])[0]
        output = build_hist(data_output)
        if generate_mask:
            mask = build_hist(data_mask)
            mask = np.logical_or(mask, add_borders(output))
            mask = np.logical_or(mask, self.get_bad_pix(file))
            return output, mask
        return output

    def get_bad_pix(self, file, threshold=0.9):
        """
        Creates a mask for observation based on badpix tables.
        """
        current_dir = os.path.dirname(__file__)
        output = np.ones((360, 360))
        for det_id in range(4):
            badpix = file[3 + det_id].data
            badpix_exp = (badpix['TIME_STOP'] - badpix['TIME'])/self.exposure
            pixpos = np.load(os.path.join(current_dir, 'pixpos', f'ref_pix{self.det}{det_id}.npy'), allow_pickle=True).item()
            for raw_x, raw_y, exp in zip(badpix['RAWX'], badpix['RAWY'], badpix_exp):
                y, x = pixpos[(raw_x, raw_y)]
                output[x-3:x+11, y-3:y+11] -= exp
        output = np.clip(output, a_min=0, a_max=None)
        self.norm_exp_map = output
        return output < threshold

    def exposure_corr(self, array):
        corr = 1 - self.norm_exp_map
        corr[corr > 0.1] = 0.
        correction_poiss = np.random.poisson(corr*array, corr.shape)
        return array + correction_poiss

    def wavdecomp(self, mode='gauss', thresh=False, occ_coeff=False):
        """
        Performs a wavelet decomposition of image.
        """
        # THRESHOLD
        if type(thresh) is int:
            thresh_max, thresh_add = thresh, thresh/2
        elif type(thresh) is tuple:
            thresh_max, thresh_add = thresh
        # INIT NEEDED VARIABLES
        wavelet = globals()[mode]
        max_level = 8
        conv_out = np.zeros(
            (max_level+1, self.data.shape[0], self.data.shape[1])
        )
        size = self.data.shape[0]
        # PREPARE ORIGINAL DATA FOR ANALYSIS: FILL THE HOLES + MIRROR + DETECTOR CORRECTION
        data = self.exposure_corr(self.data)
        data = fill_poisson(self.data)
        if occ_coeff:
            data = data*self.get_coeff()
        data = np.pad(data, data.shape[0], mode='reflect')
        # ITERATIVELY CONDUCT WAVLET DECOMPOSITION
        for i in range(max_level):
            conv = fftconvolve(data, wavelet(i), mode='same')
            conv[conv < 0] = 0
            temp_out = data-conv
            # ERRORMAP CALCULATION
            if thresh_max != 0:
                sig = atrous_sig(i)
                bkg = fftconvolve(data, wavelet(i), mode='same')
                bkg[bkg < 0] = 0
                err = (1+np.sqrt(bkg+0.75))*sig
                significant = (temp_out > thresh_max*err)[size:2*size, size:2*size]
                if thresh_add != 0:
                    add_significant = (temp_out > thresh_add*err)[size:2*size, size:2*size]
                    adj = adjecent(significant)
                    add_condition = np.logical_and(adj, add_significant)
                    while (add_condition).any():
                        significant = np.logical_or(significant, add_condition)
                        adj = adjecent(significant)
                        add_condition = np.logical_and(adj, add_significant)
                significant = np.pad(significant, significant.shape[0], mode='reflect')
                temp_out[np.logical_not(significant)] = 0
            # WRITING THE WAVELET DECOMP LAYER
            conv_out[i] = +temp_out[size:2*size, size:2*size]
            data = conv
        conv_out[max_level] = conv[size:2*size, size:2*size]
        return conv_out

    def region_to_raw(self, region):
        """
        Returns a hdu_list with positions of masked pixels in RAW coordinates.
        """
        y_region, x_region = np.where(region)
        hdus = []
        for i in range(4):
            current_dir = os.path.dirname(__file__)
            pixpos = Table(fits.getdata(os.path.join(current_dir, 'pixpos', f'nu{self.det}pixpos20100101v007.fits'), i+1))
            pixpos = pixpos[pixpos['REF_DET1X'] != -1]
            
            ref_condition = np.zeros(len(pixpos['REF_DET1X']), dtype=bool)
            for idx, (x, y) in enumerate(zip(pixpos['REF_DET1X'], pixpos['REF_DET1Y'])):
                ref_condition[idx] = np.logical_and(np.equal(x, x_region), np.equal(y, y_region)).any()
            
            positions = np.array((pixpos['RAWX'][ref_condition], pixpos['RAWY'][ref_condition]))
            if sum(ref_condition) != 0:
                positions = np.unique(positions, axis=1)
            rawx, rawy = positions[0], positions[1]

            time_start = float(78913712)
            bad_flag = np.zeros(16, dtype=bool)
            bad_flag[13] = 1

            columns = []
            columns.append(fits.Column('TIME', '1D', 's', array=len(rawx) * [time_start]))
            columns.append(fits.Column('RAWX', '1B', 'pixel', array=rawx))
            columns.append(fits.Column('RAWY', '1B', 'pixel', array=rawy))
            columns.append(fits.Column('BADFLAG', '16X', array=len(rawx) * [bad_flag]))

            hdu = fits.BinTableHDU.from_columns(columns)
            naxis1, naxis2 = hdu.header['NAXIS1'], hdu.header['NAXIS2']
            hdu.header = fits.Header.fromtextfile(os.path.join(current_dir, 'badpix_headers', f'nu{self.det}userbadpixDET{i}.txt'))
            hdu.header['NAXIS1'] = naxis1
            hdu.header['NAXIS2'] = naxis2
            hdus.append(hdu)

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header = fits.Header.fromtextfile(os.path.join(current_dir, 'badpix_headers', f'nu{self.det}userbadpix_main.txt'))
        hdu_list = fits.HDUList([
            primary_hdu,
            *hdus
        ])
        return hdu_list


def save_region(region, path, overwrite=False):
    """
    Converts region from numpy mask notation (1 for masked, 0 otherwise)
    to standart notation (0 for masked, 1 otherwise).
    Saves the region as fits file according to given path.
    """
    fits.writeto(f'{path}',
                 np.logical_not(region).astype(int),
                 overwrite=overwrite)


def process(obs_path, thresh):
    """
    Creates a mask using wavelet decomposition and produces some stats
    and metadata about the passed observation.
    Arguments: path to the file of interest and threshold,
    e.g. process('D:\\Data\\obs_cl.evt', (3, 2))
    """
    bin_num = 6

    table = {
        'obs_id': [], 'detector': [], 'ra': [], 'dec': [],
        'lon': [], 'lat': [], 't_start': [], 'exposure': [],
        'count_rate': [], 'remaining_area': [], 'cash_stat': [],
        'cash_stat_full': []
    }

    try:
        obs = Observation(obs_path)
        sky_coord = SkyCoord(ra=obs.ra*u.deg,
                             dec=obs.dec*u.deg,
                             frame='fk5').transform_to('galactic')
        lon, lat = sky_coord.l.value, sky_coord.b.value
        rem_signal, rem_area, poiss_comp, rms = np.zeros((4, 2**bin_num))
        region = np.zeros(obs.data.shape, dtype=bool)
        region_raw = -1
        rem_region = np.logical_and(region, np.logical_not(obs.data.mask))
        masked_obs = np.ma.masked_array(obs.data, mask=region)
        good_lvl = np.zeros(bin_num, dtype=bool)
        good_idx = 0
        if obs.exposure > 1000:
            wav_obs = obs.wavdecomp('atrous', thresh, occ_coeff=True)
            wav_sum = wav_obs[2:-1].sum(0)
            occ_coeff = obs.get_coeff()
            binary_arr = binary_array(bin_num)

            for idx, lvl in enumerate(binary_arr):
                try:
                    region = wav_obs[2:-1][lvl].sum(0) > 0
                except ValueError:
                    region = np.zeros(obs.data.shape, dtype=bool)

                masked_obs = np.ma.masked_array(obs.data,
                                                mask=region) * occ_coeff
                rem_region = np.logical_and(region,
                                            np.logical_not(obs.data.mask))
                rem_signal[idx] = 1-obs.data[region].sum()/obs.data.sum()
                rem_area[idx] = 1 - rem_region.sum() / np.logical_not(obs.data.mask).sum()
                poiss_comp[idx] = cstat(masked_obs.mean(), masked_obs)
                rms[idx] = np.sqrt(((masked_obs-masked_obs.mean())**2).mean())

            for idx in range(len(poiss_comp)):
                if ((poiss_comp[idx] < poiss_comp[good_idx]) and
                    (poiss_comp[idx] < poiss_comp[-1] + 0.05) and
                    (rem_area[idx] > rem_area[-1])):
                    good_idx = idx
            if good_idx == 0:
                good_idx = len(binary_arr) - 1
            good_lvl = binary_arr[good_idx]

            try:
                region = wav_obs[2:-1][good_lvl].sum(0) > 0
                region_raw = obs.region_to_raw(region.astype(int))
            except ValueError:
                region = np.zeros(obs.data.shape, dtype=bool)
                region_raw = obs.region_to_raw(region.astype(int))
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
                        ]

        else:
            wav_sum = np.zeros((360, 360))
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
                        ]

        for key, value in zip(table.keys(), to_table):
            table[key] = [value]
        return table, region.astype(int), region_raw, wav_sum
    except TypeError:
        return obs_path, -1, -1, -1


def _process_multi(args):
    return process(*args)


def process_folder(input_folder=None, start_new_file=None, fits_folder=None,
                   thresh=None):
    """
    Generates a fits-table of parameters, folder with mask images in DET1 and
    BADPIX tables in RAW for all observations in given folder.
    Note that observations with exposure < 1000 sec a skipped.
    start_new_file can be either 'y' or 'n'.
    thresh must be a tuple, e.g. (5,2).
    """
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
        print('Enter path to the output folder')
        fits_folder = input()
    
    region_folder = os.path.join(fits_folder, 'Region')
    region_raw_folder = os.path.join(fits_folder, 'Region_raw')
    wav_sum_folder = os.path.join(fits_folder, 'Wav_sum')

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
    os.makedirs(region_folder, exist_ok=True)
    os.makedirs(region_raw_folder, exist_ok=True)
    os.makedirs(wav_sum_folder, exist_ok=True)
    # FILTERING BY THE FILE SIZE
    print(f'Finished scanning folders. Found {len(obs_list)} observations.')
    table = {
        'obs_id': [], 'detector': [], 'ra': [], 'dec': [],
        'lon': [], 'lat': [], 't_start': [], 'exposure': [],
        'count_rate': [], 'remaining_area': [], 'cash_stat': [],
        'cash_stat_full': []
        }
    if start_new:
        out_table = DataFrame(table)
        out_table.to_csv(os.path.join(fits_folder, 'overview.csv'))
        out_table.to_csv(os.path.join(fits_folder, 'overview_skipped.csv'))
    # FILTERING OUT PROCESSED OBSERVATIONS
    already_processed_list = read_csv(
        os.path.join(fits_folder, 'overview.csv'), index_col=0, dtype={'obs_id': str}
    )
    already_skipped_list = read_csv(
        os.path.join(fits_folder, 'overview_skipped.csv'), index_col=0, dtype={'obs_id': str}
    )

    already_processed = (
        already_processed_list['obs_id'].astype(str) +
        already_processed_list['detector']
    ).values
    already_skipped = (
        already_skipped_list['obs_id'].astype(str) +
        already_skipped_list['detector']
    ).values

    obs_list_names = [
        curr[curr.index('nu')+2:curr.index('_cl.evt')-2]
        for curr in obs_list
    ]
    not_processed = np.array([
        (curr not in already_processed)
        for curr in obs_list_names
    ])
    not_skipped = np.array([
        (curr not in already_skipped)
        for curr in obs_list_names
    ])

    obs_list = obs_list[np.logical_and(not_processed, not_skipped)]
    print('Removed already processed observations.',
          f'{len(obs_list)} observations remain.')
    # START PROCESSING
    print('Started processing...')
    num = 0
    for group_idx in range(len(obs_list)//group_size+1):
        print(f'Started group {group_idx}')
        group_list = obs_list[group_size*group_idx:min(group_size*(group_idx+1), len(obs_list))]
        max_size = np.array([
            os.stat(file).st_size/2**20
            for file in group_list
        ]).max()
        process_num = (cpu_count() if max_size < 50 else (cpu_count()//2 if max_size < 200 else (cpu_count()//4 if max_size < 1000 else 1)))
        print(f"Max file size in group is {max_size:.2f}Mb, create {process_num} processes")
        with get_context('spawn').Pool(processes=process_num) as pool:
            packed_args = map(lambda _: (_, thresh), group_list)
            for result, region, region_raw, wav_sum in pool.imap(_process_multi, packed_args):
                if type(result) is np.str_:
                    obs_id = result[result.index('nu'):result.index('_cl.evt')]
                    print(f'{num:>3} is skipped. File {obs_id}')
                    num += 1
                    continue

                obs_name = str(result['obs_id'][0])+result['detector'][0]
                if result['exposure'][0] < 1000:
                    print(f'{num:>3} {obs_name} is skipped. Exposure < 1000')
                    DataFrame(result).to_csv(os.path.join(fits_folder, 'overview_skipped.csv'), mode='a', header=False)
                    num += 1
                    continue

                DataFrame(result).to_csv(os.path.join(fits_folder, 'overview.csv'), mode='a', header=False)
                save_region(region, os.path.join(region_folder, f'{obs_name}_region.fits'), overwrite=True)
                region_raw.writeto(os.path.join(region_raw_folder, f'{obs_name}_reg_raw.fits'), overwrite=True)
                fits.writeto(os.path.join(wav_sum_folder, f'{obs_name}_wav_sum.fits'), wav_sum, overwrite=True)

                print(f'{num:>3} {obs_name} is written.')
                num += 1
        print('Converting generated csv to fits file...')
        print(f'Current time in: {(perf_counter()-start):.2f}')
        print(f'Processed {num/len(obs_list)*100:.2f} percent')
        csv_file = read_csv(os.path.join(fits_folder, 'overview.csv'), index_col=0, dtype={'obs_id': str})
        Table.from_pandas(csv_file).write(os.path.join(fits_folder, 'overview.fits'), overwrite=True)
    print(f'Finished writing: {perf_counter()-start}')

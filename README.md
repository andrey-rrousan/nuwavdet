# nuwavdet

This pacakge is used to generate region masks separating any focused X-ray flux from background signal in NuSTAR observations.

## Installation
This package is to be used with Python 3.x.x
```python
pip install git+https://github.com/andrey-rrousan/nuwavdet
```

## Main use

To use the package in your project, import it in by writing

```python
from nuwavdet import nuwavdet as nw
```

The main functionality of the pacakge is presented with a single function
```python
process(obs_path, thresh)
```
Inputs are string with path to the _cl.evt file to use and a tuple of thresholds, e.g.
```python
process('D:\\Data\\obs_cl.evt', (3, 2))
```

Outputs of the function are:
1. dictionary with some metadata and properties of the observation after mask generation procedure.
2. region array with mask in DET1 coordinate frame. Note that this mask is for numpy mask application so 1 corresponds to masked pixel and 0 otherwise.
3. custom bad pixel table with flagged pixels in RAW coordinates. It can be exported as fits file or each separate table can be acessed directly.
4. array with the sum of wavelet planes used in the processing.

Metadata about the observation file:

1. OBS_ID
2. Detector
3. Coordinates in equatorial (ra,dec) and galactical (lon,lat) systems
4. Time of the observation in seconds
5. Exposure

Useful algorythm-related data:

6. Average count rate of unmasked area
7. Fraction of unmasked area
8. Modified Cash-statistic per bin before and after masking the detected sources

## Other uses

You can process the cl.evt file by creating an Observation class object:

```python
obs = nw.Observation(path_to_evt_file)
```

Additionally, the energy band in KeV to get events from can be passed as an argument. The default value is [3,20].

```python
obs = nuwavsource.Observation(path_to_evt_file,E_borders=[E_min,E_max])
```

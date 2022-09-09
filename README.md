# nuwavsource

This package is supposed to be used to detect the sources in NuStar observations and generate a mask excluding the signal from the sources of any kind. 

Additionaly, it generates a table containing:

Useful data about the observation:

1. OBS_ID
2. Detector
3. Coordinates in equatorial (ra,dec) and galactical (lon,lat) systems
4. Time of the observation in seconds
5. Exposure

Useful algorythm-related data:

6. Average count rate on unmasked area
7. Portion of unmasked area
8. Specific statistical metric[1] before and after masking the detected sources
9. Root-mean-square of counts in unmasked area

## Installation
This package is to be used with Python 3.x.x

To install tha package write

`pip install nuwavsource`

## Usage

To use the package in your project, import it in by writing

`from nuwavsource import nuwavsource`

You can process the cl.evt file by creating an Observation class object:

`obs = nuwavsource.Observation(path_to_evt_file)`

Additionally, the energy band to get events from can be passed as an argument:

`obs = nuwavsource.Observation(path_to_evt_file,E_borders=[E_min,E_max])`

The default value used in the code is [3,20]

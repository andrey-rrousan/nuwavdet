# nuwavsource

This package is supposed to be used to detect the sources in NuStar observations and generate a mask excluding the signal from the sources of any kind. 

Additionaly, it generates a table containing:
Useful data about the observation:
1. OBS_ID
2. Detector
3. coordinates in equatorial (ra,dec) and galactical (lon,lat) systems
4. time of the observation in seconds
5. exposure

Useful algorythm-related data:
6. average count rate on unmasked area
7. portion of unmasked area
8. specific statistical metric[1] before and after masking the detected sources
9. root-mean-square of counts in unmasked area

## Installation
This package is to be used with Python 3.x.x

To install tha package write

`pip install nuwavsource`

## Usage

To use the package in your project, import it in by writing

`from nuwavsource import nuwavsource`

The list of useful tools in this package is

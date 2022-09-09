# nuwavsource

This package is supposed to be used to detect the sources in NuStar observations and generate a mask excluding the signal from the sources of any kind. Additionaly, it generates a table containing useful data about the observation(OBS_ID, Detector, coordinates in equatorial (ra,dec) and galactical (lon,lat) systems, time of the observation in seconds, exposure) and useful algorythm-related data (average count rate on unmasked area, portion of unmasked area, specific statistical metric[1] before and after masking the detected sources and root-mean-square of counts in unmasked area)

## Installation
This package is to be used with Python 3.x.x

To install tha package write

`pip install nuwavsource`

## Usage

To use the package in your project, import it in by writing

`from nuwavsource import nuwavsource`

The list of useful tools in this package is

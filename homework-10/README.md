#CalCalc

[![Build Status](https://travis-ci.org/sam-dixon/ay250hw10.svg?branch=master)](https://travis-ci.org/sam-dixon/ay250hw10)

An interface for Wolfram Alpha.

## Installation

Run `python setup.py install`

## Usage

CalCalc can be used from the command line or as a python module.

The command line usage is

```
python calcalc.py [-h] [-s query] [-f]

-h, --help  Print help message and exit
-s query    Send a query string to WA
-f          Return the answer as a float (i.e. with no units)
```

e.g.

```
$ python calcalc.py -s 'weight of the sun * 30'
5.965305×10^31 kg  (kilograms)
$ python calcalc.py -s 'weight of the sun * 30' -f
5.965305e+31
```

To use CalCalc in the python/IPython shell:

```
>>> from calcalc import calculate
>>> calculate('weight of the sun * 30')
'5.965305×10^31 kg  (kilograms)'
>>> calculate('weight of the sun * 30', return_float=True)
5.965305e+31
```
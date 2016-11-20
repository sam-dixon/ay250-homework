#CalCalc

An interface for Wolfram Alpha.

## Installation

Run `python setup.py install`

## Usage

CalCalc can be used from the command line or as a python module.

The command line usage is

```
python CalCalc.py [-h] [-s query] [-f]

-h, --help  Print help message and exit
-s query    Send a query string to WA
-f          Return the answer as a float (i.e. with no units)
```

e.g.

```
$ python CalCalc.py -s 'weight of the sun * 30'
5.965305×10^31 kg  (kilograms)
$ python CalCalc.py -s 'weight of the sun * 30' -f
```

To use CalCalc in the python/IPython shell:

```
>>> from calcalc import calculate
>>> calculate('weight of the sun * 30')
'5.965305×10^31 kg  (kilograms)'
>>> calculate('weight of the sun * 30', return_float=True)
```
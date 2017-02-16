# Homework 2

# Problem 0
![Phillips relation](https://github.com/sam-dixon/ay250-homework/blob/master/hw_2/hw_2_data/phillips_relation.png?raw=true)

This figure from [Phillips 1993](http://adsabs.harvard.edu/abs/1993ApJ...413L.105P) shows a correlation between the decline rate and peak luminosity of 9 Type Ia supernovae. Each subplot shows the peak magnitude in a different band. The caption does a good job of explaining what the plot shows, especially by clarifying the definition of $\Delta m_{15}$. By including the best-fit lines, the reader can get a rough idea of the strength of the claimed correlation.

There are some extraneous marks in the plot -- the right axis ticks don't need to be there. The text explains that the two data points joined by a line are so joined to indicate that the reddening of one was derived by assuming its light curve and colors was similar to those of the other. This point could also be included in the caption. It may also be useful to point out that not all 9 supernovae have I-band measurements.


# Problem 1
I recreated the Union 2.1 Compilation Type Ia supernova Hubble diagram (Figure 4 in [Suzuki et al 2012](http://arxiv.org/pdf/1105.3470v1.pdf)). The Bokeh app can be run with

`bokeh serve --show bokeh_hubble.py`

The checkboxes on the right allow the user to select which sub-sample to plot. The Hubble diagram (distance modulus as a function of redshift) is plotted for the selected data sets, and below that residuals from the best-fit cosmology are shown. When no boxes are checked, all of the data is plotted.

# Problem 2
The recreated figure is `problem_2.png` and the code used to generate it is saved as `problem_2.py`

# Problem 3
The brushing code is in the Jupyter notebook `brushing.ipynb`
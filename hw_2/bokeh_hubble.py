import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.layouts import widgetbox, layout, row
from bokeh.models.widgets import CheckboxGroup
from bokeh.models import ColumnDataSource
from astropy.cosmology import FlatLambdaCDM

hubble_data = np.genfromtxt('hw_2_data/Union2.1RedshiftMag.txt', usecols=(0,1,2,3),
                            names='name, redshift, distance_mod, distance_mod_err',
                            dtype=['U32', 'f8', 'f8', 'f8'])
hubble_data = pd.DataFrame(hubble_data)
sample_info = np.genfromtxt('hw_2_data/SCPUnion2.1_AllSNe.txt', delimiter=' & ', usecols=(0,7),
                            names='name, sample_num', dtype=['U32','int'])
sample_info = pd.DataFrame(sample_info)
sample_dict = {1: 'Hamuy et al. (1996)',
               2: 'Krisciunases et al. (2005)',
               3: 'Riess et al. (1999)',
               4: 'Jha et al. (2006)',
               5: 'Kowalski et al. (2008)',
               6: 'Hicken et al. (2009)', 
               7: 'Contreras et al. (2010)',
               8: 'Holtzman et al. (2009)',
               9: 'Riess et al. (1998)',
               10: 'Perlmutter et al. (1999)',
               11: 'Barris et al. (2004)',
               12: 'Amanullah et al. (2008)',
               13: 'Knop et al. (2003)',
               14: 'Astier et al. (2006)',
               15: 'Miknaitis et al. (2007)',
               16: 'Tonry et al. (2003)',
               17: 'Riess et al. (2007)',
               18: 'Amanullah et al. (2010)',
               19: 'Suzuki et al. (2012)'}
sample_info['sample'] = pd.Series([sample_dict[i] for i in sample_info['sample_num']])

merged_data = pd.merge(hubble_data, sample_info)

colors = ['#a09336',
          '#b253bf',
          '#60b544',
          '#6667c9',
          '#b6ba38',
          '#d54b94',
          '#51bc7c',
          '#cf4156',
          '#4bb8aa',
          '#cb4626',
          '#5d92cd',
          '#db9a47',
          '#bc88d1',
          '#477a3c',
          '#9b496e',
          '#9fb36a',
          '#de8294',
          '#846d30',
          '#e57d4c',
          '#a75c38']
merged_data['color'] = pd.Series([colors[n] for n in merged_data['sample_num']])

y_err_x, y_err_y = [], []
for px, py, err in zip(merged_data['redshift'], merged_data['distance_mod'], merged_data['distance_mod_err']):
    y_err_x.append((px, px))
    y_err_y.append((py - err, py + err))

merged_data['y_err_x'] = pd.Series(y_err_x)
merged_data['y_err_y'] = pd.Series(y_err_y)

cosmo = FlatLambdaCDM(Om0=1-0.729, H0=70)
z = np.linspace(min(merged_data['redshift']), max(merged_data['redshift']), 1000)
distmods = cosmo.distmod(z=z).value

def sample_selection(attr, old, new):
    samples = [s+1 for s in new]
    selected_data = merged_data.loc[merged_data['sample_num'].isin(samples)]
    source.data = source.from_df(selected_data)

TOOLS = 'wheel_zoom,box_zoom,reset'

p = figure(title='Union 2.1 Compilation Hubble Diagram', tools=TOOLS)
p.line(z, distmods, color='black', line_width=2, alpha=0.5)
source = ColumnDataSource(merged_data)
p.circle('redshift', 'distance_mod', source=source, color='color')
p.multi_line('y_err_x', 'y_err_y', source=source, color='color', alpha=0.8)
selection = CheckboxGroup(active=[], labels=list(merged_data['sample'].unique()))
selection.on_change('active', sample_selection)
r = row(p, widgetbox(selection))
curdoc().add_root(r)

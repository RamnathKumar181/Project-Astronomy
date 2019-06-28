import yt
import numpy as np
import yt.units as units
import pylab
import matplotlib
import os
import yt
from yt.units import kpc
#fname =  'Run0/snapshot_000'

unit_base = {'UnitLength_in_cm'         : 3.08568e+21,
             'UnitMass_in_g'            :   1.989e+43,
             'UnitVelocity_in_cm_per_s' :      100000}

bbox_lim = 0.5e5 #kpc

bbox = [[0,bbox_lim],
        [0,bbox_lim],
        [0,bbox_lim]]

path = 'Outputs1/snapshots/'
for filename in os.listdir(path):
    file = os.path.join(path,filename)
    ds = yt.load(file,unit_base=unit_base)
    if ds is not None:

        ad = ds.all_data()
        p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y',
                            'particle_mass', width=(0.5e5, 0.5e5))
        p.set_unit('particle_mass', 'Msun')

        p.annotate_title('Zoomed-in Particle Plot')
        p.save()

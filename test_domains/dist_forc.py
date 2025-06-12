## dist forcing files

import sys
import os
from datetime import datetime
from parflow.tools import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path, exists
from parflow.tools.settings import set_working_directory

#-----------------------------------------------------------------------------------------
# User-defined local variables
#-----------------------------------------------------------------------------------------
forcing_location = os.path.join(sys.argv[1])
forcing_folder = sys.argv[2]

#forcing_location = os.path.join("/home/lc2465/NAIRR/test_domains/Upper_Eel/forcings/")
#forcing_folder = "WY2003"

base= static_write_dir = os.path.join(forcing_location, forcing_folder)
print("Distributing forcings from:", base)
runname = 'spinup.wy2003'

vars = ['APCP', 'DLWR', 'DSWR', 'Temp', 'SPFH', 'UGRD', 'VGRD', 'Press']

dt = 24

CONUS2 = Run(runname, __file__)
CONUS2.FileVersion = 4
CONUS2.Process.Topology.P = 2
CONUS2.Process.Topology.Q = 2
CONUS2.Process.Topology.R = 1

for var in vars:
    for i in range(1,8760,dt):
    #for i in range(1,24,dt):
        data = '/CW3E.'+var+"."+f"{i:06d}"+'_to_'+f"{(i+dt-1):06d}"+'.pfb'
        #print(data)
        CONUS2.dist(base+data)


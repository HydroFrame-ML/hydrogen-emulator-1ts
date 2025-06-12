# Transient simulation using WY2025 forcing
# set up to run one single day for analysis of predictor

import sys
import os
import shutil
from datetime import datetime
from parflow.tools import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path, exists
from parflow.tools.settings import set_working_directory

#-----------------------------------------------------------------------------------------
# User-defined local variables
#-----------------------------------------------------------------------------------------
# p1 = running with predictor but epsilon surface
# p2 = running without predictor 
# p3 = running with predictor with surface prediction

#user inputs
#runname = 'UpperEel.wy2003'
#set_working_directory(get_absolute_path('/home/lc2465/NAIRR/test_domains/Upper_Eel/transient_runs/2003'))
#forcing_path           = '/home/lc2465/NAIRR/test_domains/Upper_Eel/forcings/WY2003'

runname = sys.argv[1]
run_path = os.path.join(sys.argv[2])
forcing_path = sys.argv[3]

set_working_directory(get_absolute_path(run_path))

# ParFlow Inputs
domain_file            = '../../static_inputs/solidfile.pfsol'  
mannings_file          = '../../static_inputs/mannings.pfb'
subsurface_file        = '../../static_inputs/pf_indicator.pfb'
slope_x_file           = '../../static_inputs/slope_x.pfb'
slope_y_file           = '../../static_inputs/slope_y.pfb'
flow_barrier_file      = '../../static_inputs/pf_flowbarrier.pfb'
initial_file           = '../../static_inputs/ss_pressure_head.pfb'
initial_file           = '../../static_inputs/press_in.pfb'


#-----------------------------------------------------------------------------------------
# Create ParFlow run object 'model'
#-----------------------------------------------------------------------------------------

CONUS2 = Run(runname, __file__)
CONUS2.FileVersion = 4

## for new water year the restart is different than for a checkpoint / midpoint restart
## 
#for the start of the water year only set istep to zero, after that allow it to be read in
#fro the clm_restart.tcl counter

#istep                  = 0 
## read in CLM_istep and use that for restart time
#path_to_tcl = "clm_restart.tcl"
#lines = open(path_to_tcl, "r").readlines()[0]
#istep = [int(i) for i in lines.split() if i.isdigit()][0]

istep = 0 
clmstep                = istep + 1 

CONUS2.TimingInfo.BaseUnit      = 1.0  
CONUS2.TimingInfo.StartCount    = istep
CONUS2.TimingInfo.StartTime     = float(istep)  
CONUS2.TimingInfo.StopTime      = 24.0  #1 day test run 
#CONUS2.TimingInfo.StopTime      = 8760.0  #1 day test run 

CONUS2.TimingInfo.DumpInterval  = 1 #(-1 dumps output at every partial timestep...)
CONUS2.TimeStep.Type            = 'Constant'
CONUS2.TimeStep.Value           = 1.0 

#-----------------------------------------------------------------------------
# Set Processor topology
#-----------------------------------------------------------------------------

CONUS2.Process.Topology.P = 2
CONUS2.Process.Topology.Q = 2
CONUS2.Process.Topology.R = 1

nproc = CONUS2.Process.Topology.P * CONUS2.Process.Topology.Q * CONUS2.Process.Topology.R

##  Restart process below
#copy over RST file
# for the first timestep this gets copied manually from the prior year's directory
# as 00000, then gets copied in this directory as 00001
#for ii in range(nproc):
#    path24= '/scratch/gpfs/REEDMM/reed/CONUS2/run_outputs/WY2024/'
#    rst_to = 'clm.rst.'+f'{istep:05d}'+'.'+f'{(ii):d}'
#    rst_from = path24+'clm.rst.00000.'+f'{(ii):d}'
#    shutil.copy(rst_from,rst_to)

## manually copy initial pressure file first time 
## copy over IC file
##  
#IC_from1 = 'conus21.wy2025.out.press.'+ f'{istep:05d}'+'.pfb'
#IC_from2 = 'conus21.wy2025.out.press.'+ f'{istep:05d}'+'.pfb.dist'
#IC_to1 = 'in.press.pfb'
#IC_to2 = 'in.press.pfb.dist'
#shutil.copy(IC_from1,IC_to1)
#shutil.copy(IC_from2,IC_to2)

# copy log and timing files
#log1 = 'conus21.wy2025.out.log'
#log1_c = 'conus21.wy2025.'+ f'{istep:05d}'+'.out.log'
#log2 = 'conus21.wy2025.out.txt'
#log2_c = 'conus21.wy2025.'+f'{istep:05d}'+'.out.txt'
#log3 = 'conus21.wy2025.out.kinsol.log'
#log3_c = 'conus21.wy2025.'+ f'{istep:05d}' +'.kinsol.log' 
#log4 = 'conus21.wy2025.out.timing.csv'
#log4_c = 'conus21.wy2025.'+ f'{istep:05d}' +'.out.timing.csv'
#log5 = 'CLM.out.clm.log'
#log5_c = 'CLM.out.'+ f'{istep:05d}' +'.clm.log'

#shutil.copy(log1,log1_c)
#shutil.copy(log2,log2_c)
#shutil.copy(log3,log3_c)
# timing file only written if run is complete
#if os.path.exists(log4):
#    shutil.copy(log4,log4_c)
#shutil.copy(log5,log5_c)

# copy database input file
#inputd = 'conus21.wy2025.pfidb'
#inputd_c = 'conus21.wy2025.'+ f'{istep:05d}'+'.pfidb'
#shutil.copy(inputd,inputd_c)

# first timestep need IC from CLM but not timing
# copy CLM restart dat file
#shutil.copy('drv_clmin.restart.dat','drv_clmin.dat')

# create log file with run metadata
# desire to put more things here later 
logmeta = 'run.metalog.'+f'{istep:05d}'+'.txt'
with open(logmeta, 'w') as f:
    print("Run Restart Metadata", file=f)
    print("PF istep ",istep, file=f)
    print("CLM istep ",clmstep, file=f)

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------

CONUS2.ComputationalGrid.Lower.X = 0.0
CONUS2.ComputationalGrid.Lower.Y = 0.0
CONUS2.ComputationalGrid.Lower.Z = 0.0

CONUS2.ComputationalGrid.DX = 1000.0
CONUS2.ComputationalGrid.DY = 1000.0
CONUS2.ComputationalGrid.DZ = 200.0

CONUS2.ComputationalGrid.NX = 64
CONUS2.ComputationalGrid.NY = 62
CONUS2.ComputationalGrid.NZ = 10

#-----------------------------------------------------------------------------
# Names of the GeomInputs
#-----------------------------------------------------------------------------

CONUS2.GeomInput.Names = "domaininput indi_input"

#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------

CONUS2.GeomInput.domaininput.InputType  = 'SolidFile'
CONUS2.GeomInput.domaininput.GeomNames  = 'domain'
CONUS2.GeomInput.domaininput.FileName   = domain_file

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------

CONUS2.Geom.domain.Patches = "ocean land top lake sink bottom"
CONUS2.Geom.domain.Patches = "top bottom land"

#-----------------------------------------------------------------------------
# Indicator Geometry Input
#-----------------------------------------------------------------------------

CONUS2.GeomInput.indi_input.InputType   = 'IndicatorField'
CONUS2.GeomInput.indi_input.GeomNames   = 's1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 b1 b2'
CONUS2.Geom.indi_input.FileName         = subsurface_file 
CONUS2.dist(subsurface_file)

CONUS2.GeomInput.s1.Value = 1
CONUS2.GeomInput.s2.Value = 2
CONUS2.GeomInput.s3.Value = 3
CONUS2.GeomInput.s4.Value = 4
CONUS2.GeomInput.s5.Value = 5
CONUS2.GeomInput.s6.Value = 6
CONUS2.GeomInput.s7.Value = 7
CONUS2.GeomInput.s8.Value = 8
CONUS2.GeomInput.s9.Value = 9
CONUS2.GeomInput.s10.Value = 10
CONUS2.GeomInput.s11.Value = 11
CONUS2.GeomInput.s12.Value = 12

CONUS2.GeomInput.s13.Value = 13

CONUS2.GeomInput.b1.Value = 19
CONUS2.GeomInput.b2.Value = 20

CONUS2.GeomInput.g1.Value = 21
CONUS2.GeomInput.g2.Value = 22
CONUS2.GeomInput.g3.Value = 23
CONUS2.GeomInput.g4.Value = 24
CONUS2.GeomInput.g5.Value = 25
CONUS2.GeomInput.g6.Value = 26
CONUS2.GeomInput.g7.Value = 27
CONUS2.GeomInput.g8.Value = 28

#--------------------------------------------
# variable dz assignments
#------------------------------------------
CONUS2.Solver.Nonlinear.VariableDz = True
CONUS2.dzScale.GeomNames = 'domain'
CONUS2.dzScale.Type = 'nzList'
CONUS2.dzScale.nzListNumber = 10

# 10 layers, starts at 0 for the bottom to 9 at the top
# note this is opposite Noah/WRF
# layers are 0.1 m, 0.3 m, 0.6 m, 1.0 m, 5.0 m, 10.0 m, 25.0 m, 50.0 m, 100.0m, 200.0 m
# 200 m * 1.0 = 200 m
CONUS2.Cell._0.dzScale.Value = 1.0
# 200 m * .5 = 100 m 
CONUS2.Cell._1.dzScale.Value = 0.5
# 200 m * .25 = 50 m 
CONUS2.Cell._2.dzScale.Value = 0.25
# 200 m * 0.125 = 25 m 
CONUS2.Cell._3.dzScale.Value = 0.125
# 200 m * 0.05 = 10 m 
CONUS2.Cell._4.dzScale.Value = 0.05
# 200 m * .025 = 5 m 
CONUS2.Cell._5.dzScale.Value = 0.025
# 200 m * .005 = 1 m 
CONUS2.Cell._6.dzScale.Value = 0.005
# 200 m * 0.003 = 0.6 m 
CONUS2.Cell._7.dzScale.Value = 0.003
# 200 m * 0.0015 = 0.3 m 
CONUS2.Cell._8.dzScale.Value = 0.0015
# 200 m * 0.0005 = 0.1 m = 10 cm which is default top Noah layer
CONUS2.Cell._9.dzScale.Value = 0.0005

#------------------------------------------------------------------------------
# Flow Barrier defined by Shangguan Depth to Bedrock
#--------------------------------------------------------------

CONUS2.Solver.Nonlinear.FlowBarrierZ = True
CONUS2.FBz.Type = 'PFBFile'
CONUS2.Geom.domain.FBz.FileName = flow_barrier_file
CONUS2.dist(flow_barrier_file)

#-----------------------------------------------------------------------------
# Permeability (values in m/hr)
#-----------------------------------------------------------------------------

CONUS2.Geom.Perm.Names = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 b1 b2'

CONUS2.Geom.domain.Perm.Type = 'Constant'
CONUS2.Geom.domain.Perm.Value = 0.02

CONUS2.Geom.s1.Perm.Type = 'Constant'
CONUS2.Geom.s1.Perm.Value = 0.269022595

CONUS2.Geom.s2.Perm.Type = 'Constant'
CONUS2.Geom.s2.Perm.Value = 0.043630356

CONUS2.Geom.s3.Perm.Type = 'Constant'
CONUS2.Geom.s3.Perm.Value = 0.015841225

CONUS2.Geom.s4.Perm.Type = 'Constant'
CONUS2.Geom.s4.Perm.Value = 0.007582087

CONUS2.Geom.s5.Perm.Type = 'Constant'
CONUS2.Geom.s5.Perm.Value = 0.01818816

CONUS2.Geom.s6.Perm.Type = 'Constant'
CONUS2.Geom.s6.Perm.Value = 0.005009435

CONUS2.Geom.s7.Perm.Type = 'Constant'
CONUS2.Geom.s7.Perm.Value = 0.005492736

CONUS2.Geom.s8.Perm.Type = 'Constant'
CONUS2.Geom.s8.Perm.Value = 0.004675077

CONUS2.Geom.s9.Perm.Type = 'Constant'
CONUS2.Geom.s9.Perm.Value = 0.003386794

CONUS2.Geom.s10.Perm.Type = 'Constant'
CONUS2.Geom.s10.Perm.Value = 0.004783973

CONUS2.Geom.s11.Perm.Type = 'Constant'
CONUS2.Geom.s11.Perm.Value = 0.003979136

CONUS2.Geom.s12.Perm.Type = 'Constant'
CONUS2.Geom.s12.Perm.Value = 0.006162952

CONUS2.Geom.s13.Perm.Type = 'Constant'
CONUS2.Geom.s13.Perm.Value = 0.005009435

CONUS2.Geom.b1.Perm.Type = 'Constant'
CONUS2.Geom.b1.Perm.Value = 0.005

CONUS2.Geom.b2.Perm.Type = 'Constant'
CONUS2.Geom.b2.Perm.Value = 0.01

CONUS2.Geom.g1.Perm.Type = 'Constant'
CONUS2.Geom.g1.Perm.Value = 0.02

CONUS2.Geom.g2.Perm.Type = 'Constant'
CONUS2.Geom.g2.Perm.Value = 0.03

CONUS2.Geom.g3.Perm.Type = 'Constant'
CONUS2.Geom.g3.Perm.Value = 0.04

CONUS2.Geom.g4.Perm.Type = 'Constant'
CONUS2.Geom.g4.Perm.Value = 0.05

CONUS2.Geom.g5.Perm.Type = 'Constant'
CONUS2.Geom.g5.Perm.Value = 0.06

CONUS2.Geom.g6.Perm.Type = 'Constant'
CONUS2.Geom.g6.Perm.Value = 0.08

CONUS2.Geom.g7.Perm.Type = 'Constant'
CONUS2.Geom.g7.Perm.Value = 0.1

CONUS2.Geom.g8.Perm.Type = 'Constant'
CONUS2.Geom.g8.Perm.Value = 0.2

CONUS2.Perm.TensorType = 'TensorByGeom'
CONUS2.Geom.Perm.TensorByGeom.Names = 'domain b1 b2 g1 g2 g4 g5 g6 g7'

CONUS2.Geom.domain.Perm.TensorValX = 1.0
CONUS2.Geom.domain.Perm.TensorValY = 1.0
CONUS2.Geom.domain.Perm.TensorValZ = 1.0

CONUS2.Geom.b1.Perm.TensorValX = 1.0
CONUS2.Geom.b1.Perm.TensorValY = 1.0
CONUS2.Geom.b1.Perm.TensorValZ = 0.1

CONUS2.Geom.b2.Perm.TensorValX = 1.0
CONUS2.Geom.b2.Perm.TensorValY = 1.0
CONUS2.Geom.b2.Perm.TensorValZ = 0.1

CONUS2.Geom.g1.Perm.TensorValX = 1.0
CONUS2.Geom.g1.Perm.TensorValY = 1.0
CONUS2.Geom.g1.Perm.TensorValZ = 0.1

CONUS2.Geom.g2.Perm.TensorValX = 1.0
CONUS2.Geom.g2.Perm.TensorValY = 1.0
CONUS2.Geom.g2.Perm.TensorValZ = 0.1

CONUS2.Geom.g4.Perm.TensorValX = 1.0
CONUS2.Geom.g4.Perm.TensorValY = 1.0
CONUS2.Geom.g4.Perm.TensorValZ = 0.1

CONUS2.Geom.g5.Perm.TensorValX = 1.0
CONUS2.Geom.g5.Perm.TensorValY = 1.0
CONUS2.Geom.g5.Perm.TensorValZ = 0.1

CONUS2.Geom.g6.Perm.TensorValX = 1.0
CONUS2.Geom.g6.Perm.TensorValY = 1.0
CONUS2.Geom.g6.Perm.TensorValZ = 0.1

CONUS2.Geom.g7.Perm.TensorValX = 1.0
CONUS2.Geom.g7.Perm.TensorValY = 1.0
CONUS2.Geom.g7.Perm.TensorValZ = 0.1

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

CONUS2.SpecificStorage.Type                 = 'Constant'
CONUS2.SpecificStorage.GeomNames            = 'domain'
CONUS2.Geom.domain.SpecificStorage.Value    = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

CONUS2.Phase.Names                  = 'water'
CONUS2.Phase.water.Density.Type     = 'Constant'
CONUS2.Phase.water.Density.Value    = 1.0
CONUS2.Phase.water.Viscosity.Type   = 'Constant'
CONUS2.Phase.water.Viscosity.Value  = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

CONUS2.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

CONUS2.Gravity = 1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

CONUS2.Geom.Porosity.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 b1 b2'

CONUS2.Geom.domain.Porosity.Type = 'Constant'
CONUS2.Geom.domain.Porosity.Value = 0.33

CONUS2.Geom.s1.Porosity.Type = 'Constant'
CONUS2.Geom.s1.Porosity.Value = 0.375

CONUS2.Geom.s2.Porosity.Type = 'Constant'
CONUS2.Geom.s2.Porosity.Value = 0.39

CONUS2.Geom.s3.Porosity.Type = 'Constant'
CONUS2.Geom.s3.Porosity.Value = 0.387

CONUS2.Geom.s4.Porosity.Type = 'Constant'
CONUS2.Geom.s4.Porosity.Value = 0.439

CONUS2.Geom.s5.Porosity.Type = 'Constant'
CONUS2.Geom.s5.Porosity.Value = 0.489

CONUS2.Geom.s6.Porosity.Type = 'Constant'
CONUS2.Geom.s6.Porosity.Value = 0.399

CONUS2.Geom.s7.Porosity.Type = 'Constant'
CONUS2.Geom.s7.Porosity.Value = 0.384

CONUS2.Geom.s8.Porosity.Type = 'Constant'
CONUS2.Geom.s8.Porosity.Value = 0.482

CONUS2.Geom.s9.Porosity.Type = 'Constant'
CONUS2.Geom.s9.Porosity.Value = 0.442

CONUS2.Geom.s10.Porosity.Type = 'Constant'
CONUS2.Geom.s10.Porosity.Value = 0.385

CONUS2.Geom.s11.Porosity.Type = 'Constant'
CONUS2.Geom.s11.Porosity.Value = 0.481

CONUS2.Geom.s12.Porosity.Type = 'Constant'
CONUS2.Geom.s12.Porosity.Value = 0.459

CONUS2.Geom.s13.Porosity.Type = 'Constant'
CONUS2.Geom.s13.Porosity.Value = 0.399

CONUS2.Geom.b1.Porosity.Type = 'Constant'
CONUS2.Geom.b1.Porosity.Value = 0.05

CONUS2.Geom.b2.Porosity.Type = 'Constant'
CONUS2.Geom.b2.Porosity.Value = 0.1

CONUS2.Geom.g1.Porosity.Type = 'Constant'
#CONUS2.Geom.g1.Porosity.Value = 0.33  # changed see google sheet
CONUS2.Geom.g1.Porosity.Value = 0.12

CONUS2.Geom.g2.Porosity.Type = 'Constant'
#CONUS2.Geom.g2.Porosity.Value = 0.33  # changed 
CONUS2.Geom.g2.Porosity.Value = 0.3

CONUS2.Geom.g3.Porosity.Type = 'Constant'
#CONUS2.Geom.g3.Porosity.Value = 0.33  # changed
CONUS2.Geom.g3.Porosity.Value = 0.01

CONUS2.Geom.g4.Porosity.Type = 'Constant'
#CONUS2.Geom.g4.Porosity.Value = 0.33  # changed
CONUS2.Geom.g4.Porosity.Value = 0.15

CONUS2.Geom.g5.Porosity.Type = 'Constant'
#CONUS2.Geom.g5.Porosity.Value = 0.33  # changed
CONUS2.Geom.g5.Porosity.Value = 0.22

CONUS2.Geom.g6.Porosity.Type = 'Constant'
#CONUS2.Geom.g6.Porosity.Value = 0.33  # changed
CONUS2.Geom.g6.Porosity.Value = 0.27

CONUS2.Geom.g7.Porosity.Type = 'Constant'
CONUS2.Geom.g7.Porosity.Value = 0.06

CONUS2.Geom.g8.Porosity.Type = 'Constant'
CONUS2.Geom.g8.Porosity.Value = 0.3

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

CONUS2.Domain.GeomName = 'domain'

#----------------------------------------------------------------------------
# Mobility
#----------------------------------------------------------------------------

CONUS2.Phase.water.Mobility.Type = 'Constant'
CONUS2.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

CONUS2.Wells.Names = ''

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

CONUS2.Phase.RelPerm.Type = 'VanGenuchten'
CONUS2.Phase.RelPerm.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13'

CONUS2.Geom.domain.RelPerm.Alpha                = 0.5
CONUS2.Geom.domain.RelPerm.N                    = 2.5
CONUS2.Geom.domain.RelPerm.NumSamplePoints      = 20000
CONUS2.Geom.domain.RelPerm.MinPressureHead      = -500
CONUS2.Geom.domain.RelPerm.InterpolationMethod  = 'Linear'

CONUS2.Geom.s1.RelPerm.Alpha                    = 3.548
CONUS2.Geom.s1.RelPerm.N                        = 4.162
CONUS2.Geom.s1.RelPerm.NumSamplePoints          = 20000
CONUS2.Geom.s1.RelPerm.MinPressureHead          = -300
CONUS2.Geom.s1.RelPerm.InterpolationMethod      = 'Linear'

CONUS2.Geom.s2.RelPerm.Alpha                    = 3.467
CONUS2.Geom.s2.RelPerm.N                        = 2.738
CONUS2.Geom.s2.RelPerm.NumSamplePoints          = 20000
CONUS2.Geom.s2.RelPerm.MinPressureHead          = -300
CONUS2.Geom.s2.RelPerm.InterpolationMethod      = 'Linear'

CONUS2.Geom.s3.RelPerm.Alpha                    = 2.692
CONUS2.Geom.s3.RelPerm.N                        = 2.445
CONUS2.Geom.s3.RelPerm.NumSamplePoints          = 20000
CONUS2.Geom.s3.RelPerm.MinPressureHead          = -300
CONUS2.Geom.s3.RelPerm.InterpolationMethod      = 'Linear'

CONUS2.Geom.s4.RelPerm.Alpha                    = 0.501
CONUS2.Geom.s4.RelPerm.N                        = 2.659
CONUS2.Geom.s4.RelPerm.NumSamplePoints          = 20000
CONUS2.Geom.s4.RelPerm.MinPressureHead          = -300
CONUS2.Geom.s4.RelPerm.InterpolationMethod      = 'Linear'

CONUS2.Geom.s5.RelPerm.Alpha                    = 0.661
CONUS2.Geom.s5.RelPerm.N                        = 2.659
CONUS2.Geom.s5.RelPerm.NumSamplePoints          = 20000
CONUS2.Geom.s5.RelPerm.MinPressureHead          = -300
CONUS2.Geom.s5.RelPerm.InterpolationMethod      = 'Linear'

CONUS2.Geom.s6.RelPerm.Alpha                    = 1.122
CONUS2.Geom.s6.RelPerm.N                        = 2.479
CONUS2.Geom.s6.RelPerm.NumSamplePoints          = 20000
CONUS2.Geom.s6.RelPerm.MinPressureHead          = -300
CONUS2.Geom.s6.RelPerm.InterpolationMethod      = 'Linear'

CONUS2.Geom.s7.RelPerm.Alpha                    = 2.089
CONUS2.Geom.s7.RelPerm.N                        = 2.318
CONUS2.Geom.s7.RelPerm.NumSamplePoints          = 20000
CONUS2.Geom.s7.RelPerm.MinPressureHead          = -300
CONUS2.Geom.s7.RelPerm.InterpolationMethod      = 'Linear'

CONUS2.Geom.s8.RelPerm.Alpha                    = 0.832
CONUS2.Geom.s8.RelPerm.N                        = 2.514
CONUS2.Geom.s8.RelPerm.NumSamplePoints          = 20000
CONUS2.Geom.s8.RelPerm.MinPressureHead          = -300
CONUS2.Geom.s8.RelPerm.InterpolationMethod      = 'Linear'

CONUS2.Geom.s9.RelPerm.Alpha                    = 1.585
CONUS2.Geom.s9.RelPerm.N                        = 2.413
CONUS2.Geom.s9.RelPerm.NumSamplePoints          = 20000
CONUS2.Geom.s9.RelPerm.MinPressureHead          = -300
CONUS2.Geom.s9.RelPerm.InterpolationMethod      = 'Linear'

CONUS2.Geom.s10.RelPerm.Alpha                   = 3.311
CONUS2.Geom.s10.RelPerm.N                       = 2.202
CONUS2.Geom.s10.RelPerm.NumSamplePoints         = 20000
CONUS2.Geom.s10.RelPerm.MinPressureHead         = -300
CONUS2.Geom.s10.RelPerm.InterpolationMethod     = 'Linear'

CONUS2.Geom.s11.RelPerm.Alpha                   = 1.622
CONUS2.Geom.s11.RelPerm.N                       = 2.318
CONUS2.Geom.s11.RelPerm.NumSamplePoints         = 20000
CONUS2.Geom.s11.RelPerm.MinPressureHead         = -300
CONUS2.Geom.s11.RelPerm.InterpolationMethod     = 'Linear'

CONUS2.Geom.s12.RelPerm.Alpha                   = 1.514
CONUS2.Geom.s12.RelPerm.N                       = 2.259
CONUS2.Geom.s12.RelPerm.NumSamplePoints         = 20000
CONUS2.Geom.s12.RelPerm.MinPressureHead         = -300
CONUS2.Geom.s12.RelPerm.InterpolationMethod     = 'Linear'

CONUS2.Geom.s13.RelPerm.Alpha                   = 1.122
CONUS2.Geom.s13.RelPerm.N                       = 2.479
CONUS2.Geom.s13.RelPerm.NumSamplePoints         = 20000
CONUS2.Geom.s13.RelPerm.MinPressureHead         = -300
CONUS2.Geom.s13.RelPerm.InterpolationMethod     = 'Linear'

#-----------------------------------------------------------------------------
# Saturation
#-----------------------------------------------------------------------------

CONUS2.Phase.Saturation.Type = 'VanGenuchten'
CONUS2.Phase.Saturation.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13'

CONUS2.Geom.domain.Saturation.Alpha = 0.5
CONUS2.Geom.domain.Saturation.N = 2.5
CONUS2.Geom.domain.Saturation.SRes = 0.0001
CONUS2.Geom.domain.Saturation.SSat = 1.0

CONUS2.Geom.s1.Saturation.Alpha = 3.548
CONUS2.Geom.s1.Saturation.N = 4.162
CONUS2.Geom.s1.Saturation.SRes = 0.0001
CONUS2.Geom.s1.Saturation.SSat = 1.0

CONUS2.Geom.s2.Saturation.Alpha = 3.467
CONUS2.Geom.s2.Saturation.N = 2.738
CONUS2.Geom.s2.Saturation.SRes = 0.0001
CONUS2.Geom.s2.Saturation.SSat = 1.0

CONUS2.Geom.s3.Saturation.Alpha = 2.692
CONUS2.Geom.s3.Saturation.N = 2.445
CONUS2.Geom.s3.Saturation.SRes = 0.0001
CONUS2.Geom.s3.Saturation.SSat = 1.0

CONUS2.Geom.s4.Saturation.Alpha = 0.501
CONUS2.Geom.s4.Saturation.N = 2.659
CONUS2.Geom.s4.Saturation.SRes = 0.0001
CONUS2.Geom.s4.Saturation.SSat = 1.0

CONUS2.Geom.s5.Saturation.Alpha = 0.661
CONUS2.Geom.s5.Saturation.N = 2.659
CONUS2.Geom.s5.Saturation.SRes = 0.0001
CONUS2.Geom.s5.Saturation.SSat = 1.0

CONUS2.Geom.s6.Saturation.Alpha = 1.122
CONUS2.Geom.s6.Saturation.N = 2.479
CONUS2.Geom.s6.Saturation.SRes = 0.0001
CONUS2.Geom.s6.Saturation.SSat = 1.0

CONUS2.Geom.s7.Saturation.Alpha = 2.089
CONUS2.Geom.s7.Saturation.N = 2.318
CONUS2.Geom.s7.Saturation.SRes = 0.0001
CONUS2.Geom.s7.Saturation.SSat = 1.0

CONUS2.Geom.s8.Saturation.Alpha = 0.832
CONUS2.Geom.s8.Saturation.N = 2.514
CONUS2.Geom.s8.Saturation.SRes = 0.0001
CONUS2.Geom.s8.Saturation.SSat = 1.0

CONUS2.Geom.s9.Saturation.Alpha = 1.585
CONUS2.Geom.s9.Saturation.N = 2.413
CONUS2.Geom.s9.Saturation.SRes = 0.0001
CONUS2.Geom.s9.Saturation.SSat = 1.0

CONUS2.Geom.s10.Saturation.Alpha = 3.311
CONUS2.Geom.s10.Saturation.N = 2.202
CONUS2.Geom.s10.Saturation.SRes = 0.0001
CONUS2.Geom.s10.Saturation.SSat = 1.0

CONUS2.Geom.s11.Saturation.Alpha = 1.622
CONUS2.Geom.s11.Saturation.N = 2.318
CONUS2.Geom.s11.Saturation.SRes = 0.0001
CONUS2.Geom.s11.Saturation.SSat = 1.0

CONUS2.Geom.s12.Saturation.Alpha = 1.514
CONUS2.Geom.s12.Saturation.N = 2.259
CONUS2.Geom.s12.Saturation.SRes = 0.0001
CONUS2.Geom.s12.Saturation.SSat = 1.0

CONUS2.Geom.s13.Saturation.Alpha = 1.122
CONUS2.Geom.s13.Saturation.N = 2.479
CONUS2.Geom.s13.Saturation.SRes = 0.0001
CONUS2.Geom.s13.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

CONUS2.Cycle.Names                      = 'constant'
CONUS2.Cycle.constant.Names             = 'alltime'
CONUS2.Cycle.constant.alltime.Length    = 1
CONUS2.Cycle.constant.Repeat            = -1

#-----------------------------------------------------------------------------
# Boundary Conditions
#-----------------------------------------------------------------------------

CONUS2.BCPressure.PatchNames = CONUS2.Geom.domain.Patches

# CONUS2.Patch.ocean.BCPressure.Type = 'FluxConst'
# CONUS2.Patch.ocean.BCPressure.Cycle = 'constant'
# CONUS2.Patch.ocean.BCPressure.RefGeom = 'domain'
# CONUS2.Patch.ocean.BCPressure.RefPatch = 'top'
# CONUS2.Patch.ocean.BCPressure.alltime.Value = 0.0

# CONUS2.Patch.sink.BCPressure.Type = 'OverlandKinematic'
# #CONUS2.Patch.sink.BCPressure.Type = 'SeepageFace'
# CONUS2.Patch.sink.BCPressure.Cycle = 'constant'
# CONUS2.Patch.sink.BCPressure.RefGeom = 'domain'
# CONUS2.Patch.sink.BCPressure.RefPatch = 'top'
# CONUS2.Patch.sink.BCPressure.alltime.Value = 0.0

# CONUS2.Patch.lake.BCPressure.Type = 'OverlandKinematic'
# #CONUS2.Patch.lake.BCPressure.Type = 'SeepageFace'
# CONUS2.Patch.lake.BCPressure.Cycle = 'constant'
# CONUS2.Patch.lake.BCPressure.RefGeom = 'domain'
# CONUS2.Patch.lake.BCPressure.RefPatch = 'top'
# CONUS2.Patch.lake.BCPressure.alltime.Value = 0.0

CONUS2.Patch.land.BCPressure.Type = 'FluxConst'
CONUS2.Patch.land.BCPressure.Cycle = 'constant'
CONUS2.Patch.land.BCPressure.alltime.Value = 0.0

CONUS2.Patch.bottom.BCPressure.Type = 'FluxConst'
CONUS2.Patch.bottom.BCPressure.Cycle = 'constant'
CONUS2.Patch.bottom.BCPressure.alltime.Value = 0.0

# CONUS2.Solver.OverlandKinematic.SeepageOne = 3  ## new key
# CONUS2.Solver.OverlandKinematic.SeepageTwo = 4 ## new key

CONUS2.Patch.top.BCPressure.Type = 'OverlandKinematic'
CONUS2.Patch.top.BCPressure.Cycle = 'constant'
CONUS2.Patch.top.BCPressure.alltime.Value = 0

#-----------------------------------------------------------------------------
# Topo slopes in x-direction
#-----------------------------------------------------------------------------

CONUS2.TopoSlopesX.Type = 'PFBFile'
CONUS2.TopoSlopesX.GeomNames = 'domain'
CONUS2.TopoSlopesX.FileName = slope_x_file
CONUS2.dist(slope_x_file)


#-----------------------------------------------------------------------------
# Topo slopes in y-direction
#-----------------------------------------------------------------------------

CONUS2.TopoSlopesY.Type = 'PFBFile'
CONUS2.TopoSlopesY.GeomNames = 'domain'
CONUS2.TopoSlopesY.FileName = slope_y_file
CONUS2.dist(slope_y_file)

#-----------------------------------------------------------------------------
# Initial conditions: water pressure
#-----------------------------------------------------------------------------

CONUS2.ICPressure.Type = 'HydroStaticPatch'

CONUS2.ICPressure.Type = 'PFBFile'
CONUS2.ICPressure.GeomNames = 'domain'
CONUS2.Geom.domain.ICPressure.RefPatch = 'bottom'
CONUS2.Geom.domain.ICPressure.FileName = initial_file
CONUS2.dist(initial_file)

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

CONUS2.PhaseSources.water.Type = 'Constant'
CONUS2.PhaseSources.water.GeomNames = 'domain'
CONUS2.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Mannings coefficient
#-----------------------------------------------------------------------------

CONUS2.Mannings.Type = 'PFBFile'
CONUS2.Mannings.FileName = mannings_file
CONUS2.dist(mannings_file)

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

CONUS2.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------------------
# Set LSM parameters
#-----------------------------------------------------------------------------------------

CONUS2.Solver.LSM                   = 'CLM'
CONUS2.Solver.CLM.Print1dOut        = False
CONUS2.Solver.CLM.CLMDumpInterval   = 1

CONUS2.Solver.CLM.MetForcing        = '3D'
CONUS2.Solver.CLM.MetFileName       = 'CW3E'
CONUS2.Solver.CLM.MetFilePath       = forcing_path 
CONUS2.Solver.CLM.MetFileNT         = 24
CONUS2.Solver.CLM.IstepStart        = clmstep

CONUS2.Solver.CLM.EvapBeta          = 'Linear'
CONUS2.Solver.CLM.VegWaterStress    = 'Saturation'
CONUS2.Solver.CLM.ResSat            = 0.2
CONUS2.Solver.CLM.WiltingPoint      = 0.2
CONUS2.Solver.CLM.FieldCapacity     = 1.00
CONUS2.Solver.CLM.IrrigationType    = 'none'
## this key sets the option described in Ferguson, Jefferson, et al ESS 2016
# a setting of 0 (default) will use standard water stress distribution
CONUS2.Solver.CLM.RZWaterStress      = 1

CONUS2.Solver.CLM.RootZoneNZ        = 5
CONUS2.Solver.CLM.SoiLayer          = 4
CONUS2.Solver.CLM.ReuseCount        = 4 #10 #4 #1
CONUS2.Solver.CLM.WriteLogs         = False
CONUS2.Solver.CLM.WriteLastRST      = True
CONUS2.Solver.CLM.DailyRST          = True
CONUS2.Solver.CLM.SingleFile        = True

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

CONUS2.Solver = 'Richards'
#CONUS2.Solver.MaxIter = 250
#CONUS2.Solver.Drop = 1E-30
#CONUS2.Solver.AbsTol = 1E-9
CONUS2.Solver.MaxConvergenceFailures = 5
CONUS2.Solver.Nonlinear.MaxIter = 250
CONUS2.Solver.Nonlinear.ResidualTol = 1e-5

CONUS2.Solver.TerrainFollowingGrid = True
CONUS2.Solver.TerrainFollowingGrid.SlopeUpwindFormulation = 'Upwind'

CONUS2.Solver.WriteCLMBinary = False
CONUS2.Solver.PrintCLM = True
CONUS2.Solver.EvapTransFile = False
#CONUS2.Solver.BinaryOutDir = False

CONUS2.Solver.PrintTop = True
CONUS2.Solver.Nonlinear.EtaChoice = 'EtaConstant'
CONUS2.Solver.Nonlinear.EtaValue = 0.01
CONUS2.Solver.Nonlinear.UseJacobian = True
#CONUS2.Solver.Nonlinear.UseJacobian = False 
#CONUS2.Solver.Nonlinear.DerivativeEpsilon = 1e-16
CONUS2.Solver.Nonlinear.StepTol = 1e-15
CONUS2.Solver.Nonlinear.Globalization = 'LineSearch'

CONUS2.Solver.Linear.KrylovDimension = 100
CONUS2.Solver.Linear.MaxRestarts = 3

CONUS2.Solver.Linear.Preconditioner = 'PFMGOctree'
#CONUS2.Solver.Linear.Preconditioner = 'PFMG'
#CONUS2.Solver.Linear.Preconditioner = 'MGSemi'

CONUS2.Solver.PrintSubsurfData = True
CONUS2.Solver.PrintMask = True
CONUS2.Solver.PrintSaturation = True
CONUS2.Solver.PrintPressure = True
CONUS2.Solver.PrintSlopes = True
CONUS2.Solver.PrintMannings = True
CONUS2.Solver.PrintEvapTrans = True
CONUS2.Solver.PrintVelocities = False


# surface flow predictor 
CONUS2.Solver.SurfacePredictor = True 
#CONUS2.Solver.SurfacePredictor = False 
CONUS2.Solver.SurfacePredictor.PrintValues = True 
CONUS2.Solver.SurfacePredictor.PrintValues = False 
CONUS2.Solver.SurfacePredictor.PressureValue = 0.00001
#CONUS2.Solver.SurfacePredictor.PressureValue =  -1.0 


CONUS2.Solver.WriteSiloSpecificStorage = False
CONUS2.Solver.WriteSiloMannings = False
CONUS2.Solver.WriteSiloMask = False
CONUS2.Solver.WriteSiloSlopes = False
CONUS2.Solver.WriteSiloSubsurfData = False
CONUS2.Solver.WriteSiloPressure = False
CONUS2.Solver.WriteSiloSaturation = False
CONUS2.Solver.WriteSiloEvapTrans = False
CONUS2.Solver.WriteSiloEvapTransSum = False
CONUS2.Solver.WriteSiloOverlandSum = False
CONUS2.Solver.WriteSiloCLM = False


CONUS2.run(skip_validation=True)


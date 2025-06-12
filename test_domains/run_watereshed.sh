#!/bin/bash
#SBATCH --job-name=UpperEel
#SBATCH --ntasks=4
#SBATCH --time=30:00

module load parflow-shared

# User inputs
run_path="/home/lc2465/NAIRR/test_domains"
repo_path="/home/lc2465/NAIRR/hydrogen-emulator-1ts"
watershed_name="Upper_Eel"
water_year=2003
dist_forcings=true
forcing_folder="WY2003"

#If the forcings need to be distributed do that first
if [ "$dist_forcings" = true ] ; then
    echo 'Distributing the forcings now'
    cd "$run_path/$watershed_name/forcings"
    cp "$repo_path/test_domains/dist_forc.py" . 
    
    python dist_forc.py "$run_path/$watershed_name/forcings/" $forcing_folder
fi

# Make a directory for the WY you are runing, change directory to it and copy in the clm driver and the python run script
mkdir -p "$run_path/$watershed_name/transient_runs/WY$water_year"
cd "$run_path/$watershed_name/transient_runs/WY$water_year"
cp ../../static_inputs/drv*.dat .
cp "$repo_path/test_domains/ParFlow_WatershedRun_Shapefile.py" .


# Run Parflow
python ParFlow_WatershedRun_Shapefile.py "$watershed_name.wy$water_year" "$run_path/$watershed_name/transient_runs/WY$water_year" "$run_path/$watershed_name/forcings/WY$water_year"



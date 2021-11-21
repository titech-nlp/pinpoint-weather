#!/bin/zsh

# Convert binary file (such as Z__C_RJTD_20140101000000_GSM_GPV_Rjp_Lsurf_FD0000-0312_grib2.bin)
# to grib2 format with `wgrib` command
# Usage:
# $./convert_bin2grib.sh [path/to/GPV_RAWDATA_DIR]

for path_to_src in `find $1 -maxdepth 2 -type f | grep .bin$`
do
  path_to_trg=$path_to_src".grib2"

  echo $path_to_src $path_to_trg
  
  wgrib2 $path_to_src -grib $path_to_trg > /dev/null
  rm $path_to_src

done 

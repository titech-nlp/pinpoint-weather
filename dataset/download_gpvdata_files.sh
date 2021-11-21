#!/bin/bash
# Usage:
# $./download_gpvdata_files.sh [path_to_download_list] [path_to_save_dir]
# $./download_gpvdata_files.sh ./gpvfile_download_url.csv ./gpvdata/

INPUT_FILE=$1
DOWNLOAD_DIR=$2

while read line
do
    date=`echo $line | cut -d, -f1`
    download_url=`echo $line | cut -d, -f2`
    # create download directory if not exists
    mkdir -p $DOWNLOAD_DIR"/"$date

    # download a gpv data 
    wget -q -P $DOWNLOAD_DIR"/"$date $download_url
    echo $download_url

done < $INPUT_FILE

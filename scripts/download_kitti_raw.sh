#!/bin/bash

# Script for fetching kitti datasets

set -e

SEQUENCES=(
"2011_10_03_drive_0027" # seq 0
"2011_10_03_drive_0042" # seq 1
"2011_10_03_drive_0034" # seq 2
"2011_09_30_drive_0016" # seq 4
"2011_09_30_drive_0018" # seq 5
"2011_09_30_drive_0020" # seq 6
"2011_09_30_drive_0027" # seq 7
"2011_09_30_drive_0028" # seq 8
"2011_09_30_drive_0033" # seq 9
"2011_09_30_drive_0034" # seq 10
)


MIN_SPACE_REQ=100 # Min space requiered for KITT dataset (raw) in GB

OUTPUT_DIR="${1:-KITTI}" # Ouput directory
KITTI_TYPE=${2:-extract} # Initialy we need unsynced data

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# get avaialble space
get_aval_space() {
    availSpace=$(df "${OUTPUT_DIR}" | awk 'NR==2 { print $4 }')
    space=$(echo "scale=0; $availSpace / (1024*1024)" | bc -l)
    echo $space
}

# Download sequences
downlaod_and_unpack_kitti_file() {
    # Params:
    # 1: date and drive 
    # 2: Zip-Name
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"$1"/"$2" && unzip -o "$2" && rm "$2"
}

# Download calibration files
downlaod_and_unpack_kitti_calib() {
    # Params:
    # 1: date and drive 
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/$1 && unzip -o "$1" && rm "$1"
}


# check if we have enough sapce on the output drive
space=$(get_aval_space )
if (( space < MIN_SPACE_REQ )); then
  echo "Error: Not enough Space available. Available: ${space}GB, Requiered: ${MIN_SPACE_REQ}GB" >&2
  exit 1
fi

cd ${OUTPUT_DIR}

for seq in "${SEQUENCES[@]}"; do
    seq_zip="${seq}_${KITTI_TYPE}.zip"
    echo "Ddownloading and extraxting ${seq_zip}."
    downlaod_and_unpack_kitti_file "${seq}" "${seq_zip}"
done

for seq in "${SEQUENCES[@]}"; do
    date="${seq:0:10}"
    calib_zip="${date}_calib.zip"
    echo "Ddownloading and extraxting calibration file ${calib_zip}."
    downlaod_and_unpack_kitti_calib "${calib_zip}"
done

cd "${SCRIPT_DIR}"
echo "Done!"
exit 0

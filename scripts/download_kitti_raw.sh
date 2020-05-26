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
"2011_09_26_drive_0023" # val. seq.
"2011_09_26_drive_0039" # val. seq.
)

MIN_SPACE_REQ=100 # Min space requiered for KITT dataset (raw) in GB

OUTPUT_DIR="${1:-KITTI}" # Ouput directory
KITTI_TYPE=${2:-sync} # Initialy we need unsynced data (sync, extract)
DL_DIR="${3:-zips}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# get avaialble space
get_aval_space() {
    availSpace=$(df "${OUTPUT_DIR}" | awk 'NR==2 { print $4 }')
    space=$(echo "scale=0; $availSpace / (1024*1024)" | bc -l)
    echo $space
}

# Download sequences
downlaod_kitti_file() {
    # Params:
    # 1: date and drive
    # 2: Zip-Name
    # 3: download path
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"$1"/"$2" -P "$3" # && unzip -o "$2" && rm "$2"
}

# Download calibration files
downlaod_kitti_calib() {
    # Params:
    # 1: date and drive
    # 2: download path
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/$1 -P "$2" # && unzip -o "$1" && rm "$1"
}


# extract zip
extract_zip_file() {
  # params
  # zip file
  # output dir
  7z x "$1" -o"$2"
}

# check if we have enough sapce on the output drive
space=$(get_aval_space )
if (( space < MIN_SPACE_REQ )); then
  echo "Error: Not enough Space available. Available: ${space}GB, Requiered: ${MIN_SPACE_REQ}GB" >&2
  exit 1
fi

#### First Download zip files
DL_PATH="${DL_DIR}/${KITTI_TYPE}"
rm -rf "${DL_PATH}" || true
mkdir -p "${DL_PATH}"

for seq in "${SEQUENCES[@]}"; do
    seq_zip="${seq}_${KITTI_TYPE}.zip"
    echo "Ddownloading ${seq_zip}."
    downlaod_kitti_file "${seq}" "${seq_zip}" "${DL_PATH}"
done

for seq in "${SEQUENCES[@]}"; do
    date="${seq:0:10}"
    calib_zip="${date}_calib.zip"
    echo "Ddownloading calibration file ${calib_zip}."
    downlaod_kitti_calib "${calib_zip}" "${DL_PATH}"
done

### Extract files
UNZIP_PATH="${OUTPUT_DIR}/data/${KITTI_TYPE}"
rm -rf "${UNZIP_PATH}" || true
mkdir -p "${UNZIP_PATH}"

for seq in "${SEQUENCES[@]}"; do
    seq_zip="${seq}_${KITTI_TYPE}.zip"
    echo "Extracting ${seq_zip}."
    extract_zip_file "${DL_PATH}/${seq_zip}" "${UNZIP_PATH}"
done

for seq in "${SEQUENCES[@]}"; do
    date="${seq:0:10}"
    calib_zip="${date}_calib.zip"
    echo "Extracting ${seq_zip}."
    extract_zip_file "${DL_PATH}/${calib_zip}" "${UNZIP_PATH}"
done


cd "${SCRIPT_DIR}"
echo "Done!"
exit 0


# DeepLIO
__Deep Lidar Inertial Odometry__

#### !!!This repo is under heavy development at the moment!!!

1. Intorduction

DeepLIO is an deep learning based odometry estimation using lidar and IMU. 

Cloning deeplio
```
git clone https://github.com/ArashJavan/DeepLIO.git
```


__1.1 Dependencies__

Following packages should be already installed, before you can start using deeplio.
- pytorch 
- tqdm (optional)
- open3d (optinal)

__1.2 Preparing the KITTI Dataset__

__Downloading KITTI__

In this project the KITTI Raw dataset is used, since we also need IMU measurments.
Please run the _download_kitti_raw.sh_ script to download the KITTI raw sequences.

__Note__: You will need at least 150 GB free sapce on your hard drive.

```
$ mkdir KITTI
$ download_kitti_raw.sh ./KITTI
```
Now wait till download is completed. Well it will take a long long time so go get some coffee :)

At the end you will find all sequences extracted under KITTI folder
```
KITTI
|
|-> 2011_09_30
    |-> 2011_09_30_drive_0016_extract
        |-> image0 .. imahe3
        |->oxts
        |->velodyne_points
    .
    .
    .
|-> 2011_10_03
    |-> 2011_09_03_drive_0027_extract
    .
    .
    .
```

__Converting Frames (optional)__

In the KITTI raw unsynced sequences the velodyne frames are saved as plain text files, 
consisting of x,y,z and remission of each measured point. 
Each frame's text file consists of several thousand points, which makes these file 
huge and also reading these files takes a long time. For that reason it is better to convert
them into a binary format first.
 
Please run following script to convert the raw text files to numpy binary.
 ```
cd scripts
python ./convert_velo_txt2bin.py -p KITTI/2011_09_30/2011_09_30_drive_0016_extract/velodyne_points/data/ \
KITTI/2011_09_30/2011_09_30_drive_0018_extract/velodyne_points/data/ [more velodyne data paths]
```
You can pass only one path or multiple at once. To accelerate conversion and save time the conversion script above 
starts multiple processes, so do not be afraid if your CPU is running under 100% load.

After the converting is done, you can start with training or evaluating.


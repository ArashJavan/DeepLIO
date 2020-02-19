import os
import torch
import yaml
import torch.utils.data as data


class KittiRaw(data.Dataset):
    def __init__(self, root_path, config, transform=None):
        """
        :param root_path:
        :param config: Configuration file including split settings
        :param transform:
        """
        self.root_path = root_path
        self.transform = transform

        self.image_dir = os.path.join(self.root_path, 'image_2')
        self.lidar_dir = os.path.join(self.root_path, 'velodyne')
        self.calib_dir = os.path.join(self.root_path, 'calib')
        self.label_dir = os.path.join(self.root_path, 'label_2')
        self.plane_dir = os.path.join(self.root_path, 'planes')


if __name__ == "__main__":
    cfg = yaml.load("../config.yaml")
    dataset = KittiRaw(r"C:\Users\ajava\Datasets\KITTI\unsync", config=cfg)
    print(dataset)
import glob
import cv2
import numpy as np

from torch.utils.data import Dataset

import config


class RGBDepthDataset(Dataset):
    """
    Dataloader from Task 1 providing CARLA depth and RGB images.
    """
    def __init__(self, dataset_dir=None, mode='train', split={'train': 0.8, 'val': 0.1, 'test': 0.1}):
        if dataset_dir is not None:
            # Prepare data from specified dataset directory
            self.split = split
            self.rgb_images, self.depth_images = self.make_dataset(dataset_dir)
            self.rgb_images, self.depth_images = self.select_split(mode)
        else:
            # Load default directory
            self.rgb_images, self.depth_images = self.load_default()

    @staticmethod
    def load_default():
        """
        Load default data recursively

        :param dataset_dir: directory where the data is located
        :return: (rgb_images, depth_images), filenames of the complete dataset
        """
        rgb_images = sorted(glob.glob(config.default_rgb_data_dir + '/**/*' + ".png",
                                      recursive=True))
        depth_images = sorted(glob.glob(config.default_depth_data_dir + '/**/*' + ".png",
                                        recursive=True))

        return rgb_images, depth_images

    @staticmethod
    def make_dataset(dataset_dir):
        """
        Load the filenames of the data recursively based on the given directory.

        :param dataset_dir: directory where the data is located
        :return: (rgb_images, depth_images), filenames of the complete dataset
        """
        rgb_images = sorted(glob.glob(dataset_dir + "CameraRGB?" + '/**/*' + ".png", recursive=True))
        depth_images = sorted(glob.glob(dataset_dir + "CameraDepth?" + '/**/*' + ".png", recursive=True))

        assert len(rgb_images) == len(depth_images)
        return rgb_images, depth_images

    def select_split(self, mode):
        """
        Split the dataset depending on its mode.

        :param mode: mode of the dataset, either train, test or val
        :return: (rgb_images, depth_images), where only the indices for the corresponding data split are selected.
        """
        fraction_train = self.split['train']
        fraction_val = self.split['val']
        num_samples = len(self.rgb_images)
        num_train = int(num_samples * fraction_train)
        num_val = int(num_samples * fraction_val)

        np.random.seed(0)
        rand_perm = np.random.permutation(num_samples)

        if mode == 'train':
            idx = rand_perm[:num_train]
        elif mode == 'val':
            idx = rand_perm[num_train:num_train + num_val]
        elif mode == 'test':
            idx = rand_perm[num_train + num_val:]

        return list(np.array(self.rgb_images)[idx]), list(np.array(self.depth_images)[idx])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        """
        Base on: /storage/group/intellisys/tutorial/dataloaderExample.ipynb
        Load the selected images.
        Return the images and corresponding filenames.
        """
        rgb_img_file = self.rgb_images[idx]
        depth_img_file = self.depth_images[idx]

        rgb_img = cv2.imread(rgb_img_file)
        rgb_img = cv2.resize(rgb_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # bgr to rgb. Since opencv loads images as bgr

        depth_img = cv2.imread(depth_img_file, 0)  # Load depth data as grayscale
        depth_img = cv2.resize(depth_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        # depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)  # bgr to rgb. Since opencv loads images as bgr
        depth_img = np.expand_dims(depth_img, 2)  # Add an extra channel dimension.
        # Converts (height, width) to (height, width, channel)

        rgb_img = np.transpose(rgb_img, (2, 0, 1))  # Since Pytorch models take tensors
        depth_img = np.transpose(depth_img, (2, 0, 1))  # in (channel, width, height) format

        data = dict()
        data['rgb'] = rgb_img.copy()
        data['depth'] = depth_img.copy()
        data['filenames'] = (rgb_img_file, depth_img_file)
        return data

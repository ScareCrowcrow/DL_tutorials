import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

import warnings
warnings.filterwarnings("ignore")

plt.ion()
N = 65


def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker=".", c="r")
    plt.pause(0.001)


def show_one_figure():
    landmarks_frame = pd.read_csv("data/faces/face_landmarks.csv")
    img_name = landmarks_frame.iloc[N, 0]
    landmarks = landmarks_frame.iloc[N, 1:].as_matrix()
    landmarks = landmarks.astype("float").reshape(-1, 2)
    print('First 4 Landmarks: {}'.format(landmarks[:4]))
    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    plt.figure()
    show_landmarks(io.imread(os.path.join("data/faces/", img_name)), landmarks)
    plt.show()


class FaceLandmarksDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        super(FaceLandmarksDataset, self).__init__()
        self.landmark_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmark_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_name = os.path.join(self.root_dir, self.landmark_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmark_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype("float").reshape(-1, 2)
        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample


def use_dataset_show():
    face_dataset = FaceLandmarksDataset(csv_file="data/faces/face_landmarks.csv", root_dir="data/faces")
    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]
        print(i, sample["image"].shape, sample["landmarks"].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title("Sample #{}".format(i))
        ax.axis("off")
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break


class Rescale:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w/w, new_h/h]
        return {"image": img, "landmarks": landmarks}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor:
    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image), "landmarks": torch.from_numpy(landmarks)}


def use_transform_show():
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])
    fig = plt.figure()
    face_dataset = FaceLandmarksDataset(csv_file="data/faces/face_landmarks.csv", root_dir="data/faces")
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_landmarks(**transformed_sample)

plt.show()


def user_transforms_show():
    transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                               root_dir='data/faces/',
                                               transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                               ]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['landmarks'].size())

        if i == 3:
            break


def show_landmarks_batch(sample_batch):
    images_batch, landmarks_batch = sample_batch["image"], sample_batch["landmarks"]
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size, s=10, marker=".", c="r")
        plt.title("Batch from dataloader")


def regular_transform():
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

if __name__ == "__main__":
    # show_one_figure()
    # use_dataset_show()
    # use_transform_show()
    # user_transforms_show()
    transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                               root_dir='data/faces/',
                                               transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                               ]))
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batch in enumerate(dataloader):
        print(i_batch, sample_batch["image"].size(), sample_batch["landmarks"].size())
        if i_batch == 3:
            plt.figure()
        show_landmarks_batch(sample_batch)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

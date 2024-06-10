import numpy as np
import copy
import torch
import time

Q_KEYS = (
    "Q_00",
    "Q_01",
    "Q_02",
    "Q_11",
    "Q_12",
    "Q_22",
    "is_gray",
    "volfrac",
    "is_broken",
)

def multiply(bin2d, M=4):
    '''implemented using vstack and hstack'''
    tdat3 = torch.tensor(bin2d)
    multiplied = torch.hstack([torch.vstack([tdat3] * M)] * M)
    return multiplied


def Q_to_vec(Q):
    return [Q[0, 0], Q[1, 1], Q[2, 2], Q[0, 1], Q[1, 2], Q[0, 2]]


def vec_to_dict(data):
    sample_dict = dict()
    for i, key in enumerate(Q_KEYS):
        sample_dict[key] = data[i]
    return sample_dict


def Q_to_dict(Q):
    d = {}
    for i in range(3):
        for j in range(3):
            if i <= j:
                d["Q_" + str(i) + str(j)] = Q[i, j]
    return d


def dict_to_Q(d):
    Q = np.zeros([3, 3], dtype=float)
    Q[0, 0] = d["Q_00"]
    Q[1, 1] = d["Q_11"]
    Q[2, 2] = d["Q_22"]
    Q[0, 1] = d["Q_01"]
    Q[1, 0] = d["Q_01"]
    Q[0, 2] = d["Q_02"]
    Q[2, 0] = d["Q_02"]
    Q[1, 2] = d["Q_12"]
    Q[2, 1] = d["Q_12"]
    return Q


class StructureTransform:
    def img_transform(self, img):
        pass

    def data_transform(self, data):
        pass

    def condition(self):
        pass

    def __call__(self, sample: dict):
        if self.condition():
          sample = copy.copy(
                sample
            )  # otherwise the transform changes the original data sample (outside of the function)
          img, data = sample["image"], sample["data"]
          img = self.img_transform(img)
          data = self.data_transform(data)
            
          sample["image"] = img
          sample["data"].update(data)
        return sample


class RandomTranspose(StructureTransform):
    def __init__(self, p=0.5):
        self.p = p

    def condition(self):
        return np.random.rand() < self.p

    def img_transform(self, img):
        return img.T

    def data_transform(self, data):
        Q = dict_to_Q(data)
        Q[[0, 1], :] = Q[[1, 0], :]
        Q[:, [0, 1]] = Q[:, [1, 0]]
        data_new = Q_to_dict(Q)
        return data_new


class RandomFlipUD(StructureTransform):
    def __init__(self, p=0.5):
        self.p = p

    def condition(self):
        return np.random.rand() <= self.p

    def img_transform(self, img):
        return np.flipud(img)

    def data_transform(self, data):
        data["Q_02"] = -data["Q_02"]
        data["Q_12"] = -data["Q_12"]
        return data


class RandomFlipLR(StructureTransform):
    def __init__(self, p=0.5):
        self.p = p

    def condition(self):
        return np.random.rand() <= self.p

    def img_transform(self, img):
        return np.fliplr(img)

    def data_transform(self, data):
        data["Q_02"] = -data["Q_02"]  # FIX_ME
        data["Q_12"] = -data["Q_12"]
        return data


class RandomRoll(StructureTransform):
    def condition(self):
        return True

    def img_transform(self, img):
        nx, ny = img.shape
        shift_x = np.random.randint(-nx, nx + 1)
        shift_y = np.random.randint(-ny, ny + 1)
        img = np.roll(img, shift_x, axis=0)
        img = np.roll(img, shift_y, axis=1)
        return img

    def data_transform(self, data):
        return data

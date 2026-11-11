import os
import random

import torch
import pandas as pd
from enum import Enum
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import pickle

from utils import HistFigure

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# IMAGENET_MEAN = [0.5, 0.5, 0.5]
# IMAGENET_STD = [0.5, 0.5, 0.5]

SCORE_PICKLE_PATH = Path("./score_dict.pkl")


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class ScoreReader:
    def __init__(self, source, score_file_name):
        self.score_file_name = score_file_name
        self.source = Path(source)
        if self.score_file_name not in os.listdir(self.source):
            raise Exception("score file name don't exist in data folder")
        self.df = pd.read_excel(self.source / self.score_file_name, header=0)
        self.files = {}
        for f in list(self.source.rglob('*.jpg'))+ list(self.source.rglob('*.tif')):
            self.files[f.stem] = f

    def read_score(self):
        if os.path.exists(SCORE_PICKLE_PATH):
            return self.load()
        score_dict = dict()
        stems = self.files.keys()
        found_list = []
        for row in self.df.itertuples():
            name = str(row[1])
            found_list.append(name)
            score_data = [_data for _data in row[2:]]
            if name is None or None in score_data[-2:] or name not in stems:
                print(name)
                continue

            expert_color_score = list()
            expert_texture_score = list()
            for i in range(0, len(score_data), 2):
                expert_color_score.append(score_data[i] if score_data[i] is not None else 0)
                expert_texture_score.append(score_data[i + 1] if score_data[i] is not None else 0)
            if len(expert_texture_score) == 0 or len(expert_color_score) == 0:
                continue
            score_dict[name] = {
                "name": name,
                "path": self.files[name],
                "color": {
                    "value": expert_color_score[:-2],
                    "mean": expert_color_score[-2],
                    "var": expert_color_score[-1]
                },
                "texture": {
                    "value": expert_texture_score[:-2],
                    "mean": expert_texture_score[-2],
                    "var": expert_texture_score[-1],
                }
            }
        self.save(score_dict)
        return score_dict

    @staticmethod
    def save(data):
        with open(SCORE_PICKLE_PATH, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load():
        with open(SCORE_PICKLE_PATH, "rb") as f:
            data = pickle.load(f)
        return data


class JadeDataset(Dataset):
    def __init__(
            self,
            source,
            image_size,
            data_sheet,
            image_size_color=None,
            image_size_texture=None,
            train=False,
    ):
        self.source = Path(source)
        self.image_size = image_size

        self.image_size_color = image_size_color or image_size
        self.image_size_texture = image_size_texture or image_size
        def make_transform(size):
            ops = [transforms.Resize(size + 32), transforms.CenterCrop(size)]
            if train:
                ops += [transforms.RandomRotation(10), transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5)]
            return transforms.Compose(ops + [transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        self.transform_color = make_transform(self.image_size_color)
        self.transform_texture = make_transform(self.image_size_texture)
        self.iter_data = data_sheet
        self.len = len(self.iter_data)

    def __getitem__(self, idx):
        image_dict = self.iter_data[idx]
        image = Image.open(image_dict["path"]).convert("RGB")

        image_color = self.transform_color(image)
        image_texture = self.transform_texture(image)
        return {
            "image": image_color,
            "image_color": image_color,
            "image_texture": image_texture,
            "class": image_dict["class"],
            "name": image_dict["name"],
            "color": self.get_color_from_name(image_dict["name"]),
            "color_score": image_dict["color_score"],
            "texture_score": image_dict["texture_score"],
            "color_variance": image_dict.get("color_variance", 0.0),
            "texture_variance": image_dict.get("texture_variance", 0.0),
            "color_onehot": self.onehot(image_dict["color_score"]),
            "texture_onehot": self.onehot(image_dict["texture_score"]),
        }

    @staticmethod
    def onehot(score):
        label = torch.zeros(10)
        label[round(score) - 1] = 1
        return label

    def __len__(self):
        return self.len

    def get_color_from_name(self, name):
        # green: 0 yellow: 1 purple:2

        if name[0] == "G":
            label = 0
        elif name[0] == "Y":
            label = 1
        elif name[0] == "P":
            label = 2
        else:
            raise Exception(f"unclear color found : {name}")
        zero_label = torch.zeros((3,), dtype=torch.float)
        zero_label[label] = 1
        label = zero_label
        return label


class JadeDataLoader:
    def __init__(self, source, score_file_name, image_size, batch_size, image_size_color=None, image_size_texture=None, **kwargs):
        self.source = Path(source)
        self.score_file_name = score_file_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.image_size_color = image_size_color or image_size
        self.image_size_texture = image_size_texture or image_size
        self.score_dict = self.read_score()
        self.data_sheet = self.summary_valid_data(self.score_dict)

    def read_score(self):
        score_reader = ScoreReader(
            source=self.source,
            score_file_name=self.score_file_name,
        )
        score_dict = score_reader.read_score()
        return score_dict

    def summary_valid_data(self, score_dict):
        random.seed(123)
        image_iter_data = list()
        for image_dir_name in os.listdir(self.source):
            sub_set = list()
            dir_path = self.source / image_dir_name
            if not os.path.isdir(dir_path):
                continue
            image_dir_list = []
            for image in os.listdir(dir_path):
                if (image.endswith(".jpg") or image.endswith(".tif")) and image.split(".")[0] in score_dict.keys():
                    image_dir_list.append(image)
            for image_name in image_dir_list:
                image_path = dir_path / image_name
                score = score_dict[image_name.split(".")[0]]
                image_info_dict = {
                    "path": image_path,
                    "class": image_dir_name,
                    "name": image_name,
                    "color_list": score["color"]["value"],
                    "texture_list": score["texture"]["value"],
                    "color_score": score["color"]["mean"],
                    "texture_score": score["texture"]["mean"],
                    "color_variance": score["color"].get("var", 0.0),
                    "texture_variance": score["texture"].get("var", 0.0)
                }
                sub_set.append(image_info_dict)
            random.shuffle(sub_set)
            image_iter_data.append(sub_set)
        return image_iter_data

    def get_loader(self, fold_index, fold_num=5):
        assert 0 < fold_index <= fold_num
        rows = [row for sub_set in self.data_sheet for row in sub_set]
        # Assign folds by image/item ID globally, so duplicate views or files
        # with the same stem can never appear in both train and validation.
        groups = sorted({Path(row["name"]).stem for row in rows})
        rng = random.Random(7)
        rng.shuffle(groups)
        fold_for_id = {item_id: i % fold_num + 1 for i, item_id in enumerate(groups)}
        train_data_sheet = [row for row in rows if fold_for_id[Path(row["name"]).stem] != fold_index]
        test_data_sheet = [row for row in rows if fold_for_id[Path(row["name"]).stem] == fold_index]
        print(f"train : {len(train_data_sheet)}  -- test : {len(test_data_sheet)}")
        train_dataset = JadeDataset(
            source=self.source,
            image_size=self.image_size,
            data_sheet=train_data_sheet,
            image_size_color=self.image_size_color,
            image_size_texture=self.image_size_texture,
            train=True
        )
        test_dataset = JadeDataset(
            source=self.source,
            image_size=self.image_size,
            data_sheet=test_data_sheet,
            image_size_color=self.image_size_color,
            image_size_texture=self.image_size_texture,
            train=False
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        return {
            "train": train_dataloader,
            "test": test_dataloader,
        }


class WholeData:
    def __init__(self, source, score_file_name, image_size, batch_size, **kwargs):
        self.source = Path(source)
        self.score_file_name = score_file_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.score_dict = self.read_score()
        self.data_sheet = self.summary_valid_data(self.score_dict)

    def read_score(self):
        score_reader = ScoreReader(
            source=self.source,
            score_file_name=self.score_file_name,
        )
        score_dict = score_reader.read_score()
        return score_dict

    def summary_valid_data(self, score_dict):
        image_iter_data = dict()
        for image_dir_name in os.listdir(self.source):
            sub_set = list()
            dir_path = self.source / image_dir_name
            if not os.path.isdir(dir_path):
                continue
            image_dir_list = []
            for image in os.listdir(dir_path):
                if (image.endswith(".jpg") or image.endswith(".tif")) and image.split(".")[0] in score_dict.keys():
                    image_dir_list.append(image)
            for image_name in image_dir_list:
                image_path = dir_path / image_name
                score = score_dict[image_name.split(".")[0]]
                image_info_dict = {
                    "path": image_path,
                    "name": image_name,
                    "color_list": score["color"]["value"],
                    "texture_list": score["texture"]["value"],
                    "color_score": score["color"]["mean"],
                    "texture_score": score["texture"]["mean"]
                }
                sub_set.append(image_info_dict)
            image_iter_data[image_dir_name] = sub_set
        return image_iter_data

    def get_whole_data(self):
        return self.data_sheet


if __name__ == '__main__':
    color_score_list = list()
    texture_score_list = list()
    for s in ScoreReader("./jade", "score.xlsx").read_score().values():
        color_score_list.append(s["color"]["mean"])
        texture_score_list.append(s["texture"]["mean"])
    hist = HistFigure("color score")
    hist.add_data(color_score_list)
    hist.save("color")
    hist = HistFigure("texture score")
    hist.add_data(texture_score_list)
    hist.save("texture")

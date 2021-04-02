import math
import os
import os.path as path
import numpy as np
import scipy.io
import torch
from PIL import Image
from colorama import Fore
from torchvision import transforms
from tqdm import trange

from cache import Cache


class Database:
    def __init__(self, database_url, dataset_url='G:/School/Deep Learning/data/'):
        self.dataset_url = dataset_url
        self.db = scipy.io.loadmat(database_url)
        self.num_images = self.db.get('dbStruct')[0][0][5][0][0]
        self.num_queries = self.db.get('dbStruct')[0][0][6][0][0]
        self.num_queries //= 10
        self.num_images //= 10
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(100),  # TODO: resize/crop?
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            # transforms.LinearTransformation(),  # TODO: PCA whitening
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # TODO: use normalization?
        ])
        self.cache = Cache(self)

        # TODO: Comments
        query_filename = path.join("./preprocessing",
                                   f"{database_url.split('/')[-1].split('.')[0]}_{self.num_queries}_query_tensors.dat")
        image_filename = path.join("./preprocessing",
                                   f"{database_url.split('/')[-1].split('.')[0]}_{self.num_images}_image_tensors.dat")

        exists = path.exists(query_filename)
        if not path.exists("preprocessing"):
            os.system("mkdir preprocessing")

        if not exists:
            self.query_tensors = np.memmap(query_filename, dtype="float32", mode="w+",
                                           shape=(self.num_queries, *self.__query_tensor_from_image(0).shape))
            self.image_tensors = np.memmap(image_filename, dtype="float32", mode="w+",
                                           shape=(self.num_images, *self.__image_tensor_from_image(0).shape))

            with trange(self.num_queries + self.num_images, position=0,
                        leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.WHITE)) as t:
                t.set_description(f'{"Processing " + str(database_url.split("/")[-1]) : <32}')
                for i in t:
                    if i < self.num_queries:
                        query_id = i
                        self.query_tensors[query_id] = self.__query_tensor_from_image(query_id).detach().numpy()
                    else:
                        image_id = i - self.num_queries
                        self.image_tensors[image_id] = self.__image_tensor_from_image(image_id).detach().numpy()

            self.query_tensors.flush()
            self.image_tensors.flush()

        else:
            print("Using preprocessed", database_url.split("/")[-1].split(".")[0])

        self.query_tensors = np.memmap(query_filename, dtype="float32", mode="r",
                                       shape=(self.num_queries, *self.__query_tensor_from_image(0).shape))
        self.image_tensors = np.memmap(image_filename, dtype="float32", mode="r",
                                       shape=(self.num_images, *self.__image_tensor_from_image(0).shape))

    def update_cache(self, net, t_parent=None):
        self.cache.update(net, t_parent=t_parent)

    def geo_distance(self, query_id, image_id):
        x1, y1 = self.query_position(query_id)
        x2, y2 = self.image_position(image_id)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def __query_tensor_from_image(self, query_id):
        input_image = Image.open(self.dataset_url + self.query_name(query_id))
        return self.preprocess(input_image)

    def query_tensor_from_stash(self, query_id):
        return torch.as_tensor(self.query_tensors[query_id])

    def query_tensor_batch(self, query_id, batch_size=1):
        end = min(query_id + batch_size, len(self.query_tensors))
        return torch.as_tensor(self.query_tensors[query_id:end])

    def __image_tensor_from_image(self, image_id):
        input_image = Image.open(self.dataset_url + self.image_name(image_id))
        return self.preprocess(input_image)

    def image_tensor_from_stash(self, image_id):
        return torch.as_tensor(self.image_tensors[image_id])

    def image_tensor_batch(self, image_id, batch_size=1):
        end = min(image_id + batch_size, len(self.image_tensors))
        return torch.as_tensor(self.image_tensors[image_id:end])

    def query_position(self, query_id):
        x = self.db.get('dbStruct')[0][0][4][0][query_id]
        y = self.db.get('dbStruct')[0][0][4][1][query_id]
        return x, y

    def image_position(self, image_id):
        x = self.db.get('dbStruct')[0][0][2][0][image_id]
        y = self.db.get('dbStruct')[0][0][2][1][image_id]
        return x, y

    def query_name(self, query_id):
        return self.db.get('dbStruct')[0][0][3][query_id][0][0]

    def image_name(self, image_id):
        return self.db.get('dbStruct')[0][0][1][image_id][0][0]

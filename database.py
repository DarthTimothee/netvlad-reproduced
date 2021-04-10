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
from helpers import ram_usage, get_device

device = get_device()


class Database:
    def __init__(self, data_path, database, input_scale=224, preprocess_mode=None, whitening=False):
        self.input_scale = (input_scale, input_scale) if isinstance(input_scale, int) else input_scale
        self.preprocess_mode = preprocess_mode
        self.whitening = whitening
        self.data_path = data_path
        self.query_url = path.join(data_path, 'queries_real')
        self.__parse_db(database_path=path.join(data_path, 'datasets', f"{database}.mat"))
        self.input_transform = self.__get_input_transform()
        if self.preprocess_mode in ['disk', 'ram']:
            self.__preprocess(name=database)
        self.cache = Cache(self)

    def update_cache(self, net, t_parent=None):
        self.cache.update(net, t_parent=t_parent)

    def get_query_tensor(self, query_id, batch_size=1):
        end = min(query_id + batch_size, self.num_queries)
        if not self.preprocess_mode:
            if batch_size == 1:
                self.__query_tensor_from_image(query_id)
            return torch.stack([self.__query_tensor_from_image(i) for i in range(query_id, end)])
        return torch.as_tensor(self.query_tensors[query_id:end])

    def get_image_tensor(self, image_id, batch_size=1):
        end = min(image_id + batch_size, self.num_images)
        if not self.preprocess_mode:
            if batch_size == 1:
                self.__image_tensor_from_image(image_id)
            return torch.stack([self.__image_tensor_from_image(i) for i in range(image_id, end)])
        return torch.as_tensor(self.image_tensors[image_id:end])

    def geo_distance(self, query_id, image_id):
        x1, y1 = self.query_positions[query_id]
        x2, y2 = self.image_positions[image_id]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def query_filename(self, query_id):
        return self.query_filenames[query_id]

    def image_filename(self, image_id):
        return self.image_filenames[image_id]

    def __get_input_transform(self):
        return transforms.Compose(list(filter(None, [
            transforms.Resize(size=self.input_scale) if self.input_scale else None,
            transforms.ToTensor(),
            None if self.whitening else None,  # TODO: PCA whitening

            # The mean and std are required by required by the alexnet / vgg16 base network
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if not self.whitening else None,
        ])))

    def __image_tensor_from_image(self, image_id):
        filename = path.join(self.data_path, self.image_filename(image_id))
        input_image = Image.open(filename)
        return self.input_transform(input_image)

    def __query_tensor_from_image(self, query_id):
        filename = path.join(self.query_url, self.query_filename(query_id))
        input_image = Image.open(filename)
        return self.input_transform(input_image)

    def __parse_db(self, database_path):
        db = scipy.io.loadmat(database_path)

        self.num_images = db.get('dbStruct')[0][0][5][0][0]
        self.image_positions = [(0, 0)] * self.num_images
        self.image_filenames = [""] * self.num_images
        for image_id in range(self.num_images):
            x = db.get('dbStruct')[0][0][2][0][image_id]
            y = db.get('dbStruct')[0][0][2][1][image_id]
            f = db.get('dbStruct')[0][0][1][image_id][0][0]
            self.image_positions[image_id] = (x, y)
            self.image_filenames[image_id] = f

        self.num_queries = db.get('dbStruct')[0][0][6][0][0]
        self.query_positions = [(0, 0)] * self.num_queries
        self.query_filenames = [""] * self.num_queries
        for query_id in range(self.num_queries):
            x = db.get('dbStruct')[0][0][4][0][query_id]
            y = db.get('dbStruct')[0][0][4][1][query_id]
            f = db.get('dbStruct')[0][0][3][query_id][0][0]
            self.query_positions[query_id] = (x, y)
            self.query_filenames[query_id] = f

    def __preprocess(self, name):
        r = name
        query_stash_shape = (self.num_queries, *self.__query_tensor_from_image(0).shape)
        image_stash_shape = (self.num_images, *self.__image_tensor_from_image(0).shape)

        if self.preprocess_mode == 'ram':
            self.query_tensors = np.zeros(query_stash_shape, dtype="float32")
            self.image_tensors = np.zeros(image_stash_shape, dtype="float32")
            self.__build_stash(desc=r)

        w = '_white' if self.whitening else ''
        res = self.input_scale[0] if self.input_scale else 'fullres'
        query_filename = path.join("./preprocessing", f"{r}_{self.num_queries}_{res}{w}_query_tensors.dat")
        image_filename = path.join("./preprocessing", f"{r}_{self.num_images}_{res}{w}_image_tensors.dat")

        if path.exists(query_filename):
            print("Using preprocessed data", r)
            self.query_tensors = np.memmap(query_filename, dtype="float32", mode="r", shape=query_stash_shape)
            self.image_tensors = np.memmap(image_filename, dtype="float32", mode="r", shape=image_stash_shape)
            return

        if not path.exists("preprocessing"):
            os.system("mkdir preprocessing")
        self.query_tensors = np.memmap(query_filename, dtype="float32", mode="w+", shape=query_stash_shape)
        self.image_tensors = np.memmap(image_filename, dtype="float32", mode="w+", shape=image_stash_shape)

        self.__build_stash(desc=r)

        self.query_tensors.flush()
        self.image_tensors.flush()

    def __build_stash(self, desc=''):
        with trange(self.num_queries + self.num_images, position=0,
                    leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.WHITE)) as t:
            t.set_description(f'{"Processing " + desc : <32}')
            for i in t:
                if i < self.num_queries:
                    query_id = i
                    self.query_tensors[query_id] = self.__query_tensor_from_image(query_id).detach().numpy()
                else:
                    image_id = i - self.num_queries
                    self.image_tensors[image_id] = self.__image_tensor_from_image(image_id).detach().numpy()
                t.set_postfix(ram_usage=ram_usage())

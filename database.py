import math
import random
import numpy as np
import scipy.io
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from TrainingTuple import TrainingTuple
import time


class Vlataset(Dataset):
    def __init__(self, net, database_url='./datasets/pitts250k_train.mat'):
        self.net = net
        self.db = scipy.io.loadmat(database_url)
        self.num_images = self.db.get('dbStruct')[0][0][5][0][0]
        self.num_queries = self.db.get('dbStruct')[0][0][6][0][0]
        # TODO: load all data from the .mat into variables
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            # transforms.Resize(256),  # TODO: resize/crop?
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # TODO: keep these normalize constants?
        ])
        self.pairwise_distance = nn.PairwiseDistance()
        self.training_tuples = [TrainingTuple(self.__query_name(i)) for i in range(self.num_queries)]  # TODO
        start = time.time()
        self.__init_potential_positives()
        print("__init_potential_positives:", (start - time.time()))

    def __geo_distance(self, query_index, image_index):
        x1, y1 = self.__query_position(query_index)
        x2, y2 = self.__image_position(image_index)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def __query_position(self, query_index):
        x = self.db.get('dbStruct')[0][0][4][0][query_index]
        y = self.db.get('dbStruct')[0][0][4][1][query_index]
        return x, y

    def __image_position(self, image_index):
        x = self.db.get('dbStruct')[0][0][2][0][image_index]
        y = self.db.get('dbStruct')[0][0][2][1][image_index]
        return x, y

    def __init_potential_positives(self):
        for query_index, tt in enumerate(self.training_tuples):
            print(query_index)
            for image_index in range(self.num_images):
                if self.__geo_distance(query_index, image_index) < 10:
                    tt.potential_positives.append(image_index)

    def __write_positives_to_file(self):
        for tt in self.training_tuples:
            arr = np.array(tt.potential_positives)
            np.savetxt()

    def __query_name(self, image_id):
        return self.db.get('dbStruct')[0][0][3][image_id][0][0]

    def __image_name(self, image_id):
        return self.db.get('dbStruct')[0][0][1][image_id][0][0]

    def __get_tensor(self, query):
        input_image = Image.open('G:/School/Deep Learning/data/' + query)
        return self.preprocess(input_image)

    def __vlad_distance(self, vlad1, vlad2):
        return self.pairwise_distance(vlad1 - vlad2)  # TODO: is this correct?

    def __image_distance(self, image1, image2):
        # TODO: use cache
        vlad1 = self.net(self.__get_tensor(image1))
        vlad2 = self.net(self.__get_tensor(image2))
        return self.__vlad_distance(vlad1, vlad2)

    def __best_positive(self, training_tuple):
        sorted_positives = sorted(training_tuple.potential_positives,
                                  key=lambda x: self.__image_distance(self.__image_name(x), training_tuple.query))
        return sorted_positives[0]

    def __get_1000_negatives(self, training_tuple):
        negatives = [0] * 1000
        while len(negatives) < 1000:  # TODO: maybe optimize later?
            image_id = random.randint(0, self.num_images)
            if image_id != training_tuple.query_id \
                    and image_id not in training_tuple.previous_hard_negatives\
                    and self.geo_distance(image_id, training_tuple.query_id) > 25:
                negatives.append(image_id)
        return negatives

    def __hard_negatives(self, training_tuple):
        sampled_negatives = sorted(self.__get_1000_negatives(training_tuple),
                                   key=lambda x: self.__image_distance(self.__image_name(x), training_tuple.query))
        current_hard_negatives = sampled_negatives[:10]

        # Remember the current hard negatives for the next epoch
        previous_hard_negatives = self.previous_hard_negatives
        self.previous_hard_negatives = current_hard_negatives

        return current_hard_negatives + previous_hard_negatives

    def __getitem__(self, index):
        training_tuple = self.training_tuples[index]
        query = training_tuple.query
        best_positive = self.__best_positive(training_tuple)
        hard_negatives = self.__hard_negatives(training_tuple)
        asdf = self.__get_tensor(best_positive)
        fdsa = [self.__get_tensor(n) for n in hard_negatives]
        return self.__get_tensor(query), query, asdf, fdsa

    def __len__(self):
        return len(self.training_tuples)

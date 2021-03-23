import math
import random
from collections import defaultdict
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from TrainingTuple import TrainingTuple


class Vlataset(Dataset):
    def __init__(self, net, database_url):
        self.net = net
        self.db = scipy.io.loadmat(database_url)
        self.num_images = self.db.get('dbStruct')[0][0][5][0][0]
        self.num_queries = self.db.get('dbStruct')[0][0][6][0][0]
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            # transforms.Resize(256),  # TODO: resize/crop?
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # TODO: keep these normalize constants?
        ])
        self.pairwise_distance = nn.PairwiseDistance()
        self.training_tuples = [TrainingTuple(i, self.__query_name(i)) for i in range(self.num_queries)]
        self.__init_potential_positives()
        self.counter = 0

    def __init_potential_positives(self):
        # Group all the queries with the same position together
        pos_query = defaultdict(list)
        for query_index in range(self.num_queries):
            x, y = self.__image_position(query_index)
            pos_query[(x, y)].append(query_index)

        # Group all the images with the same position together
        pos_image = defaultdict(list)
        for image_index in range(self.num_images):
            x, y = self.__image_position(image_index)
            pos_image[(x, y)].append(image_index)

        # Find the image positions close to the query positions
        for pos_q, query_indices in pos_query.items():
            for pos_i, image_indices in pos_image.items():
                x1, y1 = pos_q
                x2, y2 = pos_i
                if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < 10:
                    # Store all the image positions in all the training tuples
                    for query_index in query_indices:
                        training_tuple = self.training_tuples[query_index]
                        for image_index in image_indices:
                            training_tuple.potential_positives.append(image_index)

    def __vlad_distance(self, vlad1, vlad2):
        return torch.sum(self.pairwise_distance(vlad1, vlad2)).detach().numpy()  # TODO: is this correct?

    def __image_distance(self, image1, image2):
        self.counter += 1
        print(self.counter)
        # TODO: use cache
        #with torch.no_grad():
        vlad1 = self.net(self.__get_tensor(image1).unsqueeze(0))
        vlad2 = self.net(self.__get_tensor(image2).unsqueeze(0))
        return self.__vlad_distance(vlad1, vlad2)

    def __best_positive(self, training_tuple):
        print("sorting potential positives")
        sorted_positives = sorted(training_tuple.potential_positives,
                                  key=lambda x: self.__image_distance(self.__image_name(x), training_tuple.query))
        return sorted_positives[0]

    def __geo_distance(self, query_index, image_index):
        x1, y1 = self.__query_position(query_index)
        x2, y2 = self.__image_position(image_index)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def __get_1000_negatives(self, training_tuple):
        print("sampling 1000 negatives")
        negatives = [0] * 1000
        while len(negatives) < 1000:  # TODO: maybe optimize later?
            print(str(len(negatives)) + "/1000")
            image_index = random.randint(0, self.num_images)
            if image_index != training_tuple.query_id \
                    and image_index not in training_tuple.previous_hard_negatives \
                    and self.__geo_distance(training_tuple.query_index, image_index) > 25:
                negatives.append(image_index)
        print("sorting 1000 negatives")
        return negatives

    def __hard_negatives(self, training_tuple):
        self.counter = 0
        sampled_negatives = sorted(self.__get_1000_negatives(training_tuple),
                                   key=lambda x: self.__image_distance(self.__image_name(x), training_tuple.query))
        current_hard_negatives = sampled_negatives[:10]

        # Remember the current hard negatives for the next epoch
        previous_hard_negatives = training_tuple.previous_hard_negatives
        training_tuple.previous_hard_negatives = current_hard_negatives

        return current_hard_negatives + previous_hard_negatives

    def __get_tensor(self, query):
        input_image = Image.open('G:/School/Deep Learning/data/' + query)
        return self.preprocess(input_image)

    def __query_position(self, query_index):
        x = self.db.get('dbStruct')[0][0][4][0][query_index]
        y = self.db.get('dbStruct')[0][0][4][1][query_index]
        return x, y

    def __image_position(self, image_index):
        x = self.db.get('dbStruct')[0][0][2][0][image_index]
        y = self.db.get('dbStruct')[0][0][2][1][image_index]
        return x, y

    def __query_name(self, image_id):
        return self.db.get('dbStruct')[0][0][3][image_id][0][0]

    def __image_name(self, image_id):
        return self.db.get('dbStruct')[0][0][1][image_id][0][0]

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

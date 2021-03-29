import math
import random
from collections import defaultdict

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset


class Vlataset(Dataset):
    def __init__(self, database):
        self.database = database
        self.pairwise_distance = nn.PairwiseDistance()
        self.potential_positives = [list() for _ in range(self.database.num_queries)]
        self.previous_hard_negatives = [list() for _ in range(self.database.num_queries)]
        self.__init_potential_positives()

    def __init_potential_positives(self):
        # Group all the queries with the same position together
        pos_query = defaultdict(list)
        for query_id in range(self.database.num_queries):
            x, y = self.database.image_position(query_id)
            pos_query[(x, y)].append(query_id)

        # Group all the images with the same position together
        pos_image = defaultdict(list)
        for image_id in range(self.database.num_images):
            x, y = self.database.image_position(image_id)
            pos_image[(x, y)].append(image_id)

        # Find the image positions close to the query positions
        for pos_q, query_indices in pos_query.items():
            for pos_i, image_indices in pos_image.items():
                x1, y1 = pos_q
                x2, y2 = pos_i
                if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < 10:
                    # Store all the image positions in all the training tuples
                    for query_id in query_indices:
                        potential_positives = self.potential_positives[query_id]
                        for image_id in image_indices:
                            potential_positives.append(image_id)

    def __distance_to_query(self, query_id, image_id):
        vlad1 = self.database.cache.query_vlads[query_id]
        vlad2 = self.database.cache.image_vlads[image_id]
        return np.linalg.norm(vlad1 - vlad2)

    def __best_positive(self, query_id):
        sorted_positives = sorted(self.potential_positives[query_id],
                                  key=lambda image_id: self.__distance_to_query(query_id, image_id))
        return sorted_positives[0]  # TODO: some queries don't have any positives?

    def __get_1000_negatives(self, query_id):
        negatives = list()
        while len(negatives) < 1000:  # TODO: maybe optimize later?
            image_id = random.randint(0, self.database.num_images - 1)
            if image_id != query_id \
                    and image_id not in self.previous_hard_negatives[query_id] \
                    and self.database.geo_distance(query_id, image_id) > 25:
                negatives.append(image_id)
        return negatives

    def __hard_negatives(self, query_id):
        all_negatives = self.__get_1000_negatives(query_id) + self.previous_hard_negatives[query_id]
        sampled_negatives = sorted(all_negatives,
                                   key=lambda image_id: self.__distance_to_query(query_id, image_id))
        hard_negatives = sampled_negatives[:10]
        self.previous_hard_negatives[query_id] = hard_negatives
        return hard_negatives

    def __getitem__(self, query_id):
        best_positive = self.__best_positive(query_id)
        hard_negatives = self.__hard_negatives(query_id)
        query_tensor = self.database.query_to_tensor(query_id)
        positive_tensor = self.database.image_to_tensor(best_positive)
        negative_tensors = [self.database.image_to_tensor(n) for n in hard_negatives]
        return query_id, query_tensor, positive_tensor, negative_tensors

    def __len__(self):
        return self.database.num_queries

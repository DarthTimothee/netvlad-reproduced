import random


class TrainingTuple:
    def __init__(self, query_id):
        self.query_id = query_id
        self.potential_positives = self.__init_potential_positives()
        self.previous_hard_negatives = []

    def get_best_positive(self):
        return self.query_id  # TODO

    def get_hard_negatives(self, num_images):
        """Returns a list of the 10 hardest images that are far away from the query image,
        plus the 10 hardest negatives from the previous epoch."""
        sampled_negatives = self.__get_1000_negatives(num_images)
        # TODO: sort the list based on 'difficulty'
        previous_hard_negatives = self.previous_hard_negatives
        current_hard_negatives = sampled_negatives[:10]
        # Remember the current hard negatives for the next epoch
        self.previous_hard_negatives = current_hard_negatives
        return current_hard_negatives + previous_hard_negatives

    def __get_potential_positives(self):
        """Returns the list of all the images that are in close geographic proximity
        to the query image."""
        return self.potential_positives

    def __init_potential_positives(self):
        """Calculate and return the list of all the images that are
        in close geographic proximity to the query image."""
        return [self.query_id]  # TODO

    def __get_1000_negatives(self, num_images):
        """Returns a list of 1000 randomly sampled image that are far away (geographically)
        from the query image."""
        negatives = list()
        while len(negatives) < 1000:
            image = random.randint(0, num_images)  # TODO: hardcode correct values
            if image not in self.potential_positives \
                    and image not in self.previous_hard_negatives:
                negatives.append(image)
        return negatives

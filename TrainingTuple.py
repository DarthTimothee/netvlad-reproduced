import random


class TrainingTuple:
    def __init__(self, query_image):
        self.query_image = query_image
        self.potential_positives = self.__init_potential_positives()
        self.previous_hard_negatives = []

    def get_potential_positives(self):
        """Returns the list of all the images that are in close geographic proximity
        to the query image."""
        return self.potential_positives

    def get_hard_negatives(self):
        """Returns a list of the 10 hardest images that are far away from the query image,
        plus the 10 hardest negatives from the previous epoch."""
        sampled_negatives = self.__get_1000_negatives()
        # TODO: sort the list based on 'difficulty'
        previous_hard_negatives = self.previous_hard_negatives
        current_hard_negatives = sampled_negatives[:10]
        # Remember the current hard negatives for the next epoch
        self.previous_hard_negatives = current_hard_negatives
        return current_hard_negatives + previous_hard_negatives

    def __init_potential_positives(self):
        """Calculate and return the list of all the images that are
        in close geographic proximity to the query image."""
        return [self.query_image]  # TODO

    def __get_1000_negatives(self):
        """Returns a list of 1000 randomly sampled image that are far away (geographically)
        from the query image."""
        negatives = list()
        while len(negatives) < 1000:
            image = random.randint(0, 250000)  # TODO: don't hardcode
            if image not in self.potential_positives \
                    and image not in self.previous_hard_negatives:
                negatives.append(image)
        return negatives

class TrainingTuple:
    def __init__(self, query):
        self.query = query
        self.potential_positives = list()
        self.previous_hard_negatives = []

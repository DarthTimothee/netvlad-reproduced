class TrainingTuple:
    def __init__(self, query_index, query):
        self.query_index = query_index
        self.query = query
        self.potential_positives = list()
        self.previous_hard_negatives = []

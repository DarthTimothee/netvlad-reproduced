from tqdm import tqdm as progress


class Cache:
    def __init__(self, database, net):
        # TODO: use memmap or something
        self.vlad_vectors = [0] * database.num_queries
        print("Caching the vlad-vectors")
        for query_id in progress(range(database.num_queries)):
            query_tensor = database.query_to_tensor(query_id).unsqueeze(0)
            self.vlad_vectors[query_id] = net(query_tensor).detach().numpy()

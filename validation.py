import numpy as np
from colorama import Fore
from annoy import AnnoyIndex
from helpers import pbar


def validate(net, database, n_trees=1000):
    all_n = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    total_correct = np.zeros(len(all_n))

    index = AnnoyIndex(net.K * net.D, "euclidean")
    t = pbar(database.cache.image_vlads, color=Fore.GREEN, desc="Building index (for validation)")
    for i, i_vlad in enumerate(t):
        index.add_item(i, i_vlad.flatten())
    index.build(n_trees)

    t = pbar(range(database.num_queries), color=Fore.GREEN, desc="Validating")
    for query_id in t:
        # Find the nearest VLAD vectors using annoy index
        nearest_image_ids = index.get_nns_by_vector(database.cache.query_vlads[query_id].flatten(), 25)
        for i, n in enumerate(all_n):
            distances = np.array([database.geo_distance(query_id, image_id) for image_id in nearest_image_ids[:n]])
            if distances.min() <= 25:
                total_correct[i] += 1
            else:
                break

    total_correct = total_correct / database.num_queries * 100
    t.set_postfix(accuracy=total_correct)
    return total_correct

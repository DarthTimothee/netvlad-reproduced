import numpy as np
from sklearn.neighbors import NearestNeighbors


def validate(net, database):
    all_n = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    total_correct = np.zeros(len(all_n))

    nn = NearestNeighbors(n_neighbors=all_n[-1], p=2)

    nn.fit(database.cache.image_vlads.reshape((-1, net.K * net.D)))
    all_neighbors = nn.kneighbors(database.cache.query_vlads.reshape((-1, net.K * net.D)), return_distance=False)

    for query_id, neighbors in enumerate(all_neighbors):
        for i, n in enumerate(all_n):
            geo_distances = np.array([database.geo_distance(query_id, image_id) for image_id in neighbors[:n]])
            if geo_distances.min() <= 25:
                total_correct[i:] += 1
                break

    total_correct = total_correct / database.num_queries * 100
    return total_correct

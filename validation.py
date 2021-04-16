import faiss
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch import cuda
from helpers import get_device
import time

device = get_device()


def validate(net, database, use_faiss=True):
    all_n = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    total_correct = np.zeros(len(all_n))

    start = time.time()
    if use_faiss:
        print('Faiss -> fitting to cache for validation')
        faiss_index = faiss.IndexFlatL2(net.D * net.K)

        if cuda.is_available():
            net.cpu()
            cuda.empty_cache()
            config = faiss.GpuIndexFlatConfig()
            config.useFloat16 = True
            res = faiss.StandardGpuResources()
            faiss_index = faiss.GpuIndexFlatL2(res, faiss_index, config)

        faiss_index.add(database.cache.image_vlads)
        _, all_neighbors = faiss_index.search(database.cache.query_vlads, max(all_n))

        if cuda.is_available():
            net.to(device)

    else:
        print('Sklearn -> fitting to cache for validation')
        nn = NearestNeighbors(n_neighbors=all_n[-1], p=2, n_jobs=-1)
        nn.fit(database.cache.image_vlads)
        all_neighbors = nn.kneighbors(database.cache.query_vlads, return_distance=False)

    print('Done fitting, starting validation...')
    print(f"took: {time.time() - start: <5.1f} seconds", flush=True)
    for query_id, neighbors in enumerate(all_neighbors):
        for i, n in enumerate(all_n):
            geo_distances = np.array([database.geo_distance(query_id, image_id) for image_id in neighbors[:n]])
            if geo_distances.min() <= 25:
                total_correct[i:] += 1
                break

    total_correct = total_correct / database.num_queries * 100
    return total_correct

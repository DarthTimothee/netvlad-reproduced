import numpy as np
import faiss
from tqdm import tqdm


def get_clusters(K, D, db, base_network, N=25, num_samples=1000):  # TODO: don't hardcode N=25

    ids = np.random.randint(low=0, high=db.num_images, size=num_samples)

    features = np.zeros((num_samples * N, D)).astype('float32')
    for i, v in tqdm(enumerate(ids)):  # TODO: batches
        features[i * N:(i + 1) * N] = base_network(db.image_to_tensor(v).unsqueeze(0)).reshape(D, N).T

    model = faiss.Kmeans(D, K, verbose=True)
    model.train(features)

    return model.centroids
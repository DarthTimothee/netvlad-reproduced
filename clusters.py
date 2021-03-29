import numpy as np
import faiss
from tqdm import tqdm
from colorama import Fore


def get_clusters(K, D, db, base_network, N=25, num_samples=100):  # TODO: don't hardcode N=25

    ids = np.random.randint(low=0, high=db.num_images, size=num_samples)

    features = np.zeros((num_samples * N, D)).astype('float32')
    # TODO: batches
    with tqdm(ids, position=0,
                            leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Fore.YELLOW)) as t:
        t.set_description("Calculating cluster centers\t")
        for i, v in enumerate(t):
            features[i * N:(i + 1) * N] = base_network(db.image_to_tensor(v).unsqueeze(0)).reshape(D, N).T

    model = faiss.Kmeans(D, K, verbose=False)
    model.train(features)

    return model.centroids
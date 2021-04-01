import numpy as np
from colorama import Fore
from tqdm import tqdm


def nearest_images_to_query(database, query_id, num_nearest):
    distances = np.linalg.norm(database.cache.image_vlads - database.cache.query_vlads[query_id], axis=1).squeeze()
    return np.argpartition(distances, range(num_nearest))[:num_nearest]


def validate(net, database):
    # database = Database('./datasets/pitts30k_val.mat', dataset_url='./data/')
    database.update_cache(net)

    all_n = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    total_correct = [0] * len(all_n)

    with tqdm(range(database.num_queries), position=0,
              leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.GREEN)) as t:
        t.set_description(f'{"Validating " : <32}')

        for query_id in t:
            # Find the nearest VLAD vectors
            nearest_image_ids = nearest_images_to_query(database, query_id, all_n[-1])
            for i, n in enumerate(all_n):
                for image_id in nearest_image_ids[:n]:
                    distance = database.geo_distance(query_id, image_id)
                    if distance <= 25:
                        total_correct[i] += 1
                        break

        for i, n in enumerate(all_n):
            total_correct[i] /= database.num_queries
            total_correct[i] *= 100

        print()
        print(total_correct)

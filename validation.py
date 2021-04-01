import numpy as np
from colorama import Fore
from tqdm import tqdm


def distance_to_query(database, query_id, image_id):
    vlad1 = database.cache.query_vlads[query_id]
    vlad2 = database.cache.image_vlads[image_id]
    print()
    print(query_id, image_id)
    print(vlad1)
    print("_________________")
    print(vlad2)
    print()
    res = np.linalg.norm(vlad1 - vlad2)
    print(res)
    return res


def nearest_images_to_query(database, query_id, num_nearest):
    nearest_images = np.zeros(num_nearest, dtype=int)
    distances = np.zeros(num_nearest)
    for image_id in range(num_nearest):
        distance = distance_to_query(database, query_id, image_id)
        nearest_images[image_id] = image_id
        distances[image_id] = distance

    worst_distance_index = np.argmax(distances)
    worst_distance = distances[worst_distance_index]
    for image_id in range(num_nearest, database.num_images):
        distance = distance_to_query(database, query_id, image_id)
        if distance < worst_distance:
            nearest_images[worst_distance_index] = image_id
            distance[worst_distance_index] = distance
            worst_distance_index = np.argmax(distances)
            worst_distance = distances[worst_distance_index]

    nearest_image_ids = [x for _, x in sorted(zip(distances, nearest_images))]
    return nearest_image_ids


def validate(net, database):
    # database = Database('./datasets/pitts30k_val.mat', dataset_url='./data/')
    database.update_cache(net)

    all_n = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    total_correct = [0] * len(all_n)

    with tqdm(range(database.num_queries), position=0,
              leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.GREEN)) as t:
        t.set_description(f'{"Validating " : <32}')

        for query_id in t:
            nearest_image_ids = nearest_images_to_query(database, query_id, all_n[-1])
            corrects = [0] * len(nearest_image_ids)
            print(nearest_image_ids)
            for i, image_id in enumerate(nearest_image_ids):
                distance = database.geo_distance(query_id, image_id)
                if distance <= 25:
                    corrects[i] = 1
            for i, n in enumerate(all_n):
                total_correct[i] += sum(corrects[:n])

        for i, n in enumerate(all_n):
            total_correct[i] /= n
            total_correct[i] /= database.num_queries
            total_correct[i] *= 100

        print(total_correct)

import os.path as path
from tempfile import mkdtemp

import numpy as np
import torch
from colorama import Fore
from tqdm.auto import tqdm


class Cache:
    def __init__(self, database):
        self.database = database
        self.net = None
        self.query_filename = path.join(mkdtemp(), "query_vlads.dat")
        self.image_filename = path.join(mkdtemp(), "image_vlads.dat")
        self.query_vlads = None
        self.image_vlads = None

    def update(self, net, t_parent=None, cache_batch_size=42):
        if not self.net:
            self.net = net

            self.query_vlads = np.memmap(self.query_filename, dtype="float32", mode="w+",
                                         shape=(self.database.num_queries, net.D, net.K, 1))
            self.image_vlads = np.memmap(self.image_filename, dtype="float32", mode="w+",
                                         shape=(self.database.num_images, net.D, net.K, 1))

        net.freeze()
        net.eval()

        n_images = self.database.num_images
        n_queries = self.database.num_queries
        total = n_queries + n_images

        with torch.no_grad():

            if t_parent:
                for query_id in range(0, n_queries, cache_batch_size):
                    q_tensors = self.database.query_to_tensor_batch(query_id, batch_size=cache_batch_size)
                    q_vlads = net(q_tensors).permute(2, 0, 1).unsqueeze(-1).detach().numpy()
                    self.query_vlads[query_id:query_id + q_vlads.shape[0]] = q_vlads
                    t_parent.set_postfix(updating_cache=f"{round((query_id / total) * 100)}%")

                for image_id in range(0, n_images, cache_batch_size):
                    i_tensors = self.database.image_to_tensor_batch(image_id, batch_size=cache_batch_size)
                    i_vlads = net(i_tensors).permute(2, 0, 1).unsqueeze(-1).detach().numpy()
                    self.image_vlads[image_id:image_id + i_vlads.shape[0]] = i_vlads
                    t_parent.set_postfix(updating_cache=f"{round(((n_queries + image_id) / total) * 100)}%")

            else:
                bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.MAGENTA)
                progress = tqdm(total=total, position=0, leave=True, bar_format=bar_format)
                progress.set_description(f'{"Building the cache" : <32}')
                for query_id in range(0, n_queries, cache_batch_size):
                    q_tensors = self.database.query_to_tensor_batch(query_id, cache_batch_size)
                    q_vlads = net(q_tensors).permute(2, 0, 1).unsqueeze(-1).detach().numpy()
                    self.query_vlads[query_id:query_id + q_vlads.shape[0]] = q_vlads
                    progress.update(q_vlads.shape[0])

                for image_id in range(0, n_images, cache_batch_size):
                    i_tensors = self.database.image_to_tensor_batch(image_id, cache_batch_size)
                    i_vlads = net(i_tensors).permute(2, 0, 1).unsqueeze(-1).detach().numpy()
                    self.image_vlads[image_id:image_id + i_vlads.shape[0]] = i_vlads
                    progress.update(i_vlads.shape[0])

                progress.close()

            net.unfreeze()
            net.train()

            self.query_vlads.flush()
            self.image_vlads.flush()

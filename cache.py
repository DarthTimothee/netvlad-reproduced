import os.path as path
from tempfile import mkdtemp
import numpy as np
import torch
from colorama import Fore

from helpers import pbar


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
                                         shape=(self.database.num_queries, net.K, net.D))
            self.image_vlads = np.memmap(self.image_filename, dtype="float32", mode="w+",
                                         shape=(self.database.num_images, net.K, net.D))

        net.freeze()

        n_images = self.database.num_images
        n_queries = self.database.num_queries
        total = n_queries + n_images

        with torch.no_grad():
            t = pbar(total=total, desc="Building the cache", color=Fore.MAGENTA) if not t_parent else None

            for query_id in range(0, n_queries, cache_batch_size):
                q_tensors = self.database.query_tensor_batch(query_id, batch_size=cache_batch_size)
                q_vlads = net(q_tensors).detach().numpy()
                self.query_vlads[query_id:query_id + q_vlads.shape[0]] = q_vlads
                if t_parent:
                    t_parent.set_postfix(caching=f"{query_id * 100 // total}%")
                else:
                    t.update(q_vlads.shape[0])

            for image_id in range(0, n_images, cache_batch_size):
                i_tensors = self.database.image_tensor_batch(image_id, batch_size=cache_batch_size)
                i_vlads = net(i_tensors).detach().numpy()
                self.image_vlads[image_id:image_id + i_vlads.shape[0]] = i_vlads
                if t_parent:
                    t_parent.set_postfix(caching=f"{(n_queries + image_id) * 100 // total}%")
                else:
                    t.update(i_vlads.shape[0])

            if not t_parent:
                t.close()

            net.unfreeze()
            self.query_vlads.flush()
            self.image_vlads.flush()

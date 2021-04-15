import os.path as path
from tempfile import mkdtemp
import numpy as np
import torch
from colorama import Fore

from helpers import pbar, get_device

device = get_device()


class Cache:
    def __init__(self, database, cache_mode='ram'):
        self.database = database
        self.net = None
        self.query_filename = path.join(mkdtemp(), "query_vlads.dat")
        self.image_filename = path.join(mkdtemp(), "image_vlads.dat")
        self.query_vlads = None
        self.image_vlads = None
        self.cache_mode = cache_mode

    def update(self, net, t_parent=None, cache_batch_size=42):

        net.freeze()
        if not self.net:
            self.net = net

            if self.cache_mode == 'ram':
                self.query_vlads = np.zeros(dtype="float32", shape=(self.database.num_queries, net.K * net.D))
                self.image_vlads = np.zeros(dtype="float32", shape=(self.database.num_images, net.K * net.D))
            else:
                self.query_vlads = np.memmap(self.query_filename, dtype="float32", mode="w+",
                                             shape=(self.database.num_queries, net.K * net.D))
                self.image_vlads = np.memmap(self.image_filename, dtype="float32", mode="w+",
                                             shape=(self.database.num_images, net.K * net.D))

        n_images = self.database.num_images
        n_queries = self.database.num_queries
        total = n_queries + n_images

        with torch.no_grad():
            t = pbar(total=total, desc="Building the cache", color=Fore.MAGENTA) if not t_parent else None

            for query_id in range(0, n_queries, cache_batch_size):
                q_tensors = self.database.get_query_tensor(query_id, batch_size=cache_batch_size)
                q_vlads = net(q_tensors).cpu().detach().numpy()
                self.query_vlads[query_id:query_id + q_vlads.shape[0]] = q_vlads
                if t_parent:
                    t_parent.set_postfix(caching=f"{query_id * 100 // total}%")
                else:
                    t.update(q_vlads.shape[0])

            for image_id in range(0, n_images, cache_batch_size):
                i_tensors = self.database.get_image_tensor(image_id, batch_size=cache_batch_size)
                i_vlads = net(i_tensors).cpu().detach().numpy()
                self.image_vlads[image_id:image_id + i_vlads.shape[0]] = i_vlads
                if t_parent:
                    t_parent.set_postfix(caching=f"{(n_queries + image_id) * 100 // total}%")
                else:
                    t.update(i_vlads.shape[0])

            net.unfreeze()
            if not t_parent:
                t.close()

            if self.cache_mode == 'disk':
                self.query_vlads.flush()
                self.image_vlads.flush()

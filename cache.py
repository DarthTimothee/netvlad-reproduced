import os.path as path
from tempfile import mkdtemp

import numpy as np
import torch
from colorama import Fore
from tqdm.auto import trange


class Cache:
    def __init__(self, database):
        self.database = database
        self.net = None
        self.query_filename = path.join(mkdtemp(), "query_vlads.dat")
        self.image_filename = path.join(mkdtemp(), "image_vlads.dat")
        self.query_vlads = None
        self.image_vlads = None

    def update(self, net):
        if not self.net:
            self.net = net

            self.query_vlads = np.memmap(self.query_filename, dtype="float32", mode="w+",
                                         shape=(self.database.num_queries, net.D, net.K, 1))
            self.image_vlads = np.memmap(self.image_filename, dtype="float32", mode="w+",
                                         shape=(self.database.num_images, net.D, net.K, 1))

        net.freeze()
        net.eval()
        # TODO: make cache in batches
        with torch.no_grad():

            with trange(self.database.num_queries + self.database.num_images, position=0,
                        leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.MAGENTA)) as t:
                t.set_description("Building the cache\t\t\t")
                for i in t:
                    if i < self.database.num_queries:
                        query_id = i
                        query_tensor = self.database.query_to_tensor(query_id).unsqueeze(0)
                        self.query_vlads[query_id] = net(query_tensor).detach().numpy()
                    else:
                        image_id = i - self.database.num_queries
                        query_tensor = self.database.image_to_tensor(image_id).unsqueeze(0)
                        self.image_vlads[image_id] = net(query_tensor).detach().numpy()
        net.unfreeze()
        net.train()

        self.query_vlads.flush()
        self.image_vlads.flush()

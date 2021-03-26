import torch
from tqdm.auto import trange
from colorama import Fore


class Cache:
    def __init__(self, database, net):
        self.database = database
        self.net = net
        # TODO: use memmap or something
        self.query_vlads = [0] * database.num_queries
        self.image_vlads = [0] * database.num_images

    def update(self):
        print("Building the cache", flush=True)
        self.net.freeze()
        self.net.eval()
        with torch.no_grad():
            for i in trange(self.database.num_queries + self.database.num_images, position=0,
                            leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
                if i < self.database.num_queries:
                    query_id = i
                    query_tensor = self.database.query_to_tensor(query_id).unsqueeze(0)
                    self.query_vlads[query_id] = self.net(query_tensor).detach().numpy()
                else:
                    image_id = i - self.database.num_queries
                    query_tensor = self.database.image_to_tensor(image_id).unsqueeze(0)
                    self.image_vlads[image_id] = self.net(query_tensor).detach().numpy()
        self.net.unfreeze()
        self.net.train()

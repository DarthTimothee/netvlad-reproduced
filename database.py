import math
import scipy.io
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from cache import Cache


class Database:
    def __init__(self, net, database_url, dataset_url='G:/School/Deep Learning/data/'):
        self.dataset_url = dataset_url
        self.db = scipy.io.loadmat(database_url)
        self.num_images = self.db.get('dbStruct')[0][0][5][0][0]
        self.num_queries = self.db.get('dbStruct')[0][0][6][0][0]
        # self.num_queries = 100
        # self.num_images = 1100
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(100),  # TODO: resize/crop?
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # TODO: keep these normalize constants?
        ])
        self.cache = Cache(self, net)

    def update_cache(self):
        self.cache.update()

    def geo_distance(self, query_id, image_id):
        x1, y1 = self.query_position(query_id)
        x2, y2 = self.image_position(image_id)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def query_to_tensor(self, query_id):
        input_image = Image.open(self.dataset_url + self.query_name(query_id))
        return self.preprocess(input_image)

    def image_to_tensor(self, image_id):
        input_image = Image.open(self.dataset_url + self.image_name(image_id))
        return self.preprocess(input_image)

    def query_position(self, query_id):
        x = self.db.get('dbStruct')[0][0][4][0][query_id]
        y = self.db.get('dbStruct')[0][0][4][1][query_id]
        return x, y

    def image_position(self, image_id):
        x = self.db.get('dbStruct')[0][0][2][0][image_id]
        y = self.db.get('dbStruct')[0][0][2][1][image_id]
        return x, y

    def query_name(self, query_id):
        return self.db.get('dbStruct')[0][0][3][query_id][0][0]

    def image_name(self, image_id):
        return self.db.get('dbStruct')[0][0][1][image_id][0][0]

import math
import scipy.io
from torchvision import transforms
from PIL import Image

from cache import Cache


class Database:
    def __init__(self, database_url, net):
        self.db = scipy.io.loadmat(database_url)
        self.num_images = self.db.get('dbStruct')[0][0][5][0][0]
        self.num_queries = self.db.get('dbStruct')[0][0][6][0][0]
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            # transforms.Resize(256),  # TODO: resize/crop?
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # TODO: keep these normalize constants?
        ])
        self.query_cache = Cache(self, net)
        self.image_cache = Cache(self, net)

    def geo_distance(self, query_id, image_id):
        x1, y1 = self.query_position(query_id)
        x2, y2 = self.image_position(image_id)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def query_to_tensor(self, query_id):
        input_image = Image.open('G:/School/Deep Learning/data/' + self.query_name(query_id))
        return self.preprocess(input_image)

    def image_to_tensor(self, image_id):
        input_image = Image.open('G:/School/Deep Learning/data/' + self.image_name(image_id))
        return self.preprocess(input_image)

    def query_position(self, query_index):
        x = self.db.get('dbStruct')[0][0][4][0][query_index]
        y = self.db.get('dbStruct')[0][0][4][1][query_index]
        return x, y

    def image_position(self, image_index):
        x = self.db.get('dbStruct')[0][0][2][0][image_index]
        y = self.db.get('dbStruct')[0][0][2][1][image_index]
        return x, y

    def query_name(self, query_id):
        return self.db.get('dbStruct')[0][0][3][query_id][0][0]

    def image_name(self, image_id):
        return self.db.get('dbStruct')[0][0][1][image_id][0][0]

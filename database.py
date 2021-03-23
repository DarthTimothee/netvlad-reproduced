import math
import scipy.io
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from TrainingTuple import TrainingTuple


class Vlataset(Dataset):
    def __init__(self, database_url='./datasets/pitts250k_train.mat'):
        self.db = scipy.io.loadmat(database_url)
        self.num_images = self.db.get('dbStruct')[0][0][5][0][0]
        self.num_queries = self.db.get('dbStruct')[0][0][6][0][0]
        self.training_tuples = [TrainingTuple(i) for i in range(self.num_queries)]  # TODO
        # TODO: load all data from the .mat into variables
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            # transforms.Resize(256),  # TODO: resize/crop?
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # TODO: keep these normalize constants?
        ])

    def get_image_name(self, image_id):
        return self.db.get('dbStruct')[0][0][1][image_id][0][0]

    def get_position(self, image_id):
        x = self.db.get('dbStruct')[0][0][2][0][image_id]
        y = self.db.get('dbStruct')[0][0][2][1][image_id]
        return x, y

    def geo_distance(self, q1, q2):
        x1, y1 = self.get_position(q1)
        x2, y2 = self.get_position(q2)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_tensor(self, query):
        input_image = Image.open('G:/School/Deep Learning/data/' + self.get_image_name(query))
        return self.preprocess(input_image)

    def __best_positive(self, training_tuple):
        best_positive = None
        return best_positive

    def __getitem__(self, index):
        training_tuple = self.training_tuples[index]
        query = training_tuple.query_id
        best_positive = training_tuple.get_best_positive()
        hard_negatives = training_tuple.get_hard_negatives(self.num_images)
        asdf = self.get_tensor(best_positive)
        fdsa = [self.get_tensor(n) for n in hard_negatives]
        return self.get_tensor(query), query, asdf, fdsa

    def __len__(self):
        return len(self.training_tuples)

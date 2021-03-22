import math
import scipy.io
from torch.utils.data import Dataset

from TrainingTuple import TrainingTuple


class Vlataset(Dataset):
    def __init__(self, database_url='./datasets/pitts250k_train.mat'):
        self.db = scipy.io.loadmat(database_url)
        self.training_tuples = [TrainingTuple(i) for i in range(10)]  # TODO
        # TODO: load all data from the .mat into variables

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

    def __getitem__(self, index):
        return self.training_tuples[index]

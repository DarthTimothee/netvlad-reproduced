import os.path as path
import time
import scipy.io
from PIL import Image
from torchvision import transforms
from tqdm import trange

DATA = '../pittsburgh250k'
DATABASE = 'datasets/pitts250k_val.mat'

db = scipy.io.loadmat(path.join(DATA, DATABASE))

num_images = db.get('dbStruct')[0][0][5][0][0]
image_filenames = [""] * num_images
for image_id in range(num_images):
    f = db.get('dbStruct')[0][0][1][image_id][0][0]
    image_filenames[image_id] = f


def get_input_transform(whitening=False, input_scale=None):
    return transforms.Compose(list(filter(None, [
        transforms.Resize(size=input_scale) if input_scale else None,
        transforms.ToTensor(),
        None if whitening else None,  # TODO: PCA whitening

        # The mean and std are required by required by the alexnet / vgg16 base network
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if not whitening else None,
    ])))


def image_from_file(image_id):
    filename = path.join(DATA, image_filenames[image_id])
    input_image = Image.open(filename)
    return input_image


transform = get_input_transform()
load_time = 0
transform_time = 0

with trange(num_images) as t:
    for i in t:
        start = time.time()
        img = image_from_file(i)
        load_time += time.time() - start

        start = time.time()
        transform(img)
        transform_time += time.time() - start

        total = load_time + transform_time
        load = load_time / total * 100
        tt = transform_time / total * 100

        t.set_postfix(load=f"{load}%", transform=f"{tt}%")

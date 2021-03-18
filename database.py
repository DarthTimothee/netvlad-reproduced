import scipy.io


db = scipy.io.loadmat('./datasets/pitts250k_train.mat')


def get_image_name(image_id):
    return db.get('dbStruct')[0][0][1][image_id][0][0]


def get_position(image_id):
    x = db.get('dbStruct')[0][0][2][0][image_id]
    y = db.get('dbStruct')[0][0][2][1][image_id]
    return x, y


# Just some test code to make sure it works
print(get_image_name(5))
print(get_position(5))

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from NetVladCNN import NetVladCNN


def alex_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.features(x)


# Hyper parameters, based on the appendix
K = 64  # amount of kernels
m = 0.1  # margin for the loss
lr = 0.001  # or 0.0001 depending on the experiment, which is halved every 5 epochs
momentum = 0.9
wd = 0.001
batch_size = 4  # batch size is 4 tuples
max_epochs = 30  # but usually convergence occurs much faster


# Setup base network
alex_base = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
alex_base.features = alex_base.features[:11]  # cut off before last ReLu
alex_base.avgpool = nn.Identity()  # bypass avg_pool and classifier
alex_base.classifier = nn.Identity()  # bypass avg_pool and classifier
alex_base.forward = type(alex_base.forward)(alex_forward, alex_base)  # TODO: find a better way to override this function. Maybe by extending the AlexNet module?

# Setup our network
net = NetVladCNN(base_cnn=alex_base, K=K)

# Import an image
filename = "data/000000_pitch1_yaw1.jpg"
input_image = Image.open(filename)

# Show the image
# plt.figure()
# plt.title(f"example image: {filename}")
# plt.imshow(input_image)
# plt.show()

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    # transforms.Resize(256),  # TODO: resize/crop?
    # transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # TODO: keep these normalize constants?
])
input_tensor = preprocess(input_image)
x = input_tensor.unsqueeze(0)  # alex_net wants a batch!

D = 256
c = np.zeros((K, D))  # TODO: get actual c (parameter) used zeros for now

y = net.forward(x, c)
print(f"y.size() = {y.size()}")

for i in range(10):
    print(net.forward(x, c))

import torch
import torchvision
import cv2
import time
import numpy as np
from torch.autograd import Variable
from torchvision import models

model = torchvision.models.vgg16(pretrained=True)
device = torch.device("cpu")
model = model.to(device)
print(model)

img = cv2.imread("data/faces/0805personali01.jpg")
img = cv2.resize(img, (224, 224))
img = img / float(255)
img = img.transpose(2, 0, 1)


data = torch.from_numpy(img[np.newaxis, :, :, :])
data = Variable(data.float().cpu())

model.eval()
t0 = time.time()
output = model(data)
#print(output)
output_numpy = (output.data).cpu().numpy()
sorted = np.argsort(output_numpy)[0][::-1]
print(sorted)
t1 = time.time()

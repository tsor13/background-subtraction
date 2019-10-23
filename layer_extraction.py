import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from scipy.ndimage import median_filter

import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import cv2
import time

# create vgg class
import torchvision.models as models
class Normalization(nn.Module):
  # def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]).cuda(), std=torch.tensor([0.229, 0.224, 0.225]).cuda()):
  def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
      super(Normalization, self).__init__()
      self.mean = torch.tensor(mean).view(-1, 1, 1)
      self.std = torch.tensor(std).view(-1, 1, 1)

  def forward(self, img):
      return (img - self.mean) / self.std

class VGGIntermediate(nn.Module):
  def __init__(self, requested=[]):
    super(VGGIntermediate, self).__init__()
    self.norm = Normalization().eval()
    self.intermediates = {}
    self.vgg = models.vgg19(pretrained=True).features.eval()
    for i, m in enumerate(self.vgg.children()):
        if isinstance(m, nn.ReLU):   # we want to set the relu layers to NOT do the relu in place.
          m.inplace = False          # the model has a hard time going backwards on the in place functions.

        if i in requested:
          def curry(i):
            def hook(module, input, output):
              self.intermediates[i] = output
            return hook
          m.register_forward_hook(curry(i))

  def forward(self, x):
    self.vgg(self.norm(x))
    return self.intermediates

class BackgroundLoss(nn.Module):
  def __init__(self, weights_ideal):
    super(BackgroundLoss, self).__init__()
    self.weights_ideal = weights_ideal
    # self.weights_ideal.cuda()

  def forward(self, weights_hat):
    diff = self.weights_ideal - weights_hat
    s = (diff * diff).sum() * .5
    return s

# layers to extract
layers = [5]
vgg = VGGIntermediate(layers).eval()


# read in video
vid = sys.argv[1]
cap = cv2.VideoCapture(vid)

dim = 400

load_and_normalize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((dim,dim)),
    transforms.ToTensor(),
])

# train background
num_background = 20
background_model = None
for n in range(num_background):
    _, frame = cap.read()
    cv2.imshow('frame', frame)
    frame_tensor = load_and_normalize(frame).unsqueeze(0)
    start = time.time()
    val = vgg(frame_tensor)
    elapsed = time.time()-start
    print(elapsed)
    if n == 0:
        background_model = val[5]
    else:
        background_model = (n*background_model + val[5]) / (n+1)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    del val

b_loss = BackgroundLoss(background_model)
learning_rate = 1e-3

for n in range(num_background):
    _, frame = cap.read()
    cv2.imshow('frame', frame)
    frame_tensor = load_and_normalize(frame).unsqueeze(0)
    optimizer = optim.Adam([frame_tensor.requires_grad_()], lr=learning_rate)
    val = vgg(frame_tensor)
    loss = b_loss(val[5])
    loss.backward()
    # gradient = F.softmax(torch.abs(frame_tensor.grad).squeeze(0).mean(dim=0).reshape(-1),dim=0).reshape(400,-1)
    gradient = torch.abs(frame_tensor.grad).squeeze(0).mean(dim=0).reshape(400,-1)
    # gradient *= 1/gradient.max()
    # gradient *= 255
    mask = gradient.data.numpy()
    cutoff = np.percentile(mask, 95)
    mask = mask>cutoff
    mask = mask.astype('uint8')
    mask *= 255
    mask = cv2.resize(mask, dsize= (1920,1080))
    # do median filter
    pdb.set_trace()
    mask = median_filter(mask, size=15)
    cv2.imshow('gradient', mask)
    mask_on_frame = frame // 2 + np.dstack([mask // 2] * 3)
    cv2.imshow('mask', mask_on_frame)
    key = cv2.waitKey(1)
    optimizer.zero_grad()
    if key == ord('q'):
        break
    del val

cap.release()
cv2.destroyAllWindows

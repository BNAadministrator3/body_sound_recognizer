import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from skimage.feature import hog
from skimage import data, exposure
import numpy as np

from readdata_bowel import DataSpeech
datapath = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect'
traindata = DataSpeech(datapath, 'train')

oimage,_ = traindata.GetData(10, mode='non-repetitive')
image = np.squeeze(oimage)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
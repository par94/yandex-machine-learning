import pandas as pd
import numpy as np
import math
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import KMeans
from skimage.io import imread, imsave
from skimage import img_as_float
import pylab

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
image = imread(os.path.join(__location__,'parrots.jpg'))
float_image = img_as_float(image)
w, h, d = float_image.shape
pixels = pd.DataFrame(np.reshape(float_image, (w*h, d)), columns=['R', 'G', 'B'])
def cluster(pixels, n_clusters=8):
    print('Clustering: ' + str(n_clusters))

    pixels = pixels.copy()
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    pixels['cluster'] = model.fit_predict(pixels)

    means = pixels.groupby('cluster').mean().values
    mean_pixels = [means[c] for c in pixels['cluster'].values]
    mean_image = np.reshape(mean_pixels, (w, h, d))
    imsave('mean_parrots_' + str(n_clusters) + '.jpg', mean_image)

    medians = pixels.groupby('cluster').median().values
    median_pixels = [medians[c] for c in pixels['cluster'].values]
    median_image = np.reshape(median_pixels, (w, h, d))
    imsave('median_parrots_' + str(n_clusters) + '.jpg', median_image)

    return mean_image, median_image

def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return 10 * math.log10(float(1) / mse)

for n in range(1, 21):
    mean_image, median_image = cluster(pixels, n)
    psnr_mean, psnr_median = psnr(float_image, mean_image), psnr(float_image, median_image)
    print(psnr_mean)
    print(psnr_median)

    if psnr_mean > 20 or psnr_median > 20:
        print(n)
        break





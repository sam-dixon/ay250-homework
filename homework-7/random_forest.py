import os
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks, blob_dog, blob_doh
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

DIR = './50_categories/'


def all_image_paths(dir=DIR):
    image_paths = []
    for cat in os.listdir(dir)[:10]:
        cat_path = os.path.join(dir, cat)
        for img in os.listdir(cat_path):
            image_paths.append(os.path.join(cat_path, img))
    return image_paths


def label_from_path(path):
    return path.split('/')[-2]


def category_paths(category, dir=DIR):
    image_paths = all_image_paths(dir)
    return list(filter(lambda x: category == label_from_path(x), image_paths))


def img_size(img):
    """Get the size of the image array in megapixels"""
    return np.prod(img.shape[:2])/1e6


def aspect_ratio(img):
    """Calculate aspect ratio M/N from (M x N x 3) RGB image array"""
    return img.shape[0]/img.shape[1]


def color_stats(img, color):
    """Calculate the average and std dev pixel value in color r, g, b, or k for an image array"""
    if color.lower() == 'k' or len(img.shape) < 3:
        img = rgb2gray(img)
        ave = np.mean(img)
        std = np.std(img)
        ran = np.max(img) - np.min(img)
        return ave, std, ran
    color_index = ['r', 'g', 'b'].index(color.lower())
    ave = np.mean(img[:, :, color_index])
    std = np.std(img[:, :, color_index])
    ran = np.max(img[:, :, color_index])-np.min(img[:, :, color_index])
    return ave, std, ran


def color_ratio(img, color1, color2):
    ave1, _, _ = color_stats(img, color1)
    ave2, _, _ = color_stats(img, color2)
    return ave1/ave2


def color_diff(img, color1, color2):
    ave1, _, _ = color_stats(img, color1)
    ave2, _, _ = color_stats(img, color2)
    return ave1-ave2


def min_max_color_position(img, color):
    """Find the x, y location of the maximum pixel value of the given color"""
    if color.lower() == 'k' or len(img.shape) < 3:
        img = rgb2gray(img)
        min_y, min_x = np.unravel_index(np.argmin(img), img.shape)
        max_y, max_x = np.unravel_index(np.argmax(img), img.shape)
        return min_x, min_y, max_x, max_y
    color_index = ['r', 'g', 'b'].index(color.lower())
    min_y, min_x = np.unravel_index(np.argmin(img[:, :, color_index]), img[:, :, color_index].shape)
    max_y, max_x = np.unravel_index(np.argmax(img[:, :, color_index]), img[:, :, color_index].shape)
    return min_x/img.shape[1], min_y/img.shape[0], max_x/img.shape[1], max_y/img.shape[0]


def min_max_distance(img, color):
    minx, miny, maxx, maxy = min_max_color_position(img, color)
    return np.sqrt((maxx-minx)**2+(maxy-miny)**2)


def num_bw(img, thresh=0.01):
    """Count the fraction of black/white pixels (defined as grayscale pixels below thresh/above 1-thresh)"""
    imgbw = rgb2gray(img)
    k = len(list(zip(*np.where(imgbw < thresh))))/img_size(img)
    w = len(list(zip(*np.where(imgbw > (1-thresh)))))/img_size(img)
    return k, w


def num_corners(img):
    """Calculate the number of corner points using Harris corner detection"""
    if img.shape[-1] == 3:
        img = rgb2gray(img)
    n = len(corner_peaks(corner_harris(img)))
    # normalize by image size
    return n/img_size(img)


def num_blobs(img):
    """
    Calculate the number of blobs in an image using the determinant of
    Hessian and difference of Gaussians
    """
    if img.shape[-1] == 3:
        img = rgb2gray(img)
    n_blob_dog = len(blob_dog(img))
    n_blob_doh = len(blob_doh(img))
    blob_diff = n_blob_doh-n_blob_dog
    return n_blob_doh, n_blob_dog, blob_diff


def split_seq(seq, size):
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
        newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq


def extract_features(image_path_list):
    feature_list = []
    for image_path in tqdm(image_path_list):
        img = imread(image_path)
        size = img_size(img)
        asp = aspect_ratio(img)
        r_ave, r_std, r_ran = color_stats(img, 'r')
        g_ave, g_std, g_ran = color_stats(img, 'g')
        b_ave, b_std, b_ran = color_stats(img, 'b')
        k_ave, k_std, k_ran = color_stats(img, 'k')
        rg = color_ratio(img, 'r', 'g')
        gb = color_ratio(img, 'g', 'b')
        br = color_ratio(img, 'b', 'r')
        drg = color_diff(img, 'r', 'g')
        dgb = color_diff(img, 'g', 'b')
        dbr = color_diff(img, 'b', 'r')
        b, w = num_bw(img)
        rminx, rminy, rmaxx, rmaxy = min_max_color_position(img, 'r')
        gminx, gminy, gmaxx, gmaxy = min_max_color_position(img, 'g')
        bminx, bminy, bmaxx, bmaxy = min_max_color_position(img, 'b')
        kminx, kminy, kmaxx, kmaxy = min_max_color_position(img, 'k')
        rdist = min_max_distance(img, 'r')
        gdist = min_max_distance(img, 'g')
        bdist = min_max_distance(img, 'b')
        kdist = min_max_distance(img, 'k')
        n_corners = num_corners(img)
        n_blob_doh, n_blob_dog, blob_diff = num_blobs(img)
        if n_corners == 0:
            blob_per_corner = -1.
        else:
            blob_per_corner = np.mean(np.array([n_blob_doh, n_blob_dog]))/n_corners
        feature = [label_from_path(image_path),
                   size,
                   asp,
                   r_ave, r_std,
                   g_ave, g_std,
                   b_ave, b_std,
                   k_ave, k_std,
                   rg, gb, br,
                   drg, dgb, dbr,
                   b, w,
                   rminx, rmaxx,
                   gminx, gmaxx,
                   bminx, bmaxx,
                   kminx, kmaxx,
                   rdist, gdist, bdist, kdist,
                   n_corners,
                   n_blob_doh, n_blob_dog, blob_diff,
                   blob_per_corner]
        feature_list.append(feature)
    return feature_list


def extract_parallel():
    """
    Extract all of the features from all of the images.
    Takes about 20 minutes to run on a 1.3 GHz 2-core Macbook Air.
    Most of the time is spent detecting blobs.
    """
    image_paths = all_image_paths()
    numprocessors = cpu_count()
    split_image_paths = split_seq(image_paths, numprocessors)
    p = Pool(numprocessors)
    result = p.map_async(extract_features, split_image_paths)
    poolresult = result.get()
    combined_result = []
    for single_proc_result in poolresult:
        for single_image_result in single_proc_result:
            combined_result.append(single_image_result)
    pickle.dump(combined_result, open('features.pkl', 'wb'))


def train_test_split(features, n=45):
    """
    Since there are different numbers of images in each category, we want
    to ensure that we aren't overtraining on categories with more images.
    To do this, we randomly select 30 images from each category to go into the
    training set. The remaining images are the validation set.

    45 is a good number, since the smallest categories have 53 images, so this
    is an approximate 90/10 split.
    """
    counts = {}
    i = 0
    train, test = [], []
    rand_ind = np.arange(0, len(features))
    np.random.shuffle(rand_ind)
    for i in rand_ind:
        f = features[i]
        if len(f) == 33:
            print(len(f), f)
        try:
            counts[f[0]] += 0
        except KeyError:
            counts[f[0]] = 0
        if counts[f[0]] < n:
            train.append(f)
            counts[f[0]] += 1
        else:
            test.append(f)
            counts[f[0]] += 1
    return np.array(train), np.array(test), counts


def random_forest(training):
    """
    Train a random forest classifier.
    """
    features = training[:, 1:].astype(np.float64)
    features[~np.isfinite(features)] = -1.
    labels = training[:, 0]
    clf = RandomForestClassifier(n_estimators=1000, verbose=1, n_jobs=-1)
    clf.fit(features, labels)
    pickle.dump(clf, open('random_forest.pkl', 'wb'))


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap='viridis'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=6)
    plt.yticks(tick_marks, classes, size=6)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    if not os.path.isfile('features.pkl'):
        extract_parallel()
    features = pickle.load(open('features.pkl', 'rb'))
    n = 40
    train, test, counts = train_test_split(features, n=n)
    if not os.path.isfile('random_forest.pkl'):
        random_forest(train)
    clf = pickle.load(open('random_forest.pkl', 'rb'))
    weights = np.array([1./(counts[t[0]]-n) for t in test])
    cm = confusion_matrix(test[:, 0], clf.predict(test[:, 1:]), sample_weight=weights)
    classes = np.unique(test[:, 0])
    plot_confusion_matrix(cm, classes)
    plt.show()

from turtle import color
import numpy as np
import scipy.io as scio
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt


def get_sim():
    iou_threshold_dict = {
        'Car': 0.7,
        'Bus': 0.7,
        'Truck': 0.7,
        'Pedestrian': 0.3,
        'Cyclist': 0.5
    }
    data = scio.loadmat('results_train.mat')
    features = data['features']
    score = data['score'][0]

    mask = score >= 0
    iou = data['iou'][0][mask]
    name = data['name'][mask]
    name = np.array([n.strip() for n in name])
    labels = data['labels'][0][mask]
    boxes = data['boxes_3d'][mask]
    dist = np.linalg.norm(boxes[:, :2], axis=1)

    val_data = scio.loadmat('results_val.mat')
    val_features = val_data['features']
    val_score = val_data['score'][0]

    mask = val_score >= 0
    val_iou = val_data['iou'][0][mask]
    val_name = val_data['name'][mask]
    val_name = np.array([n.strip() for n in val_name])
    val_labels = val_data['labels'][0][mask]
    val_boxes = val_data['boxes_3d'][mask]
    val_dist = np.linalg.norm(val_boxes[:, :2], axis=1)

    norm_features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
    norm_val_features = val_features / np.linalg.norm(val_features, axis=1)[:, np.newaxis]
    sim = np.dot(norm_val_features, norm_features.T)
    return iou, name, dist, boxes, val_iou, val_name, val_dist, val_boxes, sim

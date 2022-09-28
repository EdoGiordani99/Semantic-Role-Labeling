import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from sklearn.metrics import f1_score
from scipy.interpolate import make_interp_spline


def compute_f1(labels, preds):
    """
    Given training labels and predictions, this function removes pads and
    unnecessary predictions for bert sub - tokens.
    Args:
      labels (torch.Tensor): y batch
      preds (torch.Tensor): prediction for the batch
    """

    all_preds = []
    all_labels = []

    labels, preds = clean_labels_and_preds(labels, preds)

    for pred, label in zip(preds, labels):
        all_preds += pred
        all_labels += label

    f1 = f1_score(all_labels, all_preds, average='micro')

    return round(f1, 4)


def clean_labels_and_preds(labels, preds):
    """
    Given training labels and predictions, this function removes pads and
    unnecessary predictions for bert sub - tokens.
    Args:
      labels (torch.Tensor): y batch
      preds (torch.Tensor): prediction for the batch
    Return:
      new_preds (list): cleaned labels
      new_labels (list): cleaned labels
    """
    new_preds, new_labels = [], []

    for label, pred in zip(labels, preds):
        new_l, new_p = [], []

        for i, lab in enumerate(label):

            if lab != -100:
                new_l.append(int(lab))
                new_p.append(int(pred[i]))

        new_preds.append(new_p)
        new_labels.append(new_l)

    return new_labels, new_preds


def plot_histories(histories,
                   title: str = None,
                   smooth: bool = True,
                   smooth_factor: int = 3,
                   y_min: float = 0.0,
                   y_max: float = 1.0,
                   x_max: int = None):
    """
    Plot history function!
    Args:
        histories (Dict): dictionary containing 'name_line' : [list of numbers]
        title (str): the title of the graph
        smooth (bool): if True then the curve is smoothed
        smooth_factor (ODD int): odd number that regulates the smoothing
        range_min (float): minimum y value in the graph
        range_max (float): maximum y value in the graph
        x_max (int or None): maximum x value in the graph

    """
    names = []
    for name, hist in histories.items():

        names.append(name)

        if smooth:
            x = np.array([i for i in range(len(hist))])
            y = np.array(hist)

            x_new = np.linspace(x.min(), x.max(), 2 * len(y))

            spl = make_interp_spline(x, y, k=smooth_factor)
            y_new = spl(x_new)

            plt.plot(x_new, y_new)
            plt.legend(names)
            plt.ylim((y_min, y_max))
            if x_max is not None:
                plt.xlim((0, x_max))
            if title:
                plt.title(title)

        else:
            x = np.array([i for i in range(len(hist))])
            y = np.array(hist)

            plt.plot(x, y)
            plt.legend(names)
            plt.ylim((y_min, y_max))
            if x_max is not None:
                plt.xlim((0, x_max))
            if title:
                plt.title(title)

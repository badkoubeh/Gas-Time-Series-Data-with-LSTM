import warnings
import torch
from datetime import datetime, timedelta
import copy
import numpy as np
import pandas as pd
import seaborn as sns
# import watermark as watermark
from pylab import rcParams
import matplotlib.pyplot as plt
# from matplotlib import rc
from sklearn.model_selection import train_test_split
import pickle
from torch import nn, optim

from training import RecurrentAutoencoder, Encoder, Decoder

warnings.simplefilter('ignore')

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


NEW_PREDICTIONS = True
THRESHOLD = 0.5


def main():
    with open('train_variables_30days.pkl', 'rb') as f:
        n_train_days, history, train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset = pickle.load(f)

    model = torch.load('model_{0}days.pth'.format(n_train_days), map_location=lambda storage, loc: storage)

    if not NEW_PREDICTIONS:
        with open('predictions_{0}days.pkl'.format(n_train_days), 'rb') as f:
            train_losses, normalset_pred_losses, anomaly_pred_losses, anomaly_predictions = pickle.load(f)

        correct = sum(l <= THRESHOLD for l in normalset_pred_losses)
        print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')
        anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
        correct = sum(l > THRESHOLD for l in anomaly_pred_losses)
        print(f'Correct normal predictions: {correct}/{len(anomaly_dataset)}')
    else:
        _, train_losses = predict(model, train_dataset)

        _, normalset_pred_losses = predict(model, test_normal_dataset)

        correct = sum(l <= THRESHOLD for l in normalset_pred_losses)
        print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')

        anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
        anomaly_predictions, anomaly_pred_losses = predict(model, anomaly_dataset)
        correct = sum(l > THRESHOLD for l in anomaly_pred_losses)
        print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')

        with open('predictions_{0}days.pkl'.format(n_train_days), 'wb') as f:
            pickle.dump([train_losses, normalset_pred_losses, anomaly_pred_losses, anomaly_predictions], f)

    # ax = plt.figure(1).gca()
    # ax.plot(history['train'])
    # ax.plot(history['val'])
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['train', 'test'])
    # plt.title('Loss over training epochs')

    plt.figure(2, figsize=(16, 9), dpi=80)
    plt.title('Training Loss Distribution', fontsize=16)
    sns.distplot(train_losses, bins=50, kde=True)
    plt.xlim([0.0, 1])

    plt.figure(3, figsize=(16, 9), dpi=80)
    plt.title('Normal Loss Distribution', fontsize=16)
    sns.distplot(normalset_pred_losses, bins=50, kde=True)

    plt.figure(4, figsize=(16, 9), dpi=80)
    plt.title('Anomaly Loss Distribution', fontsize=16)
    sns.distplot(anomaly_pred_losses, bins=50, kde=True)

    plt.show()


if __name__ == '__main__':
    # input = sys.argv[1]
    main()

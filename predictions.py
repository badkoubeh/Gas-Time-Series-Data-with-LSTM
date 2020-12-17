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

from autoencoder_LSTM_training import RecurrentAutoencoder, Encoder, Decoder, create_dataset

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


seq_length = 4


# def forecast(model, X_test):
#     with torch.no_grad():
#         test_seq = X_test[:1]
#         print(test_seq)
#         preds = []
#         for _ in range(len(X_test)):
#             y_test_pred = model(test_seq)
#             pred = torch.flatten(y_test_pred).item()
#             preds.append(pred)
#             new_seq = test_seq.numpy().flatten()
#             new_seq = np.append(new_seq, [pred])
#             new_seq = new_seq[1:]
#             test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()


THRESHOLD = 0.07


def main():
    ts = pd.read_parquet('timeseries_resampled.parquet')


    with open('model_vars_2020-10-01.pkl', 'rb') as f:
        train_end_date, history, train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset = pickle.load(f)

    # model = torch.load('model_{0}.pth'.format(train_end_date), map_location=lambda storage, loc: storage)
    model = torch.load('model_2020-10-01.pth', map_location=lambda storage, loc: storage)

    # train_end_date = datetime(2020, 10, 1)
    # # test a new dataset
    # ts = ts[train_end_date:]
    # # ts = ts[datetime(2019, 4, 1): datetime(2019, 4, 10)]
    # test_normal_ts = ts[ts['target'] == 'Normal'].drop(labels='target', axis=1)
    # test_anomaly_ts = ts[ts['target'] != 'Normal'].drop(labels='target', axis=1)
    #
    # test_normal_dataset, _, _ = create_dataset(test_normal_ts)
    # test_anomaly_dataset, _, _ = create_dataset(test_anomaly_ts)

    _, train_losses = predict(model, train_dataset)

    _, normalset_pred_losses = predict(model, test_normal_dataset)

    correct = sum(l <= THRESHOLD for l in normalset_pred_losses)
    print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')

    anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
    anomaly_predictions, anomaly_pred_losses = predict(model, anomaly_dataset)
    correct = sum(l > THRESHOLD for l in anomaly_pred_losses)
    print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')

    ax = plt.figure(1).gca()
    ax.plot(history['train'])
    ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.title('Loss over training epochs')

    plt.figure(2, figsize=(16, 9), dpi=80)
    plt.title('Training Loss Distribution', fontsize=16)
    sns.distplot(train_losses, bins=50, kde=True)
    plt.xlim([0.0, 5.0])
    #
    plt.figure(3, figsize=(16, 9), dpi=80)
    plt.title('Normal Loss Distribution', fontsize=16)
    sns.distplot(normalset_pred_losses, bins=50, kde=True)
    # sns.histplot(normalset_pred_losses, bins=100, kde=True, stat='density')
    plt.xlim([0.0, 4.0])

    plt.figure(4, figsize=(16, 9), dpi=80)
    plt.title('Anomaly Loss Distribution', fontsize=16)
    sns.distplot(anomaly_pred_losses, kde=True)
    # sns.histplot(anomaly_pred_losses, binrange=(20, 100), kde=True, stat='density')
    plt.xlim([0.0, 100.0])

    plt.show()


if __name__ == '__main__':
    # input = sys.argv[1]
    main()

import pickle

import torch
from datetime import datetime, timedelta
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torch import nn, optim

### configs
n_train_days = 5
ts_train_range = datetime(2019, 1, 1) + timedelta(days=n_train_days)
print('training for {0} days of sensor data: '.format(n_train_days))

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Reading sensor data and ETL process
ts_df = pd.read_parquet('sensor_data_ts', engine='pyarrow')

class_names = ['Normal', 'GasAlarm', 'GasHigh', 'GasLow', 'GasSTEL', 'GasTWA']

columns = list(ts_df.columns)
ts_df = ts_df.rename(columns={'message_code_name': 'target'})
# ts_df['target'] = ts_df['target'].astype('str')
ts_df.loc[ts_df['target'] == 'SensorAlarmMsg', 'target'] = class_names[1]
ts_df.loc[ts_df['target'] == 'GasHighAlarmMsg', 'target'] = class_names[2]
ts_df.loc[ts_df['target'] == 'GasLowAlarmMsg', 'target'] = class_names[3]
ts_df.loc[ts_df['target'] == 'GasSTELAlarmMsg', 'target'] = class_names[4]
ts_df.loc[ts_df['target'] == 'GasTWAAlarmMsg', 'target'] = class_names[5]

ts_df = ts_df.sort_values(by=['datetime'])

ts_df = ts_df.set_index(pd.DatetimeIndex(ts_df['datetime']), drop=True)

ts_df = ts_df[ts_df['datetime'] < ts_train_range]

# original data:  28 < latitude < 30 and  -99 < longitude < -98
# Narrow down to one operation site
ts_df = ts_df[ts_df['latitude'] < 28.7]
ts_df = ts_df[ts_df['longitude'] > -98.8]

ts_df = ts_df.drop(labels=['datetime', 'latitude', 'longitude', 'message_code_id'], axis=1)


## helper function: convert to tensor
def create_dataset(df):
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(1, n_epochs + 1):
        model = model.train()

        train_losses = []
        for seq_true in train_dataset:
            # optimizer.zero_grad()

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(type(train_losses))
        print(type(train_loss))
        print(type(val_loss))
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history


def main():
    normal_df = ts_df[ts_df['target'] == 'Normal'].drop(labels='target', axis=1)
    anomaly_df = ts_df[ts_df['target'] != 'Normal'].drop(labels='target', axis=1)

    print('feature columns: ', normal_df.columns)

    train_df, validation_df = train_test_split(normal_df, test_size=0.15, random_state=RANDOM_SEED)

    validation_df, test_df = train_test_split(validation_df, test_size=0.33, random_state=RANDOM_SEED)

    train_dataset, seq_len, n_features = create_dataset(train_df)
    val_dataset, _, _ = create_dataset(validation_df)
    test_normal_dataset, _, _ = create_dataset(test_df)
    test_anomaly_dataset, _, _ = create_dataset(anomaly_df)

    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model = model.to(device)

    model, history = train_model(
        model,
        train_dataset,
        val_dataset,
        n_epochs=10
    )

    ax = plt.figure().gca()

    ax.plot(history['train'])
    ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.savefig('loss_epochs.svg')

    MODEL_PATH = 'model_{0}days.pth'.format(n_train_days)

    torch.save(model, MODEL_PATH)

    with open('train_variables_{0}days.pkl'.format(n_train_days), 'wb') as f:
        pickle.dump([n_train_days, history, train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset], f)

    f.close()


if __name__ == '__main__':
    main()

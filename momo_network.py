from finance_models.torch_trade import TimeSeriesCNN, train_model
from finance_models.utils import clean_data
from datasets import ModelDataset
import yfinance as yf


le_f = yf.download('LE=F', multi_level_index=False)

train_index= round(0.8 * len(le_f))

training_data = le_f[:train_index]

testing_data = le_f[train_index:]

train_dataset = ModelDataset(training_data, 20, 5)
val_dataset = ModelDataset(testing_data, 20, 5)

train_dataset.seasonal_decomp((110, 49, 21))
val_dataset.seasonal_decomp((110, 49, 21))


train_dataset.seasonal_features()
train_dataset.trend_features(trend=True, SMAs=True, sma_lens=[100, 21], momentum=True, momentum_lens=[21, 63], normalize_features=False)
train_dataset.prepare_features(['Seasonal', 'Trend'])

val_dataset.seasonal_features()
val_dataset.trend_features(trend=True, SMAs=True, sma_lens=[100, 21], momentum=True, momentum_lens=[21, 63], normalize_features=False)
val_dataset.prepare_features(['Seasonal', 'Trend'])

input_size = len(train_dataset.feature_columns)
input_len = len(train_dataset[0][0])
output_size = len(train_dataset[0][1])

model = TimeSeriesCNN(input_channels=1,sequence_length=input_len,num_features=input_size, output_sequence_length=5)


train_model(model, train_dataset, val_dataset, date_index=0, output_size=output_size, learning_rate=0.003)
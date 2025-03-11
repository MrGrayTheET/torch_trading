import numpy as np
import torch
from statsmodels.tsa.seasonal import STL, MSTL, seasonal_decompose
from torch_trading.feature_engineering import (atr,
                                 tr,
                                 rvol,
                                 get_trading_days,
                                 vol_scaled_returns,
                                 calc_returns,
                                 calc_daily_vol)



VOL_THRESHOLD = 5


class ModelDataset:

    def __init__(self, data, input_window=10,output_window=5, project_dir='F:\\ML\\', vol_scale=True):

        self.processed_data = None
        self.data = data
        self.window_size = input_window
        self.output_window = output_window
        self.model_info = {
            'Seasonal': {'Features': []},
            'Trend': {'Features': []},
            'Volume': {'Features': []},
            'Volatility': {'Features': []},
            'Dir': project_dir,
            'Scaling': {'Vol': True},
            'ML': {},
            'data':{}
        }
        self.feature_columns = []
        self.data_map = {}

        if vol_scale:
            ewm = self.data['Close'].ewm(halflife=252)
            means = ewm.mean()
            stds = ewm.std()
            self.data['s_close'] = np.minimum(self.data['Close'], means + VOL_THRESHOLD * stds)
            self.data['s_close'] = np.maximum(self.data['s_close'], means - VOL_THRESHOLD * stds)
            self.data['daily_returns'] = calc_returns(self.data.Close)
            self.data['daily_vol'] = calc_daily_vol(self.data.daily_returns)
            self.data['target_returns'] = vol_scaled_returns(calc_returns(self.data.s_close), self.data.daily_vol).shift(-1)
            self.data['active_positions'] = np.zeros_like(len(self.data))


    def seasonal_decomp(self, periods, method="mstl", EMA=False, span=5, plot=False, model='multiplicative',
                        plot_model=False):

        if EMA:
            data = self.data.Close.ewm(span).mean()
        else:
            data = self.data.Close
        if method == 'mstl':
            self.decomp_model = MSTL(self.data.Close, periods=periods).fit()

        elif method == 'stl':
            self.decomp_model = STL(self.data.Close, period=periods).fit()
        else:
            self.decomp_model = seasonal_decompose(self.data.Close, model=model, period=periods)

        self.model_info['Seasonal'].update({'Periods': periods})

        if plot_model:
            self.plot_seasonal_model()

        return

    def seasonal_features(self, seasonals=True, residuals=True, normalize_features=False):

        features = []

        if seasonals:
            for i in range(len(self.decomp_model.seasonal.columns)):
                self.data[f'seasonal_{i}'] = self.decomp_model.seasonal.iloc[:, i].values

                if normalize_features:
                    self.data[f'seasonal_{i}'] = self.normalize_indicator(f'seasonal_{i}')

                features.append(f'seasonal_{i}')

        if residuals:
            self.data['resid'] = self.decomp_model.resid

            if normalize_features:
                self.data['resid'] = self.normalize_indicator('resid')

            features.append('resid')

        self.model_info['Seasonal'].update({'Features': features})

        return self.data[features]

    def trend_features(self, trend=True, SMAs=False, sma_lens=None, momentum=False,
                       momentum_lens=None, BBands=False, bband_window=20, bband_width=2, normalize_features=True):

        if sma_lens is None:
            sma_lens = [20, 50, 100]

        if momentum_lens is None:
            momentum_lens = [21, 42, 63]

        features = []

        if trend:
            self.data['trend'] = self.decomp_model.trend.values
            self.data['trend_roc'] = self.data.trend.pct_change()

            if normalize_features:
                self.data['trend_x_close'] = self.normalize_ma('trend')

                features = features + ['trend_roc', 'trend_x_close']
            else:
                features = features + ['trend', 'trend_roc']

        if SMAs:
            for i in sma_lens:
                self.data[f'SMA_{i}'] = self.data.Close.rolling(i).mean()
                if normalize_features:
                    self.data[f'SMA_{i}x'] = (self.data.Close - self.data[f'SMA_{i}']) / self.data.Close
                    features.append(f'SMA_{i}x')
                else:
                    features.append(f'SMA_{i}')

        if momentum:
            for i in momentum_lens:
                if normalize_features:
                    self.data[f'mom_{i}'] = self.normalized_returns(i)

                else:
                    self.data[f'mom_{i}'] = self.data.Close.diff(i)
                features.append(f'mom_{i}')

        if BBands:
            self.data['bb_ma'] = self.data.Close.rolling(bband_window).mean()
            self.data['bb_upper'] = self.data.bb_ma + (bband_width * self.data.Close.rolling(bband_window).std())
            self.data['bb_lower'] = self.data.bb_ma - (bband_width * self.data.Close.rolling(bband_window).std())

            if normalize_features:
                for i in ['bb_ma', 'bb_upper', 'bb_lower']:
                    self.data[i + '_x'] = self.normalize_ma(i)
                    features.append(i + '_x')

            else:
                features = features + ['bb_ma', 'bb_upper', 'bb_lower']

        self.model_info['Trend'].update({'Features': features})

        return self.model_info['ML']

    def normalized_returns(self, offset):
        return (
            calc_returns(self.data['Close'], offset)/
            self.data.daily_vol/
            np.sqrt(offset)
        )

    def volume_features(self, volume_data=True, volume_MA=False, ma_lens=[5, 10], ma_diffs=False,
                        rel_vol=True, rel_by='tdays', length=3,
                        cum_vol=False, by='month', cum_len=5):

        features = []

        volume = self.data.Volume

        if volume_data:
            features.append('Volume')

        if volume_MA or ma_diffs:

            for i in ma_lens:
                self.data[f'Volume_{i}'] = volume.rolling(i).mean()
                if volume_MA: features.append(f'Volume_{i}')
                if ma_diffs:
                    self.data[f'Volume_{i}_x'] = (volume - self.data[f'Volume_{i}']) / volume.rolling(65).std()
                    features.append(f'Volume_{i}_x')

        if rel_vol:
            if rel_by == 'tdays':
                trading_days = get_trading_days(self.data.index.year.min(), self.data.index.year.max() + 1)
                self.data['tdays'] = trading_days.loc[self.data.index.date[:-1]]

            self.data[f'rvol_{length}'] = rvol(self.data, length=length, by=rel_by)
            features.append(f'rvol_{length}')

        self.model_info['Volume'].update({'Features': features})

        return self.data[features]

    def volatility_features(self, tr_ratio=False, atr_ratio_len=7, ATR=False, ATR_len=5, nATR=True, nATR_len=8,
                            TR=False):

        features = []

        if TR:
            self.data['TR'] = tr(self.data.Close)
        if tr_ratio:
            self.data['tr_ratio'] = tr(self.data) / atr(self.data, length=atr_ratio_len)
            features.append('tr_ratio')

        if ATR:
            self.data['ATR'] = atr(self.data, length=ATR_len)
            features.append('ATR')

        if nATR:
            self.data['nATR'] = atr(self.data, length=nATR_len, normalized=True)
            features.append('nATR')

        self.model_info['Volatility'].update({'Features': features})

        return self.data[features]

    def prepare_features(self, feat_types=['Trend'], normalize=True):
        for t in feat_types:
            if normalize:
                for feat in self.model_info[t]['Features']:
                    mean, std = self.data[feat].mean(), self.data[feat].std()
                    self.data[f'{feat}_norm'] = (self.data[feat] - mean)/std
                    self.feature_columns.append(f'{feat}_norm')
            else:
                self.feature_columns += self.model_info[t]['Features']

        self.process_data()

        return self
    def process_data(self, mini_batch=False, window_size=20):
        self.data = self.data.ffill()
        self.processed_data = self.data.ffill()[(~self.data.isna()| ~self.data.isnull())].dropna()
        if len(self.processed_data) < 0:
            print('Error Processing Data, values removed')
        else:
            return self.processed_data

    def __len__(self):
        return len(self.processed_data) - self.window_size

    def __getitem__(self, idx):
        """
        Get a window of data and the corresponding target returns.

        Args:
            idx: Index to access

        Returns:
            Tuple of (features, target_returns)
        """
        # Get features for the time window
        features = self.processed_data.iloc[idx: idx + self.window_size][self.feature_columns].values

        # Get target returns
        target_returns = self.processed_data.iloc[idx + self.window_size:idx + self.window_size + self.output_window, self.processed_data.columns.get_loc('target_returns')]

        # Convert to PyTorch tensors
        # The issue might be that target_returns is a scalar value or a pandas Series
        if isinstance(target_returns, (int, float)):
            # If it's a scalar, wrap it in a list
            target_tensor = torch.FloatTensor([target_returns])
        elif hasattr(target_returns, '__iter__'):
            # If it's already an iterable (like a list, numpy array, or pandas Series)
            target_tensor = torch.FloatTensor(target_returns)
        else:
            # Catch any other cases
            target_tensor = torch.FloatTensor([float(target_returns)])



        return torch.FloatTensor(features), target_tensor


# import the necessary libraries to execute this code
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

from vect_gan.synthesizers.vectgan import VectGan
from ctgan import CTGAN
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer
from sdv.metadata.single_table import SingleTableMetadata
# import model frameworks
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from sklearn.model_selection import RandomizedSearchCV as RSCV
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import autograd
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401 (Required for plotting styles)
import seaborn as sns
from tqdm import tqdm
from vect_gan.synthesizers.vectgan import VectGan
from ctgan import CTGAN
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer
from sdv.metadata.single_table import SingleTableMetadata
import torch.nn.functional as F
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch import autograd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.nn.utils import spectral_norm
from torch import autograd

# os.makedirs("new", exist_ok=True)
# os.makedirs("models", exist_ok=True)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.nn.utils import spectral_norm
from torch import autograd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch import autograd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch import autograd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.nn.utils import spectral_norm

class TempusGan:
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
        nhead=4,
        batch_size=64,
        epochs=200,
        encoder_lr=1e-4,
        decoder_lr=1e-4,
        discriminator_lr=2e-4,
        lr_step_size=20,
        lr_gamma=.75,
        device=None
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.discriminator_lr = discriminator_lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[TempusGan] Using device: {self.device}")
        self.trained = False

    # -------------------------- Pre/Post Processing -------------------------- #
    def _preprocess(self, df, discrete_columns, is_fit_call=False):
        if is_fit_call:
            self.discrete_columns = list(discrete_columns)
            self.continuous_columns = [c for c in df.columns if c not in self.discrete_columns]
            self.original_columns = list(df.columns)
            self.onehot_encoders = {}
            self.discrete_column_indices = {}
            self.scaler = StandardScaler() if self.continuous_columns else None
            self.original_discrete_data = (
                df[self.discrete_columns].copy() if self.discrete_columns else pd.DataFrame()
            )

        discrete_data_list, start_idx = [], 0
        for col in self.discrete_columns:
            if is_fit_call:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                discrete_col = ohe.fit_transform(df[[col]].values)
                self.onehot_encoders[col] = ohe
                self.discrete_column_indices[col] = (start_idx, start_idx + discrete_col.shape[1])
                start_idx += discrete_col.shape[1]
            else:
                discrete_col = self.onehot_encoders[col].transform(df[[col]].values)
            discrete_data_list.append(discrete_col.astype(np.float32))

        discrete_data = (
            np.concatenate(discrete_data_list, axis=1).astype(np.float32)
            if discrete_data_list
            else np.zeros((len(df), 0), dtype=np.float32)
        )
        if is_fit_call:
            self.discrete_dim = discrete_data.shape[1]

        if self.continuous_columns:
            continuous_data = df[self.continuous_columns].values.astype(np.float32)
            if is_fit_call:
                continuous_data = self.scaler.fit_transform(continuous_data)
            else:
                continuous_data = self.scaler.transform(continuous_data)
        else:
            continuous_data = np.zeros((len(df), 0), dtype=np.float32)

        if is_fit_call:
            self.continuous_dim = continuous_data.shape[1]

        all_data = np.concatenate([discrete_data, continuous_data], axis=1).astype(np.float32)
        if is_fit_call:
            self.data_dim = all_data.shape[1]

        return all_data

    def _inverse_transform(self, X):
        out_df = pd.DataFrame()
        for col in self.discrete_columns:
            start_idx, end_idx = self.discrete_column_indices[col]
            # Ensure the slice is one-hot encoded before inverse transforming
            one_hot_slice = np.zeros((X.shape[0], end_idx - start_idx))
            one_hot_slice[np.arange(X.shape[0]), X[:, start_idx:end_idx].argmax(1)] = 1
            inv_data = self.onehot_encoders[col].inverse_transform(one_hot_slice)
            out_df[col] = inv_data.flatten()

        if self.continuous_columns:
            continuous_start = self.discrete_dim
            inv_cont_data = self.scaler.inverse_transform(X[:, continuous_start:])
            for i, col in enumerate(self.continuous_columns):
                out_df[col] = inv_cont_data[:, i]

        return out_df[self.original_columns]

    def _sample_realistic_conditions(self, n_samples):
        if self.discrete_dim == 0:
            return torch.zeros(n_samples, 0, device=self.device)
        if len(self.original_discrete_data) > 0:
            idx = np.random.choice(len(self.original_discrete_data), n_samples, replace=True)
            sampled = self.original_discrete_data.iloc[idx].reset_index(drop=True)
            parts = [
                self.onehot_encoders[col].transform(sampled[[col]].values).astype(np.float32)
                for col in self.discrete_columns
            ]
            arr = np.concatenate(parts, axis=1)
            return torch.tensor(arr, device=self.device, dtype=torch.float32)
        return torch.zeros(n_samples, self.discrete_dim, device=self.device)

    # --------------------------- Model Components ---------------------------- #
    class _LinearAttention(nn.Module):
        def __init__(self, d_model, nhead, dropout=0.1):
            super().__init__()
            self.d_head = d_model // nhead
            self.nhead = nhead
            self.qkv_proj = nn.Linear(d_model, d_model * 3)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
        @staticmethod
        def _phi(x): return F.elu(x) + 1.0
        def forward(self, x):
            B, L, H = x.shape
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
            q, k, v = (t.view(B, L, self.nhead, self.d_head).transpose(1, 2) for t in (q, k, v))
            q, k = self._phi(q), self._phi(k)
            kv = torch.einsum('bhld,bhle->bhde', k, v)
            k_sum = k.sum(dim=2)
            z = 1.0 / (torch.einsum('bhld,bhd->bhl', q, k_sum).unsqueeze(-1) + 1e-6)
            out = torch.einsum('bhld,bhde->bhle', q, kv) * z
            out = out.transpose(1, 2).contiguous().view(B, L, H)
            return self.out_proj(self.dropout(out))

    class _LATEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, mlp_ratio=4, dropout=0.1):
            super().__init__()
            self.attn = TempusGan._LinearAttention(d_model, nhead, dropout=dropout)
            self.ln1 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(mlp_ratio * d_model, d_model), nn.Dropout(dropout),
            )
            self.ln2 = nn.LayerNorm(d_model)
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.ff(self.ln2(x))

    class _LATransformer(nn.Module):
        def __init__(self, d_model, nhead, num_layers=2, dropout=0.1):
            super().__init__()
            self.layers = nn.ModuleList(
                [TempusGan._LATEncoderLayer(d_model, nhead, dropout=dropout) for _ in range(num_layers)]
            )
        def forward(self, x):
            for layer in self.layers: x = layer(x)
            return x

    class LSTMBlock(nn.Module):
        def __init__(self, input_dim, hidden_dim, dropout_p=0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
            self.proj = nn.Linear(hidden_dim, hidden_dim)
            self.bn = nn.BatchNorm1d(hidden_dim)
        def forward(self, x):
            h, _ = self.lstm(x)
            h = F.gelu(self.proj(h))
            return self.bn(h.transpose(1, 2)).transpose(1, 2)

    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, nhead):
            super().__init__()
            self.embed = nn.Linear(1, hidden_dim)
            self.lstm = TempusGan.LSTMBlock(hidden_dim, hidden_dim)
            self.xfmr = TempusGan._LATransformer(hidden_dim, nhead, num_layers=2)
            self.norm = nn.LayerNorm(hidden_dim)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        def forward(self, x):
            h = self.embed(x.transpose(1, 2))
            h = self.lstm(h)
            h = self.xfmr(h)
            h = self.pool(self.norm(h).transpose(1, 2)).squeeze(-1)
            return self.fc_mu(h), self.fc_logvar(h)

    class Decoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim, output_dim, num_layers, nhead, cond_dim):
            super().__init__()
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim
            self.token_len = max(8, min(128, output_dim // 4))
            self.fc = nn.Sequential(
                nn.Linear(latent_dim + cond_dim, hidden_dim * self.token_len), nn.GELU()
            )
            self.xfmr = TempusGan._LATransformer(hidden_dim, nhead, num_layers=2)
            self.lstm = TempusGan.LSTMBlock(hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
            self.to_feat = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        def forward(self, z, cond):
            zc = torch.cat([z, cond], dim=1) if cond.shape[1] > 0 else z
            h = self.fc(zc).view(-1, self.token_len, self.hidden_dim)
            h = self.xfmr(h)
            h = self.lstm(h)
            h = F.interpolate(self.norm(h).transpose(1, 2), size=self.output_dim, mode='linear')
            return self.to_feat(h)

    class Discriminator(nn.Module):
        def __init__(self, input_dim, hidden_dim, cond_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim + cond_dim, hidden_dim), nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, 1),
            )
        def forward(self, x, cond):
            x_cond = torch.cat([x, cond], dim=1) if cond.shape[1] > 0 else x
            return self.model(x_cond)

    # --------------------------------- Train --------------------------------- #
    def fit(self, train_data, discrete_columns=[]):
        X = self._preprocess(train_data, discrete_columns, is_fit_call=True)
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.encoder = self.Encoder(self.data_dim, self.hidden_dim, self.latent_dim, self.num_layers, self.nhead).to(self.device)
        self.decoder = self.Decoder(self.latent_dim, self.hidden_dim, self.data_dim, self.num_layers, self.nhead, self.discrete_dim).to(self.device)
        self.discriminator = self.Discriminator(self.data_dim, self.hidden_dim, self.discrete_dim).to(self.device)

        opt_e = optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        opt_d = optim.Adam(self.decoder.parameters(), lr=self.decoder_lr)
        opt_disc = optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr)

        sch_e = optim.lr_scheduler.StepLR(opt_e, step_size=self.lr_step_size, gamma=self.lr_gamma)
        sch_d = optim.lr_scheduler.StepLR(opt_d, step_size=self.lr_step_size, gamma=self.lr_gamma)
        sch_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=self.lr_step_size, gamma=self.lr_gamma)

        critic_iter = 5
        gp_lambda = 10

        for epoch in range(self.epochs):
            for i, (x_batch,) in enumerate(loader):
                x = x_batch.to(self.device)
                B = x.size(0)
                x_seq = x.unsqueeze(1)
                cond_vec = x[:, :self.discrete_dim] if self.discrete_dim > 0 else torch.zeros(B, 0, device=self.device)

                # --- Train Discriminator (WGAN-GP) ---
                opt_disc.zero_grad()
                
                # Real samples
                d_real = self.discriminator(x, cond_vec)

                # Fake samples
                with torch.no_grad():
                    z_fake = torch.randn(B, self.latent_dim, device=self.device)
                    cond_fake = self._sample_realistic_conditions(B)
                    x_fake = self.decoder(z_fake, cond_fake).squeeze(1).detach()
                d_fake = self.discriminator(x_fake, cond_vec)

                # Gradient Penalty
                alpha = torch.rand(B, 1, device=self.device)
                x_hat = (alpha * x.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                d_hat = self.discriminator(x_hat, cond_vec)
                gradients = autograd.grad(outputs=d_hat.sum(), inputs=x_hat, create_graph=True)[0]
                gradient_penalty = gp_lambda * ((gradients.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()
                
                d_loss = torch.mean(d_fake) - torch.mean(d_real) + gradient_penalty
                d_loss.backward()
                opt_disc.step()

                # --- Train Generator & Encoder ---
                if i % critic_iter == 0:
                    opt_e.zero_grad()
                    opt_d.zero_grad()

                    # VAE Pass
                    mu, logvar = self.encoder(x_seq)
                    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) # Reparameterization Trick
                    x_rec = self.decoder(z, cond_vec).squeeze(1)

                    recon_loss = 0.0
                    if self.discrete_dim > 0:
                        recon_loss += F.binary_cross_entropy_with_logits(x_rec[:, :self.discrete_dim], x[:, :self.discrete_dim])
                    if self.continuous_dim > 0:
                        recon_loss += F.mse_loss(x_rec[:, self.discrete_dim:], x[:, self.discrete_dim:])
                    
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
                    vae_loss = recon_loss + 0.1 * kl_loss # Using 0.1 as a common weight for KL term

                    # GAN Adversarial Pass
                    z_fake_g = torch.randn(B, self.latent_dim, device=self.device)
                    cond_fake_g = self._sample_realistic_conditions(B)
                    x_fake_g = self.decoder(z_fake_g, cond_fake_g).squeeze(1)
                    g_adv_loss = -self.discriminator(x_fake_g, cond_fake_g).mean()

                    total_g_loss = vae_loss + 0.1 * g_adv_loss # Weighting the adversarial loss
                    total_g_loss.backward()
                    opt_e.step()
                    opt_d.step()

            sch_e.step()
            sch_d.step()
            sch_disc.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | D Loss: {d_loss.item():.4f} | G Adv Loss: {g_adv_loss.item():.4f} | VAE Loss: {vae_loss.item():.4f}"
                )
        self.trained = True

    # --------------------------------- Sample -------------------------------- #
    def sample(self, n=100):
        if not self.trained:
            raise RuntimeError("Model must be trained first.")

        self.decoder.eval()
        self.discriminator.eval()
        m = int(n * 3)

        print(f"Generating {n} synthetic samples (oversample-and-rank strategy)...")
        with torch.no_grad():
            z = torch.randn(m, self.latent_dim, device=self.device)
            cond_samples = self._sample_realistic_conditions(m)
            x_gen = self.decoder(z, cond_samples).squeeze(1)
            scores = self.discriminator(x_gen, cond_samples).squeeze()
            _, top_indices = torch.topk(scores, n)
            x_best = x_gen[top_indices]

        return self._inverse_transform(x_best.cpu().numpy())

class NESTED_CV:
  
    """
    NESTED_CV Class:
    - based on a dataset for long acting injectible (LAI) drug delivey systems
    - contains 12 different model architectures and non-exaustive hyperparamater spaces for those models
    - actiavted by abbriviations for these model - incorrect keywords triggers a message with available key words
    - once model type is selected, NEST_CV will be conducted, data is spli as follows:
          - outer_loop (test) done by GroupShuffleSplit where 20% of the drug-polymer groups in the dataset are held back at random
          - inner_loop (HP screening) done by GroupKFold based 10 splits in the dataset - based on drug-polymer groups
    - default is 10-folds for the NESTED_CV, but this can be entered manually
    - prints progress and reults at the end of each loop
    - configures a pandas dataframe with the reults of the NESTED_CV
    - fits and trains the best model based on the reults of the NESTED_CV
    """

    def __init__(self, datafile = "few_shot_models/Dataset_17_feat.xlsx", model_type = None):
        self.df = pd.read_excel(datafile)
          
        if model_type == 'MLR':
          self.user_defined_model = LinearRegression()
          self.p_grid = {'fit_intercept':[True, False],
                         'positive':[True, False]}
    
        elif model_type == 'lasso':
          self.user_defined_model = linear_model.Lasso()
          self.p_grid = {'alpha':[0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
                        'positive':[True, False]}

        elif model_type == 'kNN':
          self.user_defined_model = KNeighborsRegressor()
          self.p_grid ={'n_neighbors':[2, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50],
                        'weights': ["uniform", 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'leaf_size': [10, 30, 50, 75, 100],
                        'p':[1, 2],
                        'metric': ['minkowski']}

        elif model_type == 'PLS':
          self.user_defined_model = PLSRegression()
          self.p_grid ={'n_components':[2, 4, 6],
                        'max_iter': [250, 500, 750, 1000]}

        elif model_type == 'SVR':
          self.user_defined_model = SVR()
          # Fixed: Split parameter grids by kernel type to avoid invalid combinations
          self.p_grid = [
              # Linear kernel
              {'kernel': ['linear'],
               'gamma': ['scale', 'auto'],
               'C': [0.1, 0.5, 1, 2],
               'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
               'shrinking': [True, False]},
              # RBF kernel
              {'kernel': ['rbf'],
               'gamma': ['scale', 'auto'],
               'C': [0.1, 0.5, 1, 2],
               'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
               'shrinking': [True, False]},
              # Polynomial kernel
              {'kernel': ['poly'],
               'degree': [2, 3, 4, 5, 6],
               'gamma': ['scale', 'auto'],
               'C': [0.1, 0.5, 1, 2],
               'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
               'shrinking': [True, False]},
              # Sigmoid kernel
              {'kernel': ['sigmoid'],
               'gamma': ['scale', 'auto'],
               'C': [0.1, 0.5, 1, 2],
               'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
               'shrinking': [True, False]}
          ]
        
        elif model_type == 'DT':
          self.user_defined_model = DecisionTreeRegressor(random_state=4)
          self.p_grid ={'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        'splitter':['best', 'random'],
                        'max_depth':[None],
                        'min_samples_split':[2,4,6],
                        'min_samples_leaf':[1,2,4],
                        'max_features': [None, 1.0, 'sqrt','log2'],  # Fixed: 'auto' -> 1.0
                        'ccp_alpha': [0, 0.05, 0.1, 0.15]}  
        
        elif model_type == 'RF':
          self.user_defined_model = RandomForestRegressor(random_state=4)
          self.p_grid ={'n_estimators':[100,300,400],
                        'criterion':['squared_error', 'absolute_error'],
                        'max_depth':[None],
                        'min_samples_split':[2,4,6,8],
                        'min_samples_leaf':[1,2,4],
                        'min_weight_fraction_leaf':[0.0],
                        'max_features': [1.0, 'sqrt'],  # Fixed: 'auto' -> 1.0
                        'max_leaf_nodes':[None],
                        'min_impurity_decrease': [0.0],
                        'bootstrap':[True],
                        'oob_score':[True],
                        'ccp_alpha': [0, 0.005, 0.01]}

        elif model_type == 'LGBM':
          self.user_defined_model = LGBMRegressor(random_state=4)
          self.p_grid ={"n_estimators":[100,150,200,250,300,400,500,600],
                        'boosting_type': ['gbdt', 'dart', 'goss'],
                        'num_leaves':[16,32,64,128,256],
                        'learning_rate':[0.1,0.01,0.001,0.0001],
                        'min_child_weight': [0.001,0.01,0.1,1.0,10.0],
                        'subsample': [0.4,0.6,0.8,1.0],
                        'min_child_samples':[2,10,20,40,100],
                        'reg_alpha': [0, 0.005, 0.01, 0.015],
                        'reg_lambda': [0, 0.005, 0.01, 0.015]}
        
        elif model_type == 'XGB':
          self.user_defined_model = XGBRegressor(objective ='reg:squarederror')
          self.p_grid ={'booster': ['gbtree', 'gblinear', 'dart'],
                        "n_estimators":[100, 150, 300, 400],
                        'max_depth':[3, 4, 5, 6, 7, 8, 9, 10],
                        'gamma':[0, 2, 4, 6, 8, 10],
                        'learning_rate':[0.3, 0.2, 0.1, 0.05, 0.01],
                        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'min_child_weight': [1.0, 2.0, 4.0, 5.0],
                        'max_delta_step':[1, 2, 4, 6, 8, 10],
                        'reg_alpha':[0.001, 0.01, 0.1],
                        'reg_lambda': [0.001, 0.01, 0.1]}                
        
        elif model_type == 'NGB':
          b1 = DecisionTreeRegressor(criterion='squared_error', max_depth=2)
          b2 = DecisionTreeRegressor(criterion='squared_error', max_depth=4)
          b3 = DecisionTreeRegressor(criterion='squared_error', max_depth=8) 
          b4 = DecisionTreeRegressor(criterion='squared_error', max_depth=12)
          b5 = DecisionTreeRegressor(criterion='squared_error', max_depth=16)
          b6 = DecisionTreeRegressor(criterion='squared_error', max_depth=32) 
          self.user_defined_model = NGBRegressor()
          self.p_grid ={'n_estimators':[100,200,300,400,500,600,800],
                        'learning_rate': [0.1, 0.01, 0.001],
                        'minibatch_frac': [1.0, 0.8, 0.5],
                        'col_sample': [1, 0.8, 0.5],
                        'Base': [b1, b2, b3, b4, b5, b6]}
        
        else:
          print("#######################\nSELECTION UNAVAILABLE!\n#######################\n\nPlease chose one of the following options:\n\n 'MLR'for multiple linear regression\n\n 'lasso' for multiple linear regression with east absolute shrinkage and selection operator (lasso)\n\n 'kNN'for k-Nearest Neighbors\n\n 'PLS' for partial least squares\n\n 'SVR' for support vertor regressor\n\n 'DT' for decision tree\n\n 'RF' for random forest\n\n 'LGBM' for LightGBM\n\n 'XGB' for XGBoost\n\n 'NGB' for NGBoost")

    def input_target(self):
        X = self.df.drop(['Experimental_index','DP_Group','Release'],axis='columns')
        stdScale = StandardScaler().fit(X)
        self.X=stdScale.transform(X)
        self.Y = self.df['Release']
        self.G = self.df['DP_Group']
        self.E = self.df['Experimental_index']
        self.T = self.df['Time']    
    
    def cross_validation(self, input_value, synth_type):
        if input_value == None:
            NUM_TRIALS = 10
        else: 
            NUM_TRIALS = input_value

        self.itr_number = [] # create new empty list for itr number 
        self.outer_results = []
        self.inner_results = []
        self.model_params = []
        self.G_test_list = []
        self.y_test_list = []
        self.E_test_list = []
        self.T_test_list = []
        self.pred_list = []

        for i in range(NUM_TRIALS): #configure the cross-validation procedure - outer loop (test set) 
  
          cv_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i) #hold back 20% of the groups for test set

          # split data using GSS
          for train_index, test_index in cv_outer.split(self.X, self.Y, self.G):
              X_train, X_test = self.X[train_index], self.X[test_index]
              y_train, y_test = self.Y[train_index], self.Y[test_index]
              G_train, G_test = self.G[train_index], self.G[test_index]
              E_train, E_test = self.E[train_index], self.E[test_index]
              T_train, T_test = self.T[train_index], self.T[test_index]
              X_cols = self.df.drop(['Experimental_index','DP_Group','Release'], axis='columns').columns
              if synth_type == 'Normal':
                  X_train_aug, y_train_aug, G_train_aug = X_train, y_train, G_train
              else:
                  sd = pd.concat([
                    pd.DataFrame(X_train, columns=X_cols).reset_index(drop=True),
                    pd.Series(G_train, name="DP_Group").reset_index(drop=True).astype(str),
                    pd.Series(y_train, name="Release").reset_index(drop=True).astype(float),
                    pd.Series(E_train, name="Experimental_index").reset_index(drop=True).astype(str)
                ], axis=1)
                  if synth_type == 'VectGan':
                      vg = VectGan(batch_size=64, pac=8, epochs=30, verbose=False)
                      vg.fit(train_data=sd, discrete_columns=["DP_Group", "Experimental_index"])
                      syn = vg.sample(n=int(0.5 * len(X_train)))
                  elif synth_type == 'TempusGan':
                      tg = TempusGan()
                      tg.fit(train_data=sd, discrete_columns=["DP_Group", "Experimental_index"])
                      syn = tg.sample(n=int(0.5 * len(X_train)))
                  elif synth_type == 'CTGAN':
                      ct = CTGAN(epochs=30)
                      ct.fit(sd, discrete_columns=["DP_Group", "Experimental_index"])
                      syn = ct.sample(int(0.5 * len(X_train)))
                  elif synth_type == 'Gaussian Copula':
                      meta = SingleTableMetadata()
                      meta.detect_from_dataframe(sd)
                      for c in ["Experimental_index", "DP_Group"]:
                          meta.update_column(c, sdtype="categorical")
                      gc = GaussianCopulaSynthesizer(metadata=meta)
                      gc.fit(sd)
                      syn = gc.sample(num_rows=int(0.5 * len(X_train)))
                  elif synth_type == 'TVAE':
                      meta = SingleTableMetadata()
                      meta.detect_from_dataframe(sd)
                      for c in ["Experimental_index", "DP_Group"]:
                          meta.update_column(c, sdtype="categorical")
                      tvae = TVAESynthesizer(metadata=meta)
                      tvae.fit(sd)
                      syn = tvae.sample(num_rows=int(0.5 * len(X_train)))
                  X_aug = syn[X_cols].reset_index(drop=True)
                  y_aug = syn['Release'].reset_index(drop=True)
                  G_aug = syn['DP_Group'].reset_index(drop=True)
                  X_train_aug = pd.concat([pd.DataFrame(X_train, columns=X_cols), X_aug], ignore_index=True).to_numpy()
                  y_train_aug = pd.concat([pd.Series(y_train), y_aug], ignore_index=True).to_numpy()
                  G_train_aug = pd.concat([pd.Series(G_train), G_aug], ignore_index=True).to_numpy()
              # store test set information
              G_test = np.array(G_test) #prevents index from being brought from dataframe
              self.G_test_list.append(G_test)
              E_test = np.array(E_test) #prevents index from being brought from dataframe
              self.E_test_list.append(E_test)
              T_test = np.array(T_test) #prevents index from being brought from dataframe
              self.T_test_list.append(T_test)
              y_test = np.array(y_test) #prevents index from being brought from dataframe
              self.y_test_list.append(y_test)
                    
              # configure the cross-validation procedure - inner loop (validation set/HP optimization)
              cv_inner = GroupKFold(n_splits=5) #should be 10 fold group split for inner loop

              # define search space
              search = RSCV(self.user_defined_model, self.p_grid, n_iter=100, verbose=0, scoring='neg_mean_absolute_error', cv=cv_inner,  n_jobs= -2, refit=True) # should be 100
                      
              # execute search
              result = search.fit(X_train_aug, y_train_aug, groups=G_train_aug)
                  
              # get the best performing model fit on the whole training set
              best_model = result.best_estimator_

              # get the score for the best performing model and store
              best_score = abs(result.best_score_)
              self.inner_results.append(best_score)
                      
              # evaluate model on the hold out dataset
              yhat = best_model.predict(X_test)

              # store drug release predictions
              self.pred_list.append(yhat)

              # evaluate the model
              acc = mean_absolute_error(y_test, yhat)
                      
              # store the result
              self.itr_number.append(i+1)
              self.outer_results.append(acc)
              self.model_params.append(result.best_params_)

              # report progress at end of each inner loop
              print('\n################################################################\n\nSTATUS REPORT:') 
              print('Iteration '+str(i+1)+' of '+str(NUM_TRIALS)+' runs completed') 
              print('Test_Score: %.3f, Best_Valid_Score: %.3f, \n\nBest_Model_Params: \n%s' % (acc, best_score, result.best_params_))
              print("\n################################################################\n ")
          
    def results(self, tag, model_name):
        # build results DataFrame
        rows = zip(self.itr_number, self.inner_results, self.outer_results, self.model_params,
                   self.G_test_list, self.E_test_list, self.T_test_list, self.y_test_list, self.pred_list)
        CV_dataset = pd.DataFrame(rows, columns=['Iter','Valid Score','Test Score','Model Parms',
                                                 'DP_Groups','Experimental Index','Time',
                                                 'Experimental_Release','Predicted_Release'])
        CV_dataset['Score_difference'] = (CV_dataset['Valid Score'] - CV_dataset['Test Score']).abs()
        CV_dataset.sort_values(by=['Score_difference','Test Score'], ascending=True, inplace=True)
        CV_dataset.reset_index(drop=True, inplace=True)
        self.CV_dataset = CV_dataset

        # ensure result dir exists: few_shot_models\NESTED_CV_RESULTS\<tag>\
        out_dir = os.path.join("few_shot_models", "NESTED_CV_RESULTS", str(tag))
        os.makedirs(out_dir, exist_ok=True)

        # save csv + pickle
        csv_path = os.path.join(out_dir, f"nested_cv_results_{tag}_{model_name}.csv")
        pkl_path = os.path.join(out_dir, f"nested_cv_results_{tag}_{model_name}.pkl")
        CV_dataset.to_csv(csv_path, index=False)
        CV_dataset.to_pickle(pkl_path)
        print(f"[INFO] Saved CV results: {csv_path}")
        print(f"[INFO] Saved CV results (pkl): {pkl_path}")

    def best_model(self, tag, model_name=None):
        # pick best params and fit on full data
        best_params = self.CV_dataset.iloc[0, 3]
        best_model = self.user_defined_model.set_params(**best_params)
        self.best_model = best_model.fit(self.X, self.Y)

        # ensure model dir exists
        out_dir = os.path.join("few_shot_models", "Trained_models")
        os.makedirs(out_dir, exist_ok=True)

        # filename includes tag (+ model name if provided)
        suffix = f"_{model_name}" if model_name else ""
        model_path = os.path.join(out_dir, f"best_model_{tag}{suffix}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.best_model, f)
        print(f"[INFO] Saved best model: {model_path}")


def main():
    synth_types = ['TempusGan', 'VectGan', 'CTGAN', 'Gaussian Copula', 'TVAE', 'Normal']
    all_models  = ['LGBM', 'MLR', 'lasso', 'kNN', 'PLS', 'SVR', 'DT', 'RF', 'XGB', 'NGB']
    all_results = []

    for model_name in all_models:
        for synth in synth_types:
            print(f"\n==== Running Model: {model_name} | Synth: {synth} ====")
            cv_runner = NESTED_CV(datafile="few_shot_models/Dataset_17_feat.xlsx",
                                  model_type=model_name)

            cv_runner.input_target()
            cv_runner.cross_validation(input_value=5, synth_type=synth)

            # save per-synth results with tag
            cv_runner.results(tag=synth, model_name=model_name)

            # fit & save best model; include model_name in filename
            cv_runner.best_model(tag=synth, model_name=model_name)

            # collect for global summary
            df = cv_runner.CV_dataset.copy()
            df['Model'] = model_name
            df['Synth_Type'] = synth
            all_results.append(df)

    # Combine and save global summaries
    final_results = pd.concat(all_results, ignore_index=True)
    summary = (final_results
               .groupby(['Model','Synth_Type'])[['Test Score','Valid Score']]
               .agg(['mean','std','min','max'])
               .reset_index())
    pivot = final_results.pivot_table(index='Model', columns='Synth_Type',
                                      values='Test Score', aggfunc='mean')

    final_results.to_csv("all_nested_cv_results_few.csv", index=False)
    summary.to_csv("summary_by_model_synth_few.csv", index=False)
    pivot.to_csv("pivot_by_model_synth_few.csv")

    print("\nAll detailed results saved to all_nested_cv_results_few.csv")
    print("Summary saved to summary_by_model_synth_few.csv")
    print("Pivot table saved to pivot_by_model_synth_few.csv\n")
    print("=== PIVOT TABLE (Test Score Mean) ===")
    print(pivot)

if __name__ == "__main__":
    main()


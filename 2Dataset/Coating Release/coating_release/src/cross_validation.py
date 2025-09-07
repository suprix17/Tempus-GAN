"""Module for performing nested cross-validation on a given dataset."""

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

os.makedirs("new", exist_ok=True)
os.makedirs("models", exist_ok=True)
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


class TempusGan:
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
        nhead=4,
        batch_size=64,
        epochs=1500,
        encoder_lr=1e-4,
        decoder_lr=1e-4,
        discriminator_lr=2e-4,
        lr_step_size=300,
        lr_gamma=1,
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


def get_model_and_params(model_name: str) -> tuple[BaseEstimator, dict]:
    """Return an uninitialised model and a parameter grid for the given model name."""
    if model_name == "LightGBM":
        model = LGBMRegressor(verbosity=-1)
        model_parameters = {
            "n_estimators": [100, 150, 300, 400],
            "max_depth": [5, 10, 15, 20],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "subsample": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [10, 20, 30, 40],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "reg_alpha": [0.001, 0.01, 0.1],
            "reg_lambda": [0.001, 0.01, 0.1],
        }
    elif model_name == "XGBoost":
        model = XGBRegressor()
        model_parameters = {
            "booster": ["gbtree", "dart"],
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [3, 6, 9, 12],
            "gamma": [0, 1, 3, 5],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.5, 0.7, 0.9],
            "min_child_weight": [1, 2, 3, 4],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [1, 0.1, 0.01],
        }
    elif model_name == "KNN":
        model = KNeighborsRegressor()
        model_parameters = {
            "n_neighbors": np.arange(1, 31),
            "p": [1, 2, 3],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        }
    elif model_name == "SVM":
        model = svm.SVR()
        model_parameters = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [1, 0.1, 0.01, 0.001, 0.0001, "scale", "auto"],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }
    elif model_name == "RF":
        model = RandomForestRegressor()
        model_parameters = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [10, 20, 30, 40, 50, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
        }
    else:
        raise ValueError("Unsupported model name.")
    return model, model_parameters


def scale_data(
    x_train: pd.DataFrame | np.ndarray, x_test: pd.DataFrame | np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Scale training and test data using MinMaxScaler and return scaled data with the scaler."""
    scaler = MinMaxScaler()
    scaler.fit(x_train)

    x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_train.columns)

    return x_train_scaled, x_test_scaled, scaler


def apply_pls(
    x_train_scaled: pd.DataFrame,
    x_test_scaled: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    pls_components: int,
) -> tuple[np.ndarray, np.ndarray, PLSRegression]:
    """Perform PLS transformation on selected columns of the scaled train and test data."""
    plsr = PLSRegression(n_components=pls_components)
    x_train_region = x_train_scaled.loc[:, 2001.063477:158.482422]
    x_test_region = x_test_scaled.loc[:, 2001.063477:158.482422]

    plsr.fit(x_train_region, y_train)
    x_train_pls = pd.DataFrame(plsr.transform(x_train_region))
    x_test_pls = pd.DataFrame(plsr.transform(x_test_region))

    x_train_other = x_train_scaled.loc[:, "medium":"time"]
    x_test_other = x_test_scaled.loc[:, "medium":"time"]

    x_train_final = np.array(pd.concat([x_train_pls, x_train_other], axis=1))
    x_test_final = np.array(pd.concat([x_test_pls, x_test_other], axis=1))

    return x_train_final, x_test_final, plsr


def run_randomised_search(
    model: BaseEstimator,
    params: dict,
    x_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    scoring: str,
    n_jobs: int,
    n_iter: int,
    cv_inner: KFold,
    random_state: int,
) -> RandomizedSearchCV:
    """Perform a RandomizedSearchCV on the provided model and return the fitted search object."""
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv_inner,
        random_state=random_state,
    )
    search.fit(x_train, y_train)
    return search


def nested_cross_validation(
    x: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    model_name: str,
    num_iter: int = 100,
    num_loops: int = 5,
    n_splits: int = 5,
    scoring: str = "r2",
    plot: bool = True,
    n_jobs: int = -1,
    pls: int | None = None,
    data_tag: str = "Normal",     # <---- NEW
) -> pd.DataFrame:
    """Perform nested cross-validation on a given dataset with a specified model and parameters.

    Parameters:
        x: Feature dataset, either a pandas DataFrame or numpy array.
        y: Target variable, either a pandas Series or numpy array.
        model_name: Name of the model to be used.
            Choices: 'lightGBM', 'XGBoost', 'KNN', 'SVM', 'RF'.
        num_iter: Number of iterations for RandomizedSearchCV. Defaults to 100.
        num_loops: Number of loops for the outer cross-validation. Defaults to 5.
        n_splits: Number of splits for the inner KFold cross-validation. Defaults to 5.
        scoring: Scoring metric for model evaluation. Defaults to 'r2'.
        plot: Whether to plot real vs. predicted values. Defaults to True.
        n_jobs: Number of CPU workers to use for computation. Defaults to 10.
        pls: Number of PLS components to apply to a specified spectral region, if any.
            Defaults to None.

    Returns:
        A pandas DataFrame containing performance metrics for each fold.
    """
    plt.style.use(["science", "no-latex"])
    
    model, model_parameters = get_model_and_params(model_name)
    results: list[dict] = []
    all_y_test, all_y_pred = [], []
    best_model_params = None
    best_score = float("-inf")
    TARGET = "__target__"
    for i in range(num_loops):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= i)
        x_train_scaled, x_test_scaled, scaler = scale_data(x_train, x_test)
        if data_tag == "Normal":
            x_t = x_train_scaled.copy()
            y_t = y_train.reset_index(drop=True).copy()

        else:
            # ===== build joint train df (X‖y) so synths learn label too =====
            train_df = x_train_scaled.reset_index(drop=True).copy()
            train_df[TARGET] = y_train.reset_index(drop=True)

            # ===== pick synthesizer and fit on joint df =====
            if data_tag == "TempusGan":
                tg = TempusGan()
                tg.fit(train_data=train_df)  # learns both X and y columns
                syn = tg.sample(n=int(3 * len(train_df)))

            elif data_tag == "VectGan":
                vg = VectGan()
                vg.fit(train_data=train_df)
                syn = vg.sample(n=int(3 * len(train_df)))

            elif data_tag == "CTGAN":
                ct = CTGAN(epochs=50)
                ct.fit(train_df)
                syn = ct.sample(int(3 * len(train_df)))

            elif data_tag == "Gaussian Copula":
                meta = SingleTableMetadata()
                meta.detect_from_dataframe(train_df)
                gc = GaussianCopulaSynthesizer(metadata=meta)
                gc.fit(train_df)
                syn = gc.sample(int(3 * len(train_df)))

            elif data_tag == "TVAE":
                meta = SingleTableMetadata()
                meta.detect_from_dataframe(train_df)
                tvae = TVAESynthesizer(metadata=meta)
                tvae.fit(train_df)
                syn = tvae.sample(int(3 * len(train_df)))

            else:
                raise ValueError(f"Unknown data_tag: {data_tag}")

            # ===== concat original + synthetic (still joint) =====
            aug_df = pd.concat([train_df, syn], ignore_index=True)

            # ===== split back into X and y to feed model/search =====
            y_t = aug_df[TARGET].copy()
            x_t = aug_df.drop(columns=[TARGET]).copy()

        cv_inner = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        search = run_randomised_search(
            model=model,
            params=model_parameters,
            x_train=x_t,
            y_train=y_t,
            scoring=scoring,
            n_jobs=n_jobs,
            n_iter=num_iter,
            cv_inner=cv_inner,
            random_state=42,
        )

        current_best_score = search.best_score_
        if current_best_score > best_score:
            best_score = current_best_score
            best_model_params = search.best_params_

        y_pred = search.predict(x_test_scaled)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        results.append(
            {
                "Iteration": i + 1,
                "R2 Score": r2,
                "MAE": mae,
                "MSE": mse,
                "Best Model Parameters": search.best_params_,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"new/{data_tag}_{model_name}_cv_scores.csv", index=False)

    x_scaled = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)

    if pls is not None:
        plsr_full = PLSRegression(n_components=pls)
        x_region_full = x_scaled.loc[:, 2001.063477:158.482422]
        plsr_full.fit(x_region_full, y)
        x_pls_full = pd.DataFrame(plsr_full.transform(x_region_full))

        x_other_full = x_scaled.loc[:, "medium":"time"]
        x_scaled = np.array(pd.concat([x_pls_full, x_other_full], axis=1))

    if best_model_params is None:
        raise ValueError("No best model parameters found. Please check your data or parameters.")

    best_model = model.set_params(**best_model_params)
    best_model.fit(x_scaled, y)

    with open(f"models/best_{data_tag}_{model_name}_new.pkl", "wb") as file:
        pickle.dump(best_model, file)

    if plot:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=all_y_test, y=all_y_pred, edgecolor="k", s=100, alpha=0.6)
        min_val = min(*all_y_test, *all_y_pred)
        max_val = max(*all_y_test, *all_y_pred)
        plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.title(f"Actual vs. Predicted - {model_name}", fontsize=14)
        plt.savefig(f"new/predicted_vs_real_{data_tag}_{model_name}_plsr.png", dpi=300)
        plt.show()

    return results_df
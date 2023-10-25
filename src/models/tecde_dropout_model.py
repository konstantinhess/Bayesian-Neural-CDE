import torch
import pytorch_lightning as pl
import torch.nn as nn
import torchsde
import torchcde
import torch.nn.functional as F
from torch import distributions as dist
import torch.optim as optim
from src.models.CoverageMetric import CoverageMetric


class SequentialModel(nn.Module): # customize neural vector field
    def __init__(self, input_size, output_size, hidden_layers):
        super(SequentialModel, self).__init__()

        # Define layers
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.extend([nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.ReLU()])
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Sequential model
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return F.tanh(self.model(x))

class MC_dropout_trainable_net(pl.LightningModule): # dropout head
    def __init__(self, input_size, output_size, mc_samples, dropout_probability=0.5, hidden_sizes=None):
        super().__init__()

        self.mc_samples = mc_samples
        self.dropout = nn.Dropout(p=dropout_probability)

        if hidden_sizes is None:
            hidden_sizes = []
        elif isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes, )

        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            nn.init.constant_(self.layers[i].bias, 0.0)

    def forward(self, x):
        # Pass the input through each layer
        for layer in self.layers[:-1]:
            x = layer(x)
            #x = F.relu(x)
            x = F.tanh(x)
        if self.dropout.training: # i.e. during prediction
            x = x.repeat(self.mc_samples, 1, 1, 1)
            x = self.dropout(x)
        else: # create artificial dim
            x = x.unsqueeze(0)
        x = self.layers[-1](x)
        return x


class CDE_field(pl.LightningModule): # CDE submodule
    def __init__(self, hidden_size, control_size, integrand, control_path, batch_size): #control_batch, interpolation_method, ts):
        super().__init__()

        # Integrand
        self.hidden_size = hidden_size
        self.integrand = integrand

        # Control path
        self.control_size = control_size

        self.control_path = control_path
        self.batch_size = batch_size

    def forward(self, t, z):
        # Pass through integrand
        z = self.integrand(z).reshape(self.batch_size, self.hidden_size, self.control_size)
        # Matrix mult. with control path derivative and return
        return (z @ self.control_path.derivative(t).unsqueeze(-1)).squeeze(-1)


# CDE baseline module
class TE_CDE_dropout_head(pl.LightningModule):
    def __init__(self, hidden_size, hidden_layers, control_size, mc_samples, dropout_probability, intensity_weighting=False,
                 coverage_metric_confidence=0.95,
                 treatment_size=4, prediction_window=1,
                 interpolation_method='linear', learning_rate=1e-3, method='euler'):
        super().__init__()
        self.noise_type = "diagonal"  # required to use sde int module; noise is 0
        self.sde_type = "ito"  # required to use sde int module; use euler integration

        self.method = method
        self.tau = prediction_window

        self.mc_samples = mc_samples
        self.dropout_probability = dropout_probability

        self.intensity_weighting = intensity_weighting

        # Embedding network
        self.e_net = nn.Sequential(nn.Linear(control_size,hidden_size))
        nn.init.constant_(self.e_net[0].bias, 0.0)


        self.control_size = control_size
        self.treatment_size = treatment_size
        if interpolation_method == 'linear':
            self.inter_coeffs = torchcde.linear_interpolation_coeffs
            self.interpolate = torchcde.LinearInterpolation
        elif interpolation_method == 'spline':
            self.inter_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences
            self.interpolate = torchcde.CubicSpline

        # Hidden state flow
        self.hidden_size = hidden_size
        # Encoder
        self.CDE_integrand_enc = SequentialModel(input_size=hidden_size + 1,
                                           output_size=hidden_size*control_size, hidden_layers=hidden_layers)
        for layer in self.CDE_integrand_enc.model:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.bias, 0.)

        # Decoder
        self.CDE_integrand_dec = SequentialModel(input_size=hidden_size + 1,
                                                 output_size=hidden_size * treatment_size, hidden_layers=hidden_layers)
        for layer in self.CDE_integrand_dec.model:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.bias, 0.)

        # Prediction head
        self.y_net = MC_dropout_trainable_net(input_size=hidden_size, output_size=1,
                                              mc_samples=mc_samples, dropout_probability=dropout_probability)

        # Only used if intensity_weighting = True:
        if intensity_weighting:
            self.obs_net = nn.Linear(self.hidden_size, 1)

        # learning setup
        self.learning_rate = learning_rate
        self.coverage_metric = CoverageMetric(gaussian_mixture_components=False,
                                              confidence_level=coverage_metric_confidence)


    def forward_enc(self, x):
        # Required for control path interpolation and SDE integration
        ts = torch.linspace(0, 1, x.shape[1])
        control_path = self.interpolate(self.inter_coeffs(x, ts), ts)
        batch_size = x.shape[0]

        # Transform x0 to z0
        z0 = self.e_net(control_path.evaluate(0))

        # Create CDE instance based on current batch
        self.z_field = CDE_field(hidden_size=self.hidden_size, control_size=self.control_size,
                                 integrand=self.CDE_integrand_enc, control_path=control_path, batch_size=batch_size)


        # Integrate forward
        z1 = torchsde.sdeint(self, z0, ts, adaptive=True, method=self.method, options=dict(jump_t=ts))
        #z1 = torchsde.sdeint(self, torch.concatenate((torch.full((self.z_field.batch_size, 1), 0), z0), dim=-1),
        #                     names = {'drift': 'f_enc', 'diffusion': 'g_enc'}
        #                     ts, adaptive=True, method=self.method, options=dict(jump_t=ts))

        # Last observation time
        z1 = z1[-1:]
        return z1

    def forward_dec(self, x, z0):
        # Required for control path interpolation and SDE integration
        ts = torch.linspace(0, 1, x.shape[1])
        control_path = self.interpolate(self.inter_coeffs(x, ts), ts)
        batch_size = x.shape[0]

        # Create CDE instance based on current batch
        self.z_field = CDE_field(hidden_size=self.hidden_size, control_size=self.treatment_size,
                                 integrand=self.CDE_integrand_dec, control_path=control_path, batch_size=batch_size)

        # Integrate forward
        z1 = torchsde.sdeint(self, z0, ts, adaptive=True, method=self.method, options=dict(jump_t=ts))

        # Last observation time
        z1 = z1[-1:]
        return z1

    def f(self, t, z_state):
        # add time covariate
        return self.z_field(t, torch.concatenate((torch.full((self.z_field.batch_size, 1), t), z_state), dim=-1))

    def g(self, t, z_state):
        # 0 diffusion -> SDE reduces to ODE
        return torch.zeros(size=(self.z_field.batch_size, z_state[0].numel()))

    def _common_step(self, batch):
        x, y, tr_tau, active_entries = batch
        #data_X:        N_samples x (T - 5) x [cancer_volume, patient_type]
        #data_tr:       N_samples x (T - 5) x 4
        #data_y:        N_samples x (T - 5) x [1_step, 2_step, 3_step, 4_step, 5_step]
        #data_tr_tau:   N_samples x T x 4
        #data_ae:       N_samples x (T - 5) x [active_entries, active_entries_t+1, active_entries_t+2, active_entries_t+3, active_entries_t+4]

        y, active_entries = y[:, :, (self.tau - 1):self.tau], active_entries[:, :, (self.tau - 1):self.tau]
        y, active_entries = y.squeeze(dim=-1), active_entries.squeeze(dim=-1)

        # Multistep treatment assignments
        in_time = x.shape[1] - 1
        # For tau-step ahead, need grid of tau+1 treatment assignments
        tr_tau = tr_tau[:, (in_time):(in_time + self.tau), :]

        # Forward
        z1 = self.forward_enc(x)
        if self.tau > 1:
            z1 = self.forward_dec(tr_tau, z1.squeeze(dim=0))

        y_pred = self.y_net(z1).squeeze(-1).permute((0,2,1)) # mc, batch, time


        y = y[:, -1:]
        active_entries = active_entries[:, -1:]

        y_pred_unmasked, y_unmasked = y_pred.clone(), y.clone()
        # Only use active entries for loss calculation
        y_pred, y = y_pred * active_entries, y * active_entries
        # Mask missing values
        y_pred[:, y_unmasked.isnan()] = 0
        y[y_unmasked.isnan()] = 0

        # equivalent to MSE minimization
        MSE = (2*torch.pi)**0.5 * -dist.Normal(loc=y_pred, scale=1.).log_prob(y)[:, ~y_unmasked.isnan()]

        # TESAR variant
        if self.intensity_weighting:
            z1_detached = z1.detach() # need to detach from computational graph as in TESAR paper
            obs_logit = self.obs_net(z1_detached).squeeze(dim=0)  # squeeze out MC dim
            obs_loss = nn.BCEWithLogitsLoss(reduction='sum')(obs_logit, active_entries)
            obs_prob = torch.sigmoid(obs_logit[~y_unmasked.isnan()])
            obs_prob = torch.clip(obs_prob, min=0.001)  # clip for more stable training as in TESAR paper
            obs_prob = obs_prob.detach()
            weighted_MSE = (MSE / torch.sigmoid(obs_prob)).sum(dim=1)
            loss_per_MC_sample = 0.8 * weighted_MSE + 0.2 * obs_loss

        else:
            loss_per_MC_sample = MSE.sum(dim=1)

        loss = loss_per_MC_sample.mean()

        return loss, y_pred_unmasked, y_unmasked, x

    def training_step(self, batch):
        self.y_net.dropout.train()
        loss, y_pred, y, _ = self._common_step(batch)
        self.log_dict(
            {"train_loss": loss},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return {"loss": loss, "y_pred": y_pred, "y": y}

    def validation_step(self, batch, batch_idx):
        self.y_net.dropout.train()
        loss, y_pred, y, _ = self._common_step(batch)

        self.coverage_metric(y_pred.squeeze(dim=-1), y.squeeze(dim=-1))

        self.log_dict({"val_loss": loss, "coverage": self.coverage_metric, "size":size}, prog_bar=True)
        return loss


    def test_step(self, batch, batch_idx):
        self.y_net.dropout.train()
        loss, y_pred, y, _ = self._common_step(batch)
        self.coverage_metric(y_pred.squeeze(dim=-1), y.squeeze(dim=-1))
        size = self.coverage_metric.size()
        coverage = self.coverage_metric.compute()
        self.log_dict({"loss": loss, "coverage": coverage, "size": size}, prog_bar=True, logger=True)

        return {"loss": loss, "y_pred": y_pred, "y": y,
                "coverage": coverage, "size": size}


    def predict_step(self, batch, batch_idx, dataloader_idx=None):

        self.y_net.dropout.train()
        loss, y_pred, y, x = self._common_step(batch)
        self.coverage_metric(y_pred.squeeze(dim=-1), y.squeeze(dim=-1))  # abuse former time dimension for concatenation
        size = self.coverage_metric.size()
        coverage = self.coverage_metric.compute()

        return {"loss": loss, "y_pred": y_pred, "y": y, "x": x,
                "coverage": coverage, "size": size}


    def configure_optimizers(self):

            params = [self.CDE_integrand_enc.parameters(),
                      self.e_net.parameters(),
                      self.y_net.layers.parameters()
                      ]
            learning_rates = [self.learning_rate, self.learning_rate * 1e1, self.learning_rate * 1e1]

            if self.tau > 1:
                params += [self.CDE_integrand_dec.parameters()]
                learning_rates +=[self.learning_rate]

            if self.intensity_weighting:
                params += [self.obs_net.parameters()]
                learning_rates += [self.learning_rate * 1e1]

            param_groups = [{'params': param_group, 'lr': lr} for param_group, lr in zip(params, learning_rates)]

            return optim.Adam(param_groups)


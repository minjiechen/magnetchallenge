import torch
import torch.nn as nn


class Biased_Elu(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1


class SinusAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class GeneralizedCosinusUnit(nn.Module):
    def forward(self, x):
        return torch.cos(x) * x


ACTIVATION_FUNCS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "biased_elu": Biased_Elu,
    "sinus": SinusAct,
    "gcu": GeneralizedCosinusUnit,
}


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        residual=True,
        double_layered=True,
        dropout=0.0,
        act_func=None,
    ):
        super(TemporalBlock, self).__init__()
        padding = ((kernel_size - 1) // 2) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode="circular",
            )
        )
        self.relu1 = ACTIVATION_FUNCS.get(act_func, nn.Identity)()
        self.dropout1 = nn.Dropout1d(dropout)
        if double_layered:
            self.relu2 = nn.Identity()
            self.conv2 = nn.utils.weight_norm(
                nn.Conv1d(
                    n_inputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    padding_mode="circular",
                )
            )
            self.dropout2 = nn.Dropout1d(dropout)
            self.net = nn.Sequential(
                self.conv1,
                self.relu1,
                self.dropout1,
                self.conv2,
                self.relu2,
                self.dropout2,
            )
        else:
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.relu = nn.ReLU()
        self.residual = residual
        if residual:
            self.downsample = (
                nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            )
        else:
            self.downsample = None
        self.double_layered = double_layered
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.double_layered:
            self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            res = x if self.downsample is None else self.downsample(x)
            y = torch.clip(
                out + res, -10, 10
            )  # out += res is not allowed, it would become an inplace op, weird
            y = self.relu(y)
        else:
            y = out
        return y


class TCNWithScalarsAsBias(nn.Module):
    def __init__(
        self,
        num_input_scalars,
        num_input_ts=1,
        tcn_layer_cfg=None,
        scalar_layer_cfg=None,
    ):
        super().__init__()
        self.num_input_ts = num_input_ts
        self.num_input_scalar = num_input_scalars
        tcn_layer_cfg = tcn_layer_cfg or {
            "f": [
                {"units": (num_input_scalars + 1), "act_func": "tanh"},
                {"units": 8, "act_func": "tanh"},
                {"units": 1},
            ]
        }
        scalar_layer_cfg = scalar_layer_cfg or {
            "f": [
                {"units": num_input_scalars, "act_func": "tanh"},
                # {'units': 8, 'act_func': 'tanh'}
            ]
        }
        # build CNN layer path
        cnn_layers = []
        dilation_offset = tcn_layer_cfg.get("starting_dilation_rate", 2)  # >= 0
        for i, l_cfg in enumerate(tcn_layer_cfg["f"]):
            kernel_size = l_cfg.get("kernel_size", 9)
            dropout_rate = tcn_layer_cfg.get("dropout", 0.0)
            dilation_size = 2 ** (i + dilation_offset)
            if i == 0:
                in_channels = num_input_ts  # TCN acts only on B field(s) in first layer
            else:
                in_channels = tcn_layer_cfg["f"][i - 1]["units"]
            cnn_layers += [
                TemporalBlock(
                    in_channels,
                    l_cfg["units"],
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    residual=tcn_layer_cfg.get("residual", False),
                    double_layered=tcn_layer_cfg.get("double_layered", False),
                    dropout=dropout_rate,
                    act_func=l_cfg.get("act_func", nn.Identity),
                ),
            ]
            if i == 0:
                self.ts_branch = cnn_layers.pop()
        self.upper_tcn = nn.Sequential(*cnn_layers)
        # build scalar NN path
        scalar_layers = []
        fan_in = num_input_scalars
        for i, l_cfg in enumerate(scalar_layer_cfg["f"]):
            scalar_layers.append(nn.Linear(fan_in, l_cfg["units"]))
            scalar_layers.append(ACTIVATION_FUNCS.get(l_cfg["act_func"], nn.Identity)())
            fan_in = l_cfg["units"]

        self.scalar_branch = nn.Sequential(*scalar_layers)

    def __prepare_scriptable__(self):
        """This function removes the WeightNorm layer when jit scripting the nn module. This is necessary, since
        torch.jit.script would throw an error for any model possessing the weightnorm layer. This layer
        is only necessary for efficient training, and since jit scripted models are for inference only, the
        weight norm layer can be safely removed.

        Source:
        https://github.com/pytorch/audio/blob/8690e6ec9b419299076ef333d9d25d1f59fe92b1/torchaudio/models/wav2vec2/components.py#L214-L223
        """
        for hook in self.ts_branch.conv1._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.ts_branch.conv1)
        for cnn_lay in self.upper_tcn:
            for hook in cnn_lay.conv1._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(cnn_lay.conv1)
        return self

    def forward(self, x_ts, x_scalars):
        """x_ts has shape (#batch, #channels, #length) and x_scalars has (#batch, #channels)"""
        b_proc = self.ts_branch(x_ts)
        scalar_proc = self.scalar_branch(x_scalars)
        catted = torch.cat(
            [
                b_proc[:, : -scalar_proc.shape[1], :],
                b_proc[:, -scalar_proc.shape[1] :, :] + scalar_proc.unsqueeze(-1),
            ],
            dim=1,
        )
        y = self.upper_tcn(catted)
        y = y + x_ts[:, [0], :]  # residual connection to per-profile scaled B curve

        # subtract mean along time domain
        y = y - y.mean(dim=-1).unsqueeze(-1)
        return y


class LossPredictor(nn.Module):
    """The post processor replaces the 0.5 factor during the shoelace formula in this topology.
    It shall capture the difference between the power loss scalar and the polygon area between
    B and H."""

    def __init__(
        self,
        h_predictor,
    ):
        super().__init__()
        self.h_predictor = h_predictor
        self.post_processor = nn.Sequential(
            nn.Linear(self.h_predictor.num_input_scalar, 8), 
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Tanh()
        )

    def forward(self, x_ts, x_scalars, b_lim, h_lim, freq_scale):
        h_pred = self.h_predictor(x_ts, x_scalars).permute(2, 0, 1)
        freq = freq_scale * torch.exp(x_scalars[:, [0]])
        scaled_b = x_ts[:, [-1], :].permute(2, 0, 1)  # globally scaled B curve
        b_with_offset = b_lim * scaled_b + 5  # arbitrary offset
        h_with_offset = h_lim * h_pred + 5  # arbitrary offset
        ploss_pred = (
            freq
            * (0.5 + 0.1*self.post_processor(x_scalars))
            * torch.abs(
                torch.sum(
                    b_with_offset
                    * (
                        torch.roll(h_with_offset, 1, dims=0)
                        - torch.roll(h_with_offset, -1, dims=0)
                    ),
                    dim=0,
                )
            )
        )  # shoelace formula, polygon area
        return torch.log(ploss_pred), h_pred

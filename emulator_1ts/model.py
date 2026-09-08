import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Optional, Tuple
from .scalers import load_scalers, resolve_pressure_scaler_names

def get_model(
    model_name,
    model_def
):
    if model_name == 'resnet':
        return ResNet(**model_def)
    if model_name == 'convnext':
        return ConvNeXT(**model_def)
    if model_name == 'convnext_unet':
        return ConvNeXTUNet(**model_def)
    else:
        raise ValueError(f'Model {model_name} not found')

class LayerNorm2D(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


class GeGLU(nn.Module):
    """Gated GELU activation over pairs of channel projections."""

    projection_multiplier = 2

    def forward(self, x):
        values, gates = x.chunk(2, dim=1)
        return values * F.gelu(gates)


def activation_projection_multiplier(activation):
    """Number of projected channels required by an activation."""

    return getattr(activation, "projection_multiplier", 1)


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
        padding_mode='zeros',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation()
        self.padding = int(kernel_size / 2)
        projection_multiplier = activation_projection_multiplier(activation)
        projected_channels = self.out_channels * projection_multiplier
        self.conv = nn.Conv2d(
            self.in_channels,
            projected_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            padding_mode=padding_mode
        )
        # GeGLU multiplies two projected branches. Without a normalization
        # before that product, repeated encoder/decoder projections square the
        # activation scale and overflow even on the first forward pass.
        self.projection_norm = (
            LayerNorm2D(projected_channels)
            if projection_multiplier == 2
            else nn.Identity()
        )

    def forward(self, x):
        return self.activation(self.projection_norm(self.conv(x)))


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        activation,
        padding_mode='zeros',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.activation = activation()
        self.gated_activation = activation_projection_multiplier(activation) == 2
        self.depthwise_conv = nn.Conv2d(
            self.in_channels,
            self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size / 2),
            padding_mode=padding_mode,
            groups=self.in_channels
        )
        self.layer_norm = LayerNorm2D(self.hidden_channels)
        self.pointwise_conv = nn.Conv2d(
            self.hidden_channels,
            self.out_channels * activation_projection_multiplier(activation),
            kernel_size=1
        )

    def forward(self, x):
        identity = x
        out = self.depthwise_conv(x)
        out = self.layer_norm(out)
        out = self.pointwise_conv(out)
        if self.gated_activation:
            out = self.activation(out)
            out = out + identity
        else:
            out = self.activation(out + identity)
        return out



class ResNet(torch.nn.Module):

    # Declared so TorchScript still types the attribute when a scaler set
    # covers no pressure layers and the list is empty.
    pressure_scaler_names: List[str]

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=64,
        kernel_size=5,
        depth=1,
        activation=GeGLU,
        scalers=None,
        pressure_names=None,
        evaptrans_names=None,
        param_names=None,
        n_evaptrans=None,
        parameter_list=None,
        param_nlayer=None,
        padding_mode='zeros',
    ):
        super().__init__()
        self.input_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.activation = activation
        # ``scalers`` may be a mapping, a path to a scaler YAML, or None for the
        # packaged CONUS2.1 statistics, so a config can select a basin-specific
        # set with ``model_def.scalers``.
        self.scalers = load_scalers(scalers)
        self.pressure_names = pressure_names
        # Settle the layer -> scaler-key mapping once at construction. This
        # keeps the scaling loops free of string formatting and confines the
        # legacy ``press_diff_*`` key handling to one place.
        self.pressure_scaler_names = resolve_pressure_scaler_names(
            self.scalers, pressure_names
        )
        self.evaptrans_names = evaptrans_names
        self.n_evaptrans = n_evaptrans
        self.param_names = param_names
        self.parameter_list = parameter_list
        self.param_nlayer = param_nlayer
        self.padding_mode = padding_mode
        
        self.layers = [
            ConvBlock(
                self.input_channels,
                self.hidden_dim,
                self.kernel_size,
                self.activation,
                self.padding_mode,
            )
        ]
        for i in range(self.depth):
            self.layers.append(
                ResidualBlock(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    self.kernel_size,
                    self.activation,
                    self.padding_mode,
                )
            )
        self.layers.append(
            ConvBlock(
                self.hidden_dim,
                self.output_channels,
                1,
                nn.Identity,
                self.padding_mode,
            )
        )
        self.layers = nn.ModuleList(self.layers)

    @torch.jit.export
    def get_parflow_pressure(self, pressure):
        # ``unsqueeze`` returns a view onto the caller's storage, so scaling in
        # place here would rewrite ParFlow's own tensor. Copy first.
        pressure = pressure.unsqueeze(0).clone()
        self.scale_pressure(pressure)
        return pressure

    @torch.jit.export
    def scale_pressure(self, x):
        # Dims are (batch, z, y, x)
        if len(self.pressure_scaler_names) < x.shape[1]:
            raise ValueError(
                "Scalers cover fewer pressure layers than the tensor has"
            )
        for i in range(x.shape[1]):
            mu = self.scalers[self.pressure_scaler_names[i]][0]
            sigma = self.scalers[self.pressure_scaler_names[i]][1]
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma

    @torch.jit.export
    def unscale_pressure(self, x):
        # Dims are (batch, z, y, x)
        if len(self.pressure_scaler_names) < x.shape[1]:
            raise ValueError(
                "Scalers cover fewer pressure layers than the tensor has"
            )
        for i in range(x.shape[1]):
            mu = self.scalers[self.pressure_scaler_names[i]][0]
            sigma = self.scalers[self.pressure_scaler_names[i]][1]
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu

    @torch.jit.export
    def get_predicted_pressure(self, x):
        # Copy so the caller keeps its scaled prediction and a second call
        # cannot unscale the same tensor twice.
        x = x.clone()
        self.unscale_pressure(x)
        return x.squeeze()

    @torch.jit.export
    def get_parflow_evaptrans(self, evaptrans):
        if self.n_evaptrans > 0:
            evaptrans = evaptrans[0:self.n_evaptrans,:,:]
        #Grab the top n_lay layers
        elif self.n_evaptrans < 0:
            evaptrans = evaptrans[self.n_evaptrans:,:,:]
        # Slicing and unsqueezing both return views of the caller's tensor.
        evaptrans = evaptrans.unsqueeze(0).clone()
        self.scale_evaptrans(evaptrans)
        return evaptrans
    
    @torch.jit.export
    def scale_evaptrans(self, x):
        # Dims are (batch, z, y, x)
        for i, name in enumerate(self.evaptrans_names):
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma
        
    @torch.jit.export
    def unscale_evaptrans(self, x):
        # Dims are (batch, z, y, x)
        for i, name in enumerate(self.evaptrans_names):
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu

    @torch.jit.export
    def get_parflow_statics(self, statics:Dict[str, torch.Tensor]):
        parameter_data = []
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            param_temp = statics[parameter]
            if param_temp.shape[0] > 1:
                #Grab the top n bottom or top layers if specified in the param_nlayer list
                #Grab the bottom n_lay layers
                if n_lay > 0:
                    param_temp = param_temp[0:n_lay,:,:]
                #Grab the top n_lay layers
                elif n_lay < 0:
                    param_temp = param_temp[n_lay:,:,:]
            parameter_data.append(param_temp)

        # Concatenate the parameter data together
        # End result is a dims of (n_parameters, y, x). ``cat`` allocates, so
        # unlike the pressure and evaptrans paths this cannot alias the
        # caller's statics.
        parameter_data = torch.cat(parameter_data, dim=0)
        parameter_data = parameter_data.unsqueeze(0)
        self.scale_statics(parameter_data)
        return parameter_data
            
    @torch.jit.export
    def scale_statics(self, x):
        for i, name in enumerate(self.param_names):
            if name == 'mask':
                continue
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma
    
    @torch.jit.export
    def unscale_statics(self, x):
        for i, name in enumerate(self.param_names):
            if name == 'mask':
                continue
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu

    def forward(self, pressure, evaptrans, statics):
        # Concatenate the data
        x = torch.cat([pressure, evaptrans, statics], dim=1)

        for l in self.layers:
            x = l(x)

        return x
    
    def forward_autoregressive(self, initial_pressure, evaptrans_sequence, statics):
        """
        Autoregressive forward pass for multi-timestep prediction.
        
        Args:
            initial_pressure: Initial pressure state [batch, z, y, x]
            evaptrans_sequence: Evapotranspiration sequence [n_timesteps, batch, z, y, x]
            statics: Static parameters [batch, n_params, y, x]
            
        Returns:
            predictions: Sequence of predicted pressure states [n_timesteps, batch, z, y, x]
        """
        batch_size = initial_pressure.shape[0]
        n_timesteps = evaptrans_sequence.shape[0]
        predictions = []
        
        # Start with initial pressure state
        current_state = initial_pressure
        
        for t in range(n_timesteps):
            # Get evapotranspiration for current timestep
            current_evaptrans = evaptrans_sequence[t]
            
            # Predict next state
            next_state = self.forward(current_state, current_evaptrans, statics)
            predictions.append(next_state)
            
            # Use prediction as input for next timestep
            current_state = next_state
            
        return torch.stack(predictions)


class ConvNeXTBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        activation,
        padding_mode='zeros',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation()
        self.padding = int(kernel_size / 2)
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.mid_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            padding_mode=padding_mode
        )
        self.conv2 = nn.Conv2d(
            self.mid_channels,
            self.mid_channels * activation_projection_multiplier(activation),
            kernel_size=1
        )
        self.conv3 = nn.Conv2d(
            self.mid_channels,
            self.out_channels,
            kernel_size=1,
        )
        self.layer_norm = LayerNorm2D(self.mid_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.layer_norm(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out += identity
        return out


class ConvNeXT(torch.nn.Module):

    # See ResNet: keeps the empty-list case scriptable.
    pressure_scaler_names: List[str]

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_dim=64,
        hidden_dim=256,
        kernel_size=5,
        depth=1,
        activation=GeGLU,
        scalers=None,
        pressure_names=None,
        evaptrans_names=None,
        param_names=None,
        n_evaptrans=None,
        parameter_list=None,
        param_nlayer=None,
        noise_scale=1e-6,
        padding_mode='zeros',
    ):
        super().__init__()
        self.input_channels = in_channels
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.activation = activation
        # ``scalers`` may be a mapping, a path to a scaler YAML, or None for the
        # packaged CONUS2.1 statistics, so a config can select a basin-specific
        # set with ``model_def.scalers``.
        self.scalers = load_scalers(scalers)
        self.pressure_names = pressure_names
        # Settle the layer -> scaler-key mapping once at construction. This
        # keeps the scaling loops free of string formatting and confines the
        # legacy ``press_diff_*`` key handling to one place.
        self.pressure_scaler_names = resolve_pressure_scaler_names(
            self.scalers, pressure_names
        )
        self.evaptrans_names = evaptrans_names
        self.n_evaptrans = n_evaptrans
        self.param_names = param_names
        self.parameter_list = parameter_list
        self.param_nlayer = param_nlayer
        self.noise_scale = noise_scale
        self.padding_mode = padding_mode

        self.layers = [
            ConvBlock(
                self.input_channels,
                self.hidden_dim,
                1,
                self.activation,
                self.padding_mode,
            )
        ]
        for i in range(self.depth):
            self.layers.append(
                ConvNeXTBlock(
                    self.hidden_dim,
                    self.bottleneck_dim,
                    self.hidden_dim,
                    self.kernel_size,
                    self.activation,
                    self.padding_mode,
                )
            )
        self.layers.append(
            ConvBlock(
                self.hidden_dim,
                self.output_channels,
                1,
                nn.Identity,
                self.padding_mode,
            )
        )
        self.layers = nn.ModuleList(self.layers)

    @torch.jit.export
    def get_parflow_pressure(self, pressure):
        # ``unsqueeze`` returns a view onto the caller's storage, so scaling in
        # place here would rewrite ParFlow's own tensor. Copy first.
        pressure = pressure.unsqueeze(0).clone()
        self.scale_pressure(pressure)
        return pressure

    @torch.jit.export
    def scale_pressure(self, x):
        # Dims are (batch, z, y, x)
        if len(self.pressure_scaler_names) < x.shape[1]:
            raise ValueError(
                "Scalers cover fewer pressure layers than the tensor has"
            )
        for i in range(x.shape[1]):
            mu = self.scalers[self.pressure_scaler_names[i]][0]
            sigma = self.scalers[self.pressure_scaler_names[i]][1]
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma

    @torch.jit.export
    def unscale_pressure(self, x):
        # Dims are (batch, z, y, x)
        if len(self.pressure_scaler_names) < x.shape[1]:
            raise ValueError(
                "Scalers cover fewer pressure layers than the tensor has"
            )
        for i in range(x.shape[1]):
            mu = self.scalers[self.pressure_scaler_names[i]][0]
            sigma = self.scalers[self.pressure_scaler_names[i]][1]
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu

    @torch.jit.export
    def get_predicted_pressure(self, x):
        # Copy so the caller keeps its scaled prediction and a second call
        # cannot unscale the same tensor twice.
        x = x.clone()
        self.unscale_pressure(x)
        return x.squeeze()

    @torch.jit.export
    def get_parflow_evaptrans(self, evaptrans):
        if self.n_evaptrans > 0:
            evaptrans = evaptrans[0:self.n_evaptrans,:,:]
        #Grab the top n_lay layers
        elif self.n_evaptrans < 0:
            evaptrans = evaptrans[self.n_evaptrans:,:,:]
        # Slicing and unsqueezing both return views of the caller's tensor.
        evaptrans = evaptrans.unsqueeze(0).clone()
        self.scale_evaptrans(evaptrans)
        return evaptrans
    
    @torch.jit.export
    def scale_evaptrans(self, x):
        # Dims are (batch, z, y, x)
        for i, name in enumerate(self.evaptrans_names):
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma
        
    @torch.jit.export
    def unscale_evaptrans(self, x):
        # Dims are (batch, z, y, x)
        for i, name in enumerate(self.evaptrans_names):
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu

    @torch.jit.export
    def get_parflow_statics(self, statics:Dict[str, torch.Tensor]):
        parameter_data = []
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            param_temp = statics[parameter]
            if param_temp.shape[0] > 1:
                #Grab the top n bottom or top layers if specified in the param_nlayer list
                #Grab the bottom n_lay layers
                if n_lay > 0:
                    param_temp = param_temp[0:n_lay,:,:]
                #Grab the top n_lay layers
                elif n_lay < 0:
                    param_temp = param_temp[n_lay:,:,:]
            parameter_data.append(param_temp)

        # Concatenate the parameter data together
        # End result is a dims of (n_parameters, y, x). ``cat`` allocates, so
        # unlike the pressure and evaptrans paths this cannot alias the
        # caller's statics.
        parameter_data = torch.cat(parameter_data, dim=0)
        parameter_data = parameter_data.unsqueeze(0)
        self.scale_statics(parameter_data)
        return parameter_data
            
    @torch.jit.export
    def scale_statics(self, x):
        for i, name in enumerate(self.param_names):
            if name == 'mask':
                continue
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma
    
    @torch.jit.export
    def unscale_statics(self, x):
        for i, name in enumerate(self.param_names):
            if name == 'mask':
                continue
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu

    def forward(self, pressure, evaptrans, statics):
        # Concatenate the data
        x = torch.cat([pressure, evaptrans, statics], dim=1)
        # Add small noise to the input to help with stability
        if self.training:
            x = x + torch.randn_like(x) * self.noise_scale

        for l in self.layers:
            x = l(x)

        return pressure + x

    def forward_autoregressive(self, initial_pressure, evaptrans_sequence, statics):
        """
        Autoregressive forward pass for multi-timestep prediction.
        
        Args:
            initial_pressure: Initial pressure state [batch, z, y, x]
            evaptrans_sequence: Evapotranspiration sequence [n_timesteps, batch, z, y, x]
            statics: Static parameters [batch, n_params, y, x]
            
        Returns:
            predictions: Sequence of predicted pressure states [n_timesteps, batch, z, y, x]
        """
        batch_size = initial_pressure.shape[0]
        n_timesteps = evaptrans_sequence.shape[0]
        predictions = []
        
        # Start with initial pressure state
        current_state = initial_pressure
        
        for t in range(n_timesteps):
            # Get evapotranspiration for current timestep
            current_evaptrans = evaptrans_sequence[t]
            
            # Predict next state
            next_state = self.forward(current_state, current_evaptrans, statics)
            predictions.append(next_state)
            
            # Use prediction as input for next timestep
            current_state = next_state
            
        return torch.stack(predictions)


def plan_downsample_factors(
    input_height: int,
    input_width: int,
    max_levels: int,
    min_coarse_cells: int,
    mode: str = "auto",
    manual_factors: Optional[List[List[int]]] = None,
) -> List[Tuple[int, int]]:
    """Resolve a deterministic, aspect-ratio-aware U-Net pooling schedule.

    Each axis is halved only while doing so leaves at least
    ``min_coarse_cells`` along that axis. Once the shorter axis reaches the
    minimum, only the longer axis continues to be reduced.
    """

    if input_height < 1 or input_width < 1:
        raise ValueError("input_height and input_width must be positive")
    if max_levels < 1:
        raise ValueError("max_levels must be at least 1")
    if min_coarse_cells < 1:
        raise ValueError("min_coarse_cells must be at least 1")
    if mode not in ("auto", "manual"):
        raise ValueError("downsample_mode must be 'auto' or 'manual'")

    if mode == "manual":
        if manual_factors is None or len(manual_factors) != max_levels:
            raise ValueError(
                "manual downsampling requires one factor per U-Net level"
            )
        schedule = []
        for factor in manual_factors:
            if len(factor) != 2 or factor[0] not in (1, 2) or factor[1] not in (1, 2):
                raise ValueError("downsample factors must be [1 or 2, 1 or 2]")
            schedule.append((int(factor[0]), int(factor[1])))
    else:
        schedule = []
        height, width = input_height, input_width
        for _ in range(max_levels):
            factor_y = 2 if height // 2 >= min_coarse_cells else 1
            factor_x = 2 if width // 2 >= min_coarse_cells else 1
            schedule.append((factor_y, factor_x))
            height //= factor_y
            width //= factor_x

    # Validate manual schedules against the same minimum-size guarantee.
    height, width = input_height, input_width
    for factor_y, factor_x in schedule:
        next_height = height // factor_y
        next_width = width // factor_x
        if next_height < min_coarse_cells or next_width < min_coarse_cells:
            raise ValueError(
                "downsampling schedule would reduce a feature-map axis below "
                f"min_coarse_cells={min_coarse_cells}: "
                f"{height}x{width} -> {next_height}x{next_width}"
            )
        height, width = next_height, next_width
    return schedule


class MaskedConvNeXTStage(nn.Module):
    """ConvNeXT blocks that keep inactive basin cells neutral."""

    def __init__(
        self,
        channels: int,
        n_blocks: int,
        kernel_size: int,
        bottleneck_ratio: float,
        activation,
        padding_mode: str,
    ):
        super().__init__()
        mid_channels = max(1, int(round(channels * bottleneck_ratio)))
        self.blocks = nn.ModuleList(
            [
                ConvNeXTBlock(
                    channels,
                    mid_channels,
                    channels,
                    kernel_size,
                    activation,
                    padding_mode,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x)
            x = x * mask
        return x


class MaskedDownsample(nn.Module):
    """Mask-normalized pooling followed by a learned channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: Tuple[int, int],
        activation,
    ):
        super().__init__()
        self.factor_y = factor[0]
        self.factor_x = factor[1]
        self.projection = ConvBlock(
            in_channels, out_channels, 1, activation, padding_mode="zeros"
        )

    def forward(self, x, mask):
        kernel = [self.factor_y, self.factor_x]
        pooled_mask = F.avg_pool2d(mask, kernel_size=kernel, stride=kernel)
        pooled_values = F.avg_pool2d(
            x * mask, kernel_size=kernel, stride=kernel
        )
        x = pooled_values / pooled_mask.clamp_min(1.0e-8)
        mask = (pooled_mask > 0).to(dtype=x.dtype)
        x = self.projection(x) * mask
        return x, mask


class MaskedEncoderLevel(nn.Module):
    """One encoder stage paired with its fixed downsampling operation."""

    def __init__(self, stage: MaskedConvNeXTStage, downsample: MaskedDownsample):
        super().__init__()
        self.stage = stage
        self.downsample = downsample

    def forward(self, x, mask):
        skip = self.stage(x, mask)
        x, coarse_mask = self.downsample(skip, mask)
        return x, coarse_mask, skip


class MaskedDecoderStage(nn.Module):
    """Shape-safe U-Net upsampling, skip fusion, and masked ConvNeXT blocks."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        n_blocks: int,
        kernel_size: int,
        bottleneck_ratio: float,
        activation,
        padding_mode: str,
    ):
        super().__init__()
        self.fusion = ConvBlock(
            in_channels + skip_channels,
            skip_channels,
            1,
            activation,
            padding_mode="zeros",
        )
        self.stage = MaskedConvNeXTStage(
            skip_channels,
            n_blocks,
            kernel_size,
            bottleneck_ratio,
            activation,
            padding_mode,
        )

    def forward(self, x, skip, mask):
        x = F.interpolate(
            x,
            size=[skip.shape[-2], skip.shape[-1]],
            mode="bilinear",
            align_corners=False,
        )
        x = self.fusion(torch.cat([x, skip], dim=1)) * mask
        return self.stage(x, mask)


class ConvNeXTUNet(ConvNeXT):
    """Mask-aware multiscale ConvNeXT emulator with anisotropic pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        input_width: int,
        base_channels: int = 64,
        channel_multipliers: Optional[List[int]] = None,
        blocks_per_level: Optional[List[int]] = None,
        max_levels: int = 3,
        min_coarse_cells: int = 8,
        downsample_mode: str = "auto",
        downsample_factors: Optional[List[List[int]]] = None,
        bottleneck_ratio: float = 0.5,
        kernel_size: int = 3,
        activation=GeGLU,
        scalers=None,
        pressure_names=None,
        evaptrans_names=None,
        param_names=None,
        n_evaptrans=None,
        parameter_list=None,
        param_nlayer=None,
        noise_scale: float = 1.0e-6,
        padding_mode: str = "replicate",
    ):
        # ConvNeXT supplies the ParFlow scaling and autoregressive helper
        # methods. Its flat network is intentionally not constructed here.
        nn.Module.__init__(self)
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]
        if blocks_per_level is None:
            blocks_per_level = [1] * (max_levels + 1)
        if len(channel_multipliers) != max_levels + 1:
            raise ValueError("channel_multipliers must have max_levels + 1 entries")
        if len(blocks_per_level) != max_levels + 1:
            raise ValueError("blocks_per_level must have max_levels + 1 entries")
        if any(value < 1 for value in channel_multipliers):
            raise ValueError("channel multipliers must be positive")
        if any(value < 1 for value in blocks_per_level):
            raise ValueError("each U-Net level must contain at least one block")
        if bottleneck_ratio <= 0:
            raise ValueError("bottleneck_ratio must be positive")
        if param_names is None or "mask" not in param_names:
            raise ValueError("convnext_unet requires the binary 'mask' static channel")

        self.input_channels = in_channels
        self.output_channels = out_channels
        self.input_height = input_height
        self.input_width = input_width
        self.max_levels = max_levels
        self.min_coarse_cells = min_coarse_cells
        self.kernel_size = kernel_size
        self.activation = activation
        # ``scalers`` may be a mapping, a path to a scaler YAML, or None for the
        # packaged CONUS2.1 statistics, so a config can select a basin-specific
        # set with ``model_def.scalers``.
        self.scalers = load_scalers(scalers)
        self.pressure_names = pressure_names
        # Settle the layer -> scaler-key mapping once at construction. This
        # keeps the scaling loops free of string formatting and confines the
        # legacy ``press_diff_*`` key handling to one place.
        self.pressure_scaler_names = resolve_pressure_scaler_names(
            self.scalers, pressure_names
        )
        self.evaptrans_names = evaptrans_names
        self.n_evaptrans = n_evaptrans
        self.param_names = param_names
        self.parameter_list = parameter_list
        self.param_nlayer = param_nlayer
        self.noise_scale = noise_scale
        self.padding_mode = padding_mode
        self.mask_channel = param_names.index("mask")

        schedule = plan_downsample_factors(
            input_height,
            input_width,
            max_levels,
            min_coarse_cells,
            mode=downsample_mode,
            manual_factors=downsample_factors,
        )
        self.downsample_factor_y = [factor[0] for factor in schedule]
        self.downsample_factor_x = [factor[1] for factor in schedule]

        channels = [base_channels * value for value in channel_multipliers]
        self.stem = ConvBlock(
            in_channels, channels[0], 1, activation, padding_mode="zeros"
        )
        encoder_stages = [
            MaskedConvNeXTStage(
                channels[level],
                blocks_per_level[level],
                kernel_size,
                bottleneck_ratio,
                activation,
                padding_mode,
            )
            for level in range(max_levels)
        ]
        downsamplers = [
            MaskedDownsample(
                channels[level],
                channels[level + 1],
                schedule[level],
                activation,
            )
            for level in range(max_levels)
        ]
        self.encoder_levels = nn.ModuleList(
            [
                MaskedEncoderLevel(stage, downsample)
                for stage, downsample in zip(encoder_stages, downsamplers)
            ]
        )
        self.bottleneck_stage = MaskedConvNeXTStage(
            channels[max_levels],
            blocks_per_level[max_levels],
            kernel_size,
            bottleneck_ratio,
            activation,
            padding_mode,
        )
        self.decoder_stages = nn.ModuleList(
            [
                MaskedDecoderStage(
                    channels[level + 1],
                    channels[level],
                    blocks_per_level[level],
                    kernel_size,
                    bottleneck_ratio,
                    activation,
                    padding_mode,
                )
                for level in range(max_levels - 1, -1, -1)
            ]
        )
        self.output_projection = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, pressure, evaptrans, statics):
        mask = (statics[:, self.mask_channel:self.mask_channel + 1] > 0.5).to(
            dtype=pressure.dtype
        )
        dynamic_inputs = torch.cat([pressure, evaptrans], dim=1)
        if self.training:
            dynamic_inputs = dynamic_inputs + (
                torch.randn_like(dynamic_inputs) * self.noise_scale
            )
        x = torch.cat([dynamic_inputs, statics], dim=1)
        x = self.stem(x) * mask

        skip_features = torch.jit.annotate(List[torch.Tensor], [])
        skip_masks = torch.jit.annotate(List[torch.Tensor], [])
        for encoder in self.encoder_levels:
            skip_masks.append(mask)
            x, mask, skip = encoder(x, mask)
            skip_features.append(skip)

        x = self.bottleneck_stage(x, mask)
        for decoder_index, decoder in enumerate(self.decoder_stages):
            skip_index = self.max_levels - decoder_index - 1
            x = decoder(
                x, skip_features[skip_index], skip_masks[skip_index]
            )

        full_mask = skip_masks[0]
        delta = self.output_projection(x) * full_mask
        return (pressure + delta) * full_mask

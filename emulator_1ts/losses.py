"""Loss functions that target the quantity the ParFlow solver responds to.

Plain pressure MSE is a poor training objective for a nonlinear-solver first
guess.  Benchmarking showed the emulator reaching a pressure RMSE of 1.3e-3
while *raising* the Richards residual KINSOL has to reduce by a median factor
of 2.8, so the model lost all 168 gated timesteps.  Two properties of the
residual explain most of that gap and are cheap to put into the loss:

``vertical_weight``
    The residual depends on vertical pressure *differences*, not pointwise
    pressure.  A prediction can track pressure closely and still corrupt the
    layer-to-layer gradient that sets the vertical Darcy flux.

``tail_weight``
    The residual is dominated by the worst cells, while a mean-squared error
    is dominated by the bulk.  Paired runs showed a max absolute difference of
    0.057 against an RMSE of 1.3e-3, a factor of 40.

``ponding_weight``
    The sign of the top-layer pressure selects the overland-flow branch of
    the residual.  Cells near p=0 are invisible to MSE (errors of ~1e-3 m
    against a scaled sigma of 0.45), but the gated benchmark showed ~90
    surface cells per hour predicted on the dry side of ponded truth, which
    stalled KINSOL with 3-5 backtracking Newton iterations every timestep.
    A hinge on the ponding state keeps a constant gradient until the cell
    crosses to the correct side, which MSE cannot provide as the error
    shrinks.

All terms are applied only to valid ParFlow cells.  See ``StructuredLoss`` for
the calling convention that lets these coexist with plain ``nn.MSELoss``.
"""

import torch
from torch import nn


class StructuredLoss(nn.Module):
    """Base class for losses that need spatial structure.

    The default path in ``train.calculate_masked_loss`` selects valid cells with
    ``predictions[valid_mask]``, which flattens the tensor and destroys the
    z/y/x layout.  Any loss deriving from this class is instead handed the full
    ``(batch, z, y, x)`` tensors plus the mask, so it can difference neighbours
    before reducing.
    """

    def forward(self, predictions, targets, valid_mask):
        raise NotImplementedError


class PressureFirstGuessLoss(StructuredLoss):
    """Pointwise error, optionally augmented with tail and vertical terms.

    Args:
        base: ``"mse"`` or ``"mae"`` for the pointwise term.
        vertical_weight: Weight on the vertical pressure-gradient error.
        tail_weight: Weight on the mean squared error of the worst cells.
        tail_fraction: Fraction of valid cells counted as the tail.
        ponding_weight: Weight on the top-layer ponding-state hinge.
        ponding_margin: Extra scaled-unit margin the prediction must clear on
            the correct side of p=0 before the hinge releases.  Zero keeps the
            hinge from distorting genuinely near-zero pressures; only raise it
            if flips persist after the plain hinge converges.
        top_zero_scaled: Scaled-space value of physical zero pressure for the
            top layer, ``-mu/sigma`` of its scaler.  Required when
            ``ponding_weight`` is positive because the tensors arrive in
            scaled units, where the overland switch is not at zero.
        layer_sigmas: Per-layer pressure standard deviations used to train the
            scalers.  Predictions and targets arrive in scaled units with a
            different sigma per layer (they span roughly 1.1 to 8.6 here), so
            differencing adjacent layers directly would compare inconsistent
            units.  Supplying these recovers the physical gradient.  They are
            normalized to mean 1, which keeps the term's magnitude comparable
            to the pointwise term while preserving relative layer weighting.
    """

    def __init__(
        self,
        base: str = "mse",
        vertical_weight: float = 0.0,
        tail_weight: float = 0.0,
        tail_fraction: float = 0.05,
        ponding_weight: float = 0.0,
        ponding_margin: float = 0.0,
        top_zero_scaled=None,
        layer_sigmas=None,
    ):
        super().__init__()
        if base not in ("mse", "mae"):
            raise ValueError(f"Loss {base} not supported")
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must lie in (0, 1]")
        if vertical_weight < 0.0 or tail_weight < 0.0 or ponding_weight < 0.0:
            raise ValueError("Loss weights must be non-negative")
        if ponding_margin < 0.0:
            raise ValueError("ponding_margin must be non-negative")
        if ponding_weight > 0.0 and top_zero_scaled is None:
            raise ValueError(
                "ponding_weight requires top_zero_scaled (-mu/sigma of the "
                "top pressure layer's scaler)"
            )

        self.base = base
        self.vertical_weight = float(vertical_weight)
        self.tail_weight = float(tail_weight)
        self.tail_fraction = float(tail_fraction)
        self.ponding_weight = float(ponding_weight)
        self.ponding_margin = float(ponding_margin)
        self.top_zero_scaled = (
            None if top_zero_scaled is None else float(top_zero_scaled)
        )

        if layer_sigmas is None:
            self.register_buffer("layer_sigmas", None)
        else:
            sigmas = torch.as_tensor(layer_sigmas, dtype=torch.get_default_dtype())
            if sigmas.ndim != 1:
                raise ValueError("layer_sigmas must be one dimensional")
            if torch.any(sigmas <= 0):
                raise ValueError("layer_sigmas must be positive")
            self.register_buffer("layer_sigmas", sigmas / sigmas.mean())

    def _pointwise(self, error, valid_mask):
        selected = error[valid_mask]
        if self.base == "mse":
            return torch.mean(selected**2)
        return torch.mean(torch.abs(selected))

    def _tail(self, error, valid_mask):
        """Mean squared error over the worst ``tail_fraction`` of valid cells."""
        squared = (error[valid_mask]) ** 2
        count = squared.numel()
        k = max(1, int(round(self.tail_fraction * count)))
        if k >= count:
            return torch.mean(squared)
        # topk on the flattened valid cells; no sort of the full tensor needed
        return torch.mean(torch.topk(squared, k, sorted=False).values)

    def _vertical(self, error, valid_mask):
        """Squared error of the layer-to-layer pressure difference.

        The mean terms of the per-layer scalers cancel when differencing, so
        only the sigmas are needed to recover a physical gradient error:
        ``d[z] = sigma[z+1] * e[z+1] - sigma[z] * e[z]``.
        """
        if error.shape[1] < 2:
            return error.new_zeros(())

        scaled = error
        if self.layer_sigmas is not None:
            if self.layer_sigmas.numel() != error.shape[1]:
                raise ValueError(
                    f"layer_sigmas has {self.layer_sigmas.numel()} entries but "
                    f"the prediction has {error.shape[1]} layers"
                )
            sigmas = self.layer_sigmas.to(error.dtype).view(1, -1, 1, 1)
            scaled = error * sigmas

        gradient_error = scaled[:, 1:] - scaled[:, :-1]
        # A pair is usable only where both layers are active cells.
        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        if not torch.any(pair_mask):
            return error.new_zeros(())
        return torch.mean(gradient_error[pair_mask] ** 2)

    def _ponding(self, predictions, targets, valid_mask):
        """Hinge on the top-layer ponding state (sign of physical pressure).

        The gradient stays constant until a wrong-side cell crosses to the
        correct side of p=0, unlike the pointwise term whose gradient vanishes
        with the error magnitude.  Correct-side cells contribute zero (with
        ``ponding_margin`` at its default of 0), so the term never distorts
        cells the model already gets right.
        """
        centered = predictions[:, -1] - self.top_zero_scaled
        target_sign = torch.where(
            targets[:, -1] > self.top_zero_scaled,
            centered.new_ones(()),
            -centered.new_ones(()),
        )
        top_mask = valid_mask[:, -1]
        if not torch.any(top_mask):
            return predictions.new_zeros(())
        hinge = torch.relu(self.ponding_margin - centered * target_sign)
        return hinge[top_mask].mean()

    def forward(self, predictions, targets, valid_mask):
        if predictions.ndim != 4:
            raise ValueError(
                "PressureFirstGuessLoss expects (batch, z, y, x) tensors; got "
                f"shape {tuple(predictions.shape)}"
            )
        valid_mask = valid_mask.to(torch.bool)
        error = predictions - targets

        loss = self._pointwise(error, valid_mask)
        if self.tail_weight > 0.0:
            loss = loss + self.tail_weight * self._tail(error, valid_mask)
        if self.vertical_weight > 0.0:
            loss = loss + self.vertical_weight * self._vertical(error, valid_mask)
        if self.ponding_weight > 0.0:
            loss = loss + self.ponding_weight * self._ponding(
                predictions, targets, valid_mask
            )
        return loss

    def extra_repr(self) -> str:
        return (
            f"base={self.base}, vertical_weight={self.vertical_weight}, "
            f"tail_weight={self.tail_weight}, tail_fraction={self.tail_fraction}, "
            f"ponding_weight={self.ponding_weight}, "
            f"ponding_margin={self.ponding_margin}"
        )

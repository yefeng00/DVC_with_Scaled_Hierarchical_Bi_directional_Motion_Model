import numpy as np
import scipy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .entropy_coder import _EntropyCoder, pmf_to_cdf, pmf_to_quantized_cdf
from ..ops.bound_ops import LowerBound


class BaseProbModel(nn.Module):
    r"""Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(self, likelihood_bound=1e-9, entropy_coder_precision=16):
        super().__init__()

        self.entropy_coder_precision = entropy_coder_precision

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)
        self.noise_level = 0.4

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def update(self):
        pass

    def forward(self, *args):
        raise NotImplementedError()

    def _quantize(self, inputs, mode, means=None):
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(self.noise_level)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        if self.training:
            outputs = torch.round(outputs) - outputs.detach() + outputs
        else:
            outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    @staticmethod
    def _dequantize(inputs, means=None):
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.float()
        return outputs

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    def compress(self, inputs, indexes, means=None, coder: _EntropyCoder=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self._quantize(inputs, "symbols", means)

        # if len(inputs.size()) != 4:
        #     raise ValueError("Invalid `inputs` size. Expected a 4-D tensor.")

        # if inputs.size() != indexes.size():
        #     raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        coder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(),
            indexes.reshape(-1).int().tolist(),
            self._quantized_cdf.tolist(),
            self._cdf_length.reshape(-1).int().tolist(),
            self._offset.reshape(-1).int().tolist(),
        )

    def decompress(self, indexes, means=None, coder: _EntropyCoder=None):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """

        # if len(indexes.size()) != 4:
        #     raise ValueError("Invalid `indexes` size. Expected a 4-D tensor.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:-2] != indexes.size()[:-2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size() and (
                means.size(2) != 1 or means.size(3) != 1
            ):
                raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new(indexes.size())

        values = coder.decode_stream(
            indexes.reshape(-1).int().tolist(),
            cdf.tolist(),
            self._cdf_length.reshape(-1).int().tolist(),
            self._offset.reshape(-1).int().tolist(),
        )
        outputs = torch.Tensor(values).reshape(outputs.size())
        outputs = self._dequantize(outputs, means)
        return outputs


class FactorizedProbModel(BaseProbModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`__
    for an introduction.
    """

    def __init__(
        self,
        channels,
        *args,
        tail_mass=1e-9,
        init_scale=10,
        filters=(3, 3, 3, 3),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        self._biases = nn.ParameterList()
        self._factors = nn.ParameterList()
        self._matrices = nn.ParameterList()

        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self._matrices.append(nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self._biases.append(nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self._factors.append(nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def _medians(self):
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force=False):
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:  # pylint: disable=E0203
            return

        super().update()
        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2

    def loss(self):
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs, stop_gradient):
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = self._matrices[i]
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = self._biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self._factors):
                factor = self._factors[i]
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @torch.jit.unused
    def _likelihood(self, inputs):
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        return likelihood

    def forward(self, x):
        # Convert to (channels, ... , batch) format
        x = x.permute(1, 2, 3, 0).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)
        x_tilde = self._quantize(
            values, "noise" if self.training else "dequantize", self._medians()
        )
        x_hat = self._quantize(values, "dequantize", self._medians())

        # Add noise or quantize
        likelihood = self._likelihood(x_tilde)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        # Convert back to input tensor shape
        x_hat = x_hat.reshape(shape)
        x_hat = x_hat.permute(3, 0, 1, 2).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(3, 0, 1, 2).contiguous()

        return x_hat, likelihood

    @staticmethod
    def _build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C).view(1, -1, 1, 1)
        indexes = indexes.int()
        return indexes.repeat(N, 1, H, W)

    def compress(self, x, coder: _EntropyCoder = None):
        indexes = self._build_indexes(x.size())
        medians = self._medians().detach().view(1, -1, 1, 1)
        return super().compress(x, indexes, medians, coder)

    def decompress(self, size, coder: _EntropyCoder = None):
        output_size = (1, self._quantized_cdf.size(0), size[0], size[1])
        indexes = self._build_indexes(output_size)
        medians = self._medians().detach().view(1, -1, 1, 1)
        return super().decompress(indexes, medians, coder)


class GaussianProbModel(BaseProbModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`__
    for more information.
    """

    def __init__(self, *args, scale_bound=0.11, tail_mass=1e-9, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale_min = 0.11
        self.scale_max = 256.0
        self.scale_level = 64
        self.scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

        self.tail_mass = float(tail_mass)
        if scale_bound > 0:
            self.lower_bound_scale = LowerBound(scale_bound)
        else:
            raise ValueError("Invalid parameters")

    @staticmethod
    def get_scale_table(min, max, levels):
        return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update(self):
        super().update()

        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood

    def forward(self, x, scales, means=None):
        x_tilde = self._quantize(
            x, "noise" if self.training else "dequantize", means
        )
        x_hat = self._quantize(x, "dequantize", means)
        likelihood = self._likelihood(x_tilde, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return x_hat, likelihood

    def build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = (torch.log(scales) - self.log_scale_min) / self.log_scale_step + 1
        indexes = indexes.clamp_(0, self.scale_level - 1)
        return indexes.int()


class LaplacianProbModel(GaussianProbModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`__
    for more information.
    """

    def __init__(self, *args, scale_bound=0.01, tail_mass=1e-9, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale_min = 0.01
        self.scale_max = 100.0
        self.scale_level = 256
        self.scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

        self.tail_mass = float(tail_mass)
        if scale_bound > 0:
            self.lower_bound_scale = LowerBound(scale_bound)
        else:
            raise ValueError("Invalid parameters")

    def _standardized_cumulative(self, inputs, scales):
        dist = torch.distributions.laplace.Laplace(torch.zeros_like(scales), scales)
        # Using the complementary error function maximizes numerical precision.
        return dist.cdf(inputs)

    def update(self):
        pmf_center = (torch.zeros_like(self.scale_table) + 50).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        for i in range(50, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = self._standardized_cumulative(samples, samples_scale.squeeze(1))
            probs = torch.squeeze(probs)
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)
        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        upper = self._standardized_cumulative((0.5 - samples), samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples), samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = (torch.log(scales) - self.log_scale_min) / self.log_scale_step
        indexes = indexes.clamp_(0, self.scale_level - 1)
        return indexes.int()

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values), scales)
        lower = self._standardized_cumulative((-half - values), scales)
        likelihood = upper - lower

        return likelihood


class Bitparm(nn.Module):
    def __init__(self, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class CdfHelper(object):
    def __init__(self):
        super().__init__()
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def set_cdf(self, offset, quantized_cdf, cdf_length):
        self._offset = offset.reshape(-1).int().tolist()
        self._quantized_cdf = quantized_cdf.tolist()
        self._cdf_length = cdf_length.reshape(-1).int().tolist()

    def get_cdf_info_list(self):
        return self._quantized_cdf, \
            self._cdf_length, \
            self._offset


class FactorizedProbModel2(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        self.channel = channel

        self.cdf_helper = None

    def _logits_cumulative(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)

    def forward(self, x):
        likelihoods = self._logits_cumulative(x + 0.5) - self._logits_cumulative(x - 0.5)
        likelihoods = torch.clamp(likelihoods, 1e-9)
        return likelihoods

    def update(self, force=False):
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if not force:  # pylint: disable=E0203
            return

        self.cdf_helper = CdfHelper()
        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.channel), device=device)

            minima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) - i
                samples = samples[None, :, None, None]
                probs = self._logits_cumulative(samples)
                probs = torch.squeeze(probs)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) + i
                samples = samples[None, :, None, None]
                probs = self._logits_cumulative(samples)
                probs = torch.squeeze(probs)
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            offset = -minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            max_length = pmf_length.max()
            device = pmf_start.device
            samples = torch.arange(max_length, device=device)

            samples = samples[None, :] + pmf_start[:, None, None]

            half = float(0.5)

            lower = self._logits_cumulative(samples - half).squeeze(0)
            upper = self._logits_cumulative(samples + half).squeeze(0)
            pmf = upper - lower

            pmf = pmf[:, 0, :]
            tail_mass = lower[:, 0, :1] + (1.0 - upper[:, 0, -1:])

            quantized_cdf = pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.cdf_helper.set_cdf(offset, quantized_cdf, cdf_length)

    @staticmethod
    def build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C).view(1, -1, 1, 1)
        indexes = indexes.int()
        return indexes.repeat(N, 1, H, W)

    def compress(self, x, coder: _EntropyCoder):
        indexes = self.build_indexes(x.size())
        return coder.encode_with_indexes(x.reshape(-1).int().tolist(),
                                         indexes[0].reshape(-1).int().tolist(),
                                         *self.cdf_helper.get_cdf_info_list())

    def decompress(self, size, coder: _EntropyCoder):
        output_size = (1, self.channel, size[0], size[1])
        indexes = self.build_indexes(output_size)
        val = coder.decode_stream(indexes.reshape(-1).int().tolist(),
                                               *self.cdf_helper.get_cdf_info_list())
        val = val.reshape(indexes.shape)
        return val.float()


class SymmetricProbModel(object):
    def __init__(self, distribution='laplace'):
        assert distribution in ['laplace', 'gaussian']
        self.distribution = distribution
        self.scale_min = 0.01
        self.scale_max = 64.0
        self.scale_level = 256
        self.scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)
        self.entropy_coder = None
        self.cdf_helper = None

    @staticmethod
    def get_scale_table(min, max, levels):
        return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

    def update(self, force=False):
        if not force:
            return
        self.cdf_helper = CdfHelper()

        pmf_center = torch.zeros_like(self.scale_table) + 50
        scales = torch.zeros_like(pmf_center) + self.scale_table
        mu = torch.zeros_like(scales)
        if self.distribution == 'laplace':
            gaussian = torch.distributions.laplace.Laplace(mu, scales)
        elif self.distribution == 'gaussian':
            gaussian = torch.distributions.normal.Normal(mu, scales)
        for i in range(50, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = gaussian.cdf(samples)
            probs = torch.squeeze(probs)
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.9999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()

        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        mu = torch.zeros_like(scales)
        gaussian = torch.distributions.laplace.Laplace(mu, scales)

        upper = gaussian.cdf(samples + 0.5)
        lower = gaussian.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.cdf_helper.set_cdf(-pmf_center, quantized_cdf, pmf_length+2)

    def build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = (torch.log(scales) - self.log_scale_min) / self.log_scale_step
        indexes = indexes.clamp_(0, self.scale_level - 1)
        return indexes.int()

    def encode(self, x, scales, coder: _EntropyCoder):
        indexes = self.build_indexes(scales)
        return coder.encode_with_indexes(x.reshape(-1).int().tolist(),
                                                      indexes.reshape(-1).int().tolist(),
                                                      *self.cdf_helper.get_cdf_info_list())

    def decode_stream(self, scales, coder: _EntropyCoder):
        indexes = self.build_indexes(scales)
        val = coder.decode_stream(indexes.reshape(-1).int().tolist(),
                                               *self.cdf_helper.get_cdf_info_list())
        val = val.reshape(scales.shape)
        return val.float()

    def set_decoder_cdf(self, coder: _EntropyCoder):
        coder.set_decoder_cdf(*self.cdf_helper.get_cdf_info_list())

    def encode_with_indexes(self, symbols_list, indexes_list, coder):
        return coder.encode_with_indexes(symbols_list, indexes_list,
                                                      *self.cdf_helper.get_cdf_info_list())

    def decode_stream_only_indexes(self, indexes, coder: _EntropyCoder):
        return coder.decode_stream_only_indexes(indexes)


class SymmetricConditional(nn.Module):
    """Symmetric conditional entropy model.
    Args:
        likelihood_bound: Float. If positive, the returned likelihood values are
            ensured to be greater than or equal to this value. This prevents very
            large gradients with a typical entropy loss (defaults to 1e-9).
    """
    def __init__(self, scale_bound=1e-9, likelihood_bound=1e-9):
        super(SymmetricConditional, self).__init__()
        self.scale_bound = scale_bound
        self.likelihood_bound = likelihood_bound

    def standardized_cumulative(self, inputs):
        """Evaluate the standardized cumulative density.

        This function should be optimized to give the best possible numerical
        accuracy for negative input values.

        Args:
            inputs: `Tensor`. The values at which to evaluate the cumulative density.
        Returns:
            A `Tensor` of the same shape as `inputs`, containing the cumulative
            density evaluated at the given inputs.
        """
        raise NotImplementedError('Must inherit from SymmetricConditional.')

    def get_likelihood(self, inputs, mean, scale):
        """
        This assumes that the standardized cumulative has the property
        1 - c(x) = c(-x), which means we can compute differences equivalently in
        the left or right tail of the cumulative. The point is to only compute
        differences in the left tail. This increases numerical stability: c(x) is
        1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
        done with much higher precision than subtracting two numbers close to 1.
        """
        values = inputs - mean
        values = abs(values)
        upper = self.standardized_cumulative((0.5 - values) / scale)
        lower = self.standardized_cumulative((-0.5 - values) / scale)
        likelihood = upper - lower
        # likelihood_mean = torch.mean(likelihood, dim=1)

        return likelihood

    def forward(self, x, mean_scale, p_mean=0):
        if mean_scale.shape[1] == x.shape[1] * 2:
            mean, scale = mean_scale.chunk(2, dim=1)
        elif mean_scale.shape[2] == x.shape[2] * 2:
            mean, scale = mean_scale.chunk(2, dim=2)
        mean = mean + p_mean
        scale = scale.abs()
        scale = scale.clamp(min=self.scale_bound)
        likelihood = self.get_likelihood(
            x, mean, scale).clamp(min=self.likelihood_bound)
        return likelihood


class GaussianConditional(SymmetricConditional):
    """Conditional Gaussian entropy model."""
    def __init__(self, **kwargs):
        super(GaussianConditional, self).__init__(**kwargs)

    def standardized_cumulative(self, inputs):
        const = -(2**-0.5)
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(const * inputs)

    def forward(self, x, mean_scale, p_mean=0):
        return super(GaussianConditional, self).forward(x, mean_scale)


class LaplacianConditional(SymmetricConditional):
    """Conditional Laplacian entropy model."""
    def __init__(self, **kwargs):
        super(LaplacianConditional, self).__init__(**kwargs)

    def standardized_cumulative(self, inputs):
        return

    def forward(self, y, mean_scale):
        mu, sigma = mean_scale.chunk(2, dim=1)
        y = y - mu
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(torch.zeros_like(sigma), sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return probs.clamp(self.likelihood_bound) + 1e-5


class AsymmetricConditional(SymmetricConditional):
    def __init__(self, scale_bound=1e-9, likelihood_bound=1e-9):
        super(AsymmetricConditional, self).__init__()

    def get_likelihood(self, inputs, mean, scale_l, scale_r):
        values = inputs - mean

        upper = torch.empty_like(values)
        lower = torch.empty_like(values)
        upper[values < -0.5] = self.standardized_cumulative(
            (0.5 + values) / scale_l)[values < -0.5]
        lower[values < 0.5] = self.standardized_cumulative(
            (-0.5 + values) / scale_l)[values < 0.5]
        upper[values >= -0.5] = self.standardized_cumulative(
            (0.5 + values) / scale_r)[values >= -0.5]
        lower[values >= 0.5] = self.standardized_cumulative(
            (-0.5 + values) / scale_r)[values >= 0.5]
        likelihood = upper - lower
        return likelihood

    def forward(self, x, mean_scale):
        mean, scale_l, scale_r = mean_scale.chunk(3, dim=1)
        scale_l, scale_r = scale_l.abs(), scale_r.abs()
        scale_l = scale_l.clamp(min=self.scale_bound)
        scale_r = scale_r.clamp(min=self.scale_bound)
        likelihood = self.get_likelihood(
            x, mean, scale_l, scale_r).clamp(min=self.likelihood_bound)
        return likelihood


class GaussianConditionalAsy(AsymmetricConditional):
    """Conditional Gaussian entropy model."""
    def __init__(self, **kwargs):
        super(GaussianConditionalAsy, self).__init__(**kwargs)

    def standardized_cumulative(self, inputs):
        const = -(2**-0.5)
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(const * inputs)

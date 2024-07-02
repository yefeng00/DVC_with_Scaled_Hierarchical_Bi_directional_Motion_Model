import math
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..logger import Singleton

# isort: off; pylint: disable=E0611,E0401

# isort: on; pylint: enable=E0611,E0401

def pmf_to_quantized_cdf(pmf, precision=16):
    from .MLCodec_CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf

def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length, precision=16):
    cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
    for i, p in enumerate(pmf):
        prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
        _cdf = pmf_to_quantized_cdf(prob, precision)
        cdf[i, : _cdf.size(0)] = _cdf
    return cdf

@Singleton
class _EntropyCoder:
    """Proxy class to an actual entropy coder class."""

    def __init__(self):
        from .MLCodec_rans import BufferedRansEncoder, RansDecoder
        self._encoder = BufferedRansEncoder()
        self._decoder = RansDecoder()

    def encode_with_indexes(self, *args, **kwargs):
        self._encoder.encode_with_indexes(*args, **kwargs)

    def flush_encoder(self):
        return self._encoder.flush()

    def reset_encoder(self):
        self._encoder.reset()

    def set_stream(self, stream):
        self._decoder.set_stream(stream)

    def set_decoder_cdf(self, cdf, cdf_length, offset):
        self._decoder.set_cdf(cdf, cdf_length, offset)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)
    
    def decode_stream(self, indexes, cdf, cdf_length, offset):
        rv = self._decoder.decode_stream(indexes, cdf, cdf_length, offset)
        rv = np.array(rv)
        rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
        return rv

    def decode_stream_only_indexes(self, indexes):
        rv = self._decoder.decode_stream_only_indexes(indexes)
        return rv


def get_entropy_coder() -> _EntropyCoder:
    return _EntropyCoder()
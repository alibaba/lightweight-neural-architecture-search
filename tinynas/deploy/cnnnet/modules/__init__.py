# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from .blocks_basic import (BaseSuperBlock, ConvKXBN, ConvKXBNRELU,
                           network_weight_stupid_init)
from .super_res_k1dwk1 import ResK1DWK1, SuperResK1DWK1
from .super_res_k1dwsek1 import ResK1DWSEK1, SuperResK1DWSEK1
from .super_quant_res_k1dwk1 import SuperQuantResK1DWK1
from .super_res_k1kxk1 import ResK1KXK1, SuperResK1KXK1
from .super_res_k1kx import ResK1KX, SuperResK1KX
from .super_res_kxkx import ResKXKX, SuperResKXKX

__all_blocks__ = {
    'ConvKXBN': ConvKXBN,
    'ConvKXBNRELU': ConvKXBNRELU,
    'BaseSuperBlock': BaseSuperBlock,
    'ResK1KXK1': ResK1KXK1,
    'ResK1KX': ResK1KX,
    'ResKXKX': ResKXKX,
    'ResK1DWK1': ResK1DWK1,
    'ResK1DWSEK1': ResK1DWSEK1,
    'SuperResK1KXK1': SuperResK1KXK1,
    'SuperResK1KX': SuperResK1KX,
    'SuperResKXKX': SuperResKXKX,
    'SuperResK1DWK1': SuperResK1DWK1,
    'SuperResK1DWSEK1': SuperResK1DWSEK1,
    'SuperQuantResK1DWK1': SuperQuantResK1DWK1,
}

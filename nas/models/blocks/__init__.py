from .blocks_basic import *
from .SuperResConvK1KXK1 import *
from .SuperResK1DWK1 import *

__all_blocks__ = {
    'ConvKXBN': ConvKXBN,
    'ConvKXBNRELU': ConvKXBNRELU,
    'BaseSuperBlock': BaseSuperBlock,
    'ResConvK1KXK1': ResConvK1KXK1,
    'SuperResConvK1KXK1': SuperResConvK1KXK1,
    'ResK1DWK1': ResK1DWK1,
    'SuperResK1DWK1': SuperResK1DWK1,
}


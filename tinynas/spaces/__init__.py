# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from .base import BaseSpace 
from .builder import build_space
from .space_k1kxk1 import Spacek1kxk1
from .space_k1dwk1 import Spacek1dwk1
from .space_k1dwsek1 import Spacek1dwsek1
from .space_k1kx import Spacek1kx
from .space_kxkx import Spacekxkx
from .space_3d_k1dwk1 import Space3Dk1dwk1
from .space_quant_k1dwk1 import SpaceQuantk1dwk1

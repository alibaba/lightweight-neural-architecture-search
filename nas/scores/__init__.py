from .compute_entropy import ComputeEntropyScore
from .compute_madnas import ComputeMadnasScore

__all_scores__ = {
    'entropy': ComputeEntropyScore,
    'madnas': ComputeMadnasScore,
}

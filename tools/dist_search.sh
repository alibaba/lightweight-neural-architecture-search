# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at

CONFIG=$1
shift 1
CFG_OPTIONS="${*:-""}"

nproc=64
set -e

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mpirun --allow-run-as-root -np ${nproc} -H 127.0.0.1:${nproc} -bind-to none -map-by slot -mca pml ob1 \
  -mca btl ^openib -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
python3 $(dirname "$0")/search.py ${CONFIG}  ${CFG_OPTIONS}
# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at

CONFIG=$1
shift 1
CFG_OPTIONS="${*:-""}"

set -e

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3  $(dirname "$0")/search.py ${CONFIG}  ${CFG_OPTIONS}
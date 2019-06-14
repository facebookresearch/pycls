#!/usr/bin/env bash

# Hacky script for launching IN benchmarks
#
# Example usage:
#    ./cluster/run_benchmarks.sh

export TOOL=benchmark

BIN="cluster/launch.sh"
DIR="configs/baselines/in1k"

function NGPU_ON() {
  export GPU=$1
  export CPU=$(((8 * $GPU) > 48 ? 48 : (8 * $GPU)))
  export MEM=$(((32 * $GPU) > 200 ? 200 : (32 * $GPU)))
}

function NGPU_OFF() {
  unset GPU
  unset CPU
  unset MEM
}

TAG="BENCH_PYTH"

NGPU_ON 1
CFG="R-50-1x64d_bs32_1gpu.yaml"
${BIN} "${DIR}/${CFG}" "${TAG}" build
NGPU_OFF

NGPU_ON 2
CFG="R-50-1x64d_bs64_2gpu.yaml"
${BIN} "${DIR}/${CFG}" "${TAG}" skip-build
NGPU_OFF

NGPU_ON 4
CFG="R-50-1x64d_bs128_4gpu.yaml"
${BIN} "${DIR}/${CFG}" "${TAG}" skip-build
NGPU_OFF

NGPU_ON 8
CFG="R-50-1x64d_bs256_8gpu.yaml"
${BIN} "${DIR}/${CFG}" "${TAG}" skip-build
NGPU_OFF

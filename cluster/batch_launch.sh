#!/usr/bin/env bash

# Hacky script for batch launching cifar baselines
#
# Example usage:
#    T=baselines M=resnet D=56 ./cluster/batch_launch.sh

BIN="cluster/launch.sh"
DIR="configs/${T-baselines}/cifar10"

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


M=${M-vgg}
D=${D-16}
TAG="CIFAR_DEF-SCH_${M^^}-${D}_BS-128"

NGPU_ON 1
CFG="${M}${D}_bs128_1gpu.yaml"
${BIN} "${DIR}/${CFG}" "${TAG}" build
${BIN} "${DIR}/${CFG}" "${TAG}" skip-build
${BIN} "${DIR}/${CFG}" "${TAG}" skip-build
${BIN} "${DIR}/${CFG}" "${TAG}" skip-build
${BIN} "${DIR}/${CFG}" "${TAG}" skip-build
NGPU_OFF

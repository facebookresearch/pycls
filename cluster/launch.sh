#!/usr/bin/env bash

# To use V100 hosts use:
#   GPU_TYPE=V100 ./cluster/launch.sh ...

CFG=$1
NAME="${2}_${CFG}"
BUILD=${3-build}

# Resources
GPU=${GPU-8}
CPU=${CPU-48}
MEM=${MEM-200}

# Use ENTITLEMENT if it's set in the env, otherwise use gpu_fair
ENTITLEMENT=${ENTITLEMENT-gpu_fair}
# GPU_TYPE can be K40, M40, P100, V100 (see https://fburl.com/wiki/xjxcs4hl)
GPU_TYPE=${GPU_TYPE-P100}
CAPABILITIES="GPU_${GPU_TYPE}_HOST"

TOOLS=experimental/deeplearning/ilijar/pycls/tools
TOOL=${TOOL-train_net}_xar

if [ "${BUILD}" == "build" ]
then
  echo "BUILDING: ${TOOLS}:${TOOL}"
  buck build \
    @mode/opt \
    -c python.native_link_strategy=separate \
    ${TOOLS}:${TOOL}
fi
BINARY=$(buck targets -v 0 --show-full-output ${TOOLS}:${TOOL} | cut -d ' ' -f 2)

echo "POOL:     $ENTITLEMENT"
echo "GPU_TYPE: $GPU_TYPE"
echo "RES:      $GPU,$CPU,$MEM"
echo "BINARY:   $BINARY"
echo "NAME:     $NAME"
echo "CFG:      $CFG"
cat "$CFG"

fry flow-gpu --name "${NAME}" \
  --flow-entitlement "${ENTITLEMENT}" \
  --resources '{"gpu": '$GPU', "cpu_core": '$CPU', "ram_gb": '$MEM'}' \
  --capabilities '["'$CAPABILITIES'"]' \
  --environment "{\"PYTHONUNBUFFERED\": \"True\"}" \
  --copy-file "${CFG}" . \
  --binary-type local \
  --disable-source-snapshot true \
  --retry 9 \
  -- \
  "${BINARY}" \
  --cfg "$(basename "${CFG}")" \
  OUT_DIR ./output

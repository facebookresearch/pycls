#!/bin/bash -ev
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Run this script at project root by "./dev/linter.sh" before you commit.

{
	black --version | grep "19.3b0" > /dev/null
} || {
	echo "Linter requires black==19.3b0 !"
	exit 1
}

echo "Running isort..."
isort -y -sp ./dev

echo "Running black..."
black . --exclude pycls/datasets/data/

echo "Running flake8..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 . --config ./dev/.flake8
else
  python3 -m flake8 . --config ./dev/.flake8
fi

command -v arc > /dev/null && {
  echo "Running arc lint ..."
  arc lint
}

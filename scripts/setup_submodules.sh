#!/usr/bin/env bash
set -e

echo "Initializing git submodules..."
git submodule update --init --recursive

echo "Installing Accelergy core..."
cd accelergy
pip install .
cd ..

echo "Installing Accelergy plugins..."
for plug in \
  accelergy_plugins/accelergy-aladdin-plug-in \
  accelergy_plugins/accelergy-library-plug-in \
  accelergy_plugins/accelergy-cacti-plug-in \
  accelergy_plugins/accelergy-adc-plug-in \
  accelergy_plugins/accelergy-neurosim-plugin; do

  echo "Building and installing $plug"
  cd "$plug"

  if [[ "$plug" == *"accelergy-cacti-plug-in" ]]; then
    make build
  fi

  if [[ "$plug" == *"accelergy-neurosim-plugin" ]]; then
    python setup.py build_ext
  fi

  pip install .
  cd - >/dev/null
done

echo "All submodules installed successfully."
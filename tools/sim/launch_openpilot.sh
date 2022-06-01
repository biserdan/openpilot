#!/bin/bash

export PASSIVE="0"
export NOBOARD="1"
export SIMULATION="1"
export FINGERPRINT="HONDA CIVIC 2016"

if [ "$HOSTNAME" = ba22openpilot-2 ]; then
  echo "launch_openpilot: use camerad"
  export BLOCK="loggerd"
else
  echo "launch_openpilot: no camerad"
  export BLOCK="camerad,loggerd"
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd ../../selfdrive/manager && exec ./manager.py

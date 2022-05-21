#!/bin/bash

export PASSIVE="0"
export SIMULATION="1"
export NOBOARD="1"
export FINGERPRINT="HONDA CIVIC 2016"

export BLOCK="dmonitoringd,dmonitoringmodeld,loggerd"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd ../../selfdrive/manager && ./manager.py

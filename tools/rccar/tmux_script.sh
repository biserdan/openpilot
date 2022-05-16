#!/bin/bash
tmux new -d -s rccar
tmux send-keys "./launch_openpilot.sh" ENTER
tmux neww
tmux send-keys "./bridge.py $*" ENTER
tmux a -t rccar

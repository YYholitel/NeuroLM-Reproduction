#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-full20-20261109-r1
SOCKET=/tmp/clawdbot-tmux-sockets/neurolm-pretrain-full.sock
mkdir -p /tmp/clawdbot-tmux-sockets
if tmux -S "$SOCKET" has-session -t pretrain_neurolm 2>/dev/null; then
    printf 'Session already exists; refusing a duplicate launch.\n' >&2
    exit 1
fi
tmux -S "$SOCKET" new-session -d -s pretrain_neurolm -n runner
tmux -S "$SOCKET" set-option -t pretrain_neurolm remain-on-exit on
tmux -S "$SOCKET" send-keys -t pretrain_neurolm:0.0 -l -- "set -o pipefail; python -u '$ROOT/drive.py' 2>&1 | tee '$ROOT/tmux.log'; exit \$?"
tmux -S "$SOCKET" send-keys -t pretrain_neurolm:0.0 Enter
printf 'tmux -S %s attach -t pretrain_neurolm\n' "$SOCKET"
printf 'tmux -S %s capture-pane -p -J -t pretrain_neurolm:0.0 -S -100\n' "$SOCKET"

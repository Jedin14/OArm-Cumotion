#!/usr/bin/env bash
# Install a systemd service that starts the L2 pipeline at boot, after networking and after the
# host clock is synchronised (see setup_time_sync.sh).
#
#   sudo ./install_service.sh [user]          # default user: the one calling sudo
set -euo pipefail
if [[ $EUID -ne 0 ]]; then echo "run with sudo"; exit 1; fi
RUN_USER="${1:-${SUDO_USER:-$USER}}"
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$WS/install/setup.bash" ]]; then echo "build the workspace first (colcon build) in $WS"; exit 1; fi

cat > /etc/systemd/system/l2-perception.service <<UNIT
[Unit]
Description=Unitree L2 stationary perception (driver, stabilizer, obstacle detection)
Wants=network-online.target time-sync.target
After=network-online.target time-sync.target

[Service]
Type=simple
User=$RUN_USER
# l2.sh spins the lidar up on start and down again when the service is stopped
ExecStart="$WS/scripts/l2.sh" start rviz:=false
KillSignal=SIGTERM
KillMode=mixed
TimeoutStopSec=40
Restart=on-failure
RestartSec=3
LimitRTPRIO=50

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable l2-perception.service
echo "Installed. Start now: sudo systemctl start l2-perception ; logs: journalctl -u l2-perception -f"

#!/usr/bin/env bash
# Make the host clock trustworthy BEFORE the lidar pipeline starts (fixes the "1970" problem of
# computers without an RTC battery, e.g. Jetson boards).
#
#   sudo ./setup_time_sync.sh                    # use the default (internet) NTP pool
#   sudo ./setup_time_sync.sh 192.168.10.1       # use a local NTP server (robot / site network)
#
# * installs chrony (replaces systemd-timesyncd)
# * "makestep 1 -1": always STEP a wrong clock instead of slewing it for hours
# * enables chrony-wait.service, so systemd's time-sync.target is reached only once the clock is
#   really synchronised; l2-perception.service (install_service.sh) starts after that target.
# The l2_driver additionally waits (parameter wait_for_clock_sync) and the downstream nodes reset
# cleanly if the clock is stepped while running, so nothing drifts or crashes either way.
set -euo pipefail
if [[ $EUID -ne 0 ]]; then echo "run with sudo"; exit 1; fi

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y chrony

mkdir -p /etc/chrony/conf.d
{
  echo "# Unitree L2 pipeline"
  echo "makestep 1.0 -1"
  echo "rtcsync"
  for s in "$@"; do echo "server $s iburst prefer"; done
} > /etc/chrony/conf.d/unitree-l2.conf
grep -q "^confdir /etc/chrony/conf.d" /etc/chrony/chrony.conf || \
  echo "confdir /etc/chrony/conf.d" >> /etc/chrony/chrony.conf

systemctl enable --now chrony
systemctl enable chrony-wait.service 2>/dev/null || \
  echo "note: chrony-wait.service not available; time-sync.target will not wait for sync"
systemctl restart chrony
sleep 2
chronyc tracking | sed -n '1,6p'
echo "Done. 'timedatectl' should report: System clock synchronized: yes"

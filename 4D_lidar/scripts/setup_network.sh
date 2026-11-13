#!/usr/bin/env bash
# Configure a network interface for the Unitree L2 (factory default: lidar 192.168.1.62:6101
# sends to host 192.168.1.2:6201).
#
#   sudo ./setup_network.sh <interface> [host_address/prefix]
#   sudo ./setup_network.sh enx00e04c680001
#
# Creates a persistent NetworkManager profile "unitree-l2" on that interface and raises the
# default UDP receive buffer (the SDK has no receive thread, so the kernel socket buffer is
# what absorbs scheduling hiccups). Use a DEDICATED port (second NIC / USB-Ethernet adapter):
# the profile replaces the addressing of that interface.
#
# Quick, non-persistent alternative for a test:  sudo ip addr add 192.168.1.2/24 dev <interface>
set -euo pipefail

IFACE="${1:-}"
ADDR="${2:-192.168.1.2/24}"
if [[ -z "$IFACE" ]]; then
  echo "usage: sudo $0 <interface> [192.168.1.2/24]"; echo; ip -br link; exit 1
fi
if [[ $EUID -ne 0 ]]; then echo "run with sudo"; exit 1; fi
if ! ip link show "$IFACE" >/dev/null 2>&1; then echo "no interface $IFACE"; exit 1; fi
if ip route show default | grep -q "dev $IFACE "; then
  echo "WARNING: $IFACE carries your default route (it is probably your LAN / internet uplink)."
  read -r -p "Reconfigure it for the lidar anyway? [y/N] " a; [[ "$a" == "y" ]] || exit 1
fi

nmcli con delete unitree-l2 >/dev/null 2>&1 || true
nmcli con add type ethernet ifname "$IFACE" con-name unitree-l2 \
  ipv4.method manual ipv4.addresses "$ADDR" ipv4.never-default yes ipv6.method ignore \
  connection.autoconnect yes >/dev/null
nmcli con up unitree-l2

cat > /etc/sysctl.d/60-unitree-l2.conf <<'SYS'
# Unitree L2: room for ~2 s of lidar packets in each UDP socket buffer
net.core.rmem_default = 4194304
net.core.rmem_max = 8388608
SYS
sysctl --system >/dev/null

echo "Interface $IFACE -> $ADDR. Pinging the lidar at 192.168.1.62 ..."
ping -c 3 -W 1 192.168.1.62 && echo "Lidar reachable." || \
  echo "No reply (the L2 may not answer ICMP; check that UDP arrives: sudo tcpdump -i $IFACE -c 5 udp port 6201)"

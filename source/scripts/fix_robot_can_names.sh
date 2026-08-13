#!/usr/bin/env bash
set -euo pipefail

# Keep the two USB-CAN adapters on their semantic, stable interface names.
# udev can match each adapter by serial, but cannot atomically swap can1/can2
# when the kernel enumerates them in the opposite order.  This script performs
# that swap through temporary names after both adapters are present.

# Verified from live traffic on 2026-08-07 (Piper status IDs 0x251-0x2A8).
arm_serial="002600355246570520323934"
base_serial="003100494148570C20343133"

find_interface() {
  local wanted_serial="$1"
  local interface device usb_device serial
  for interface in /sys/class/net/can*; do
    [[ -e "${interface}/device" ]] || continue
    device="$(readlink -f "${interface}/device")"
    # A CAN netdev points at the USB interface directory (for example
    # .../1-2.4/1-2.4:1.0); its parent USB device owns the serial attribute.
    usb_device="$(dirname "${device}")"
    serial="$(cat "${usb_device}/serial" 2>/dev/null || true)"
    if [[ "${serial}" == "${wanted_serial}" ]]; then
      basename "${interface}"
      return 0
    fi
  done
  return 1
}

arm_interface="$(find_interface "${arm_serial}" || true)"
base_interface="$(find_interface "${base_serial}" || true)"

# Do nothing while one adapter is absent; it will be corrected on the next run.
[[ -n "${arm_interface}" && -n "${base_interface}" ]] || exit 0

if [[ "${arm_interface}" == "can1" && "${base_interface}" == "can2" ]]; then
  exit 0
fi

ip link set "${arm_interface}" down
ip link set "${base_interface}" down
ip link set "${arm_interface}" name robot_arm_tmp
ip link set "${base_interface}" name robot_base_tmp
ip link set robot_arm_tmp name can1
ip link set robot_base_tmp name can2

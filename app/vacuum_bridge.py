#!/usr/bin/env python3
"""The vacuum relay: /beast_arm/vacuum (std_msgs/Bool) -> the ESP32 over USB
serial, b'1' on / b'0' off.

The pick stack (and the web GUI, which runs it) publishes /beast_arm/vacuum,
latched, on a change only: true just before the cup touches the box, false at
the drop pose and when the arm is back with nothing held. Left arm only.

Terminal 1 starts it (native/run_launch_everything.sh; vacuum:=false leaves it
out, vacuum_port:=... another port). On its own, with vacuum:=false there:

    native/run_vacuum_bridge.sh
    native/run_vacuum_bridge.sh port:=/dev/serial/by-id/usb-... baud:=115200

Parameters:
  port    our ESP32 (default, ESP32): its own USB port by id -- the same name
          whichever ttyACM number it gets (it is /dev/ttyACM0 now). Or another
          path, or auto: the only device in /dev/serial/by-id (with none or
          several it waits and says which there are; it never guesses)
  baud    115200
  topic   /beast_arm/vacuum
  resend  1.0 s: the wanted state written again this often, so an ESP32 that
          restarts (a pump or relay brownout, a USB reset) is switched back
          within a second. 0 = only on a change

How it behaves:
  - Subscribed latched (transient local, depth 1) like the publisher: started
    after the pick stack, it still gets the state now.
  - Unplugged or the port fails: it keeps running, reopens the port every
    second, and writes the wanted state again once it is back.
  - Many ESP32 boards restart when the port opens: it waits 2 s after opening
    before it writes. So (re)starting this node while a box hangs on the cup
    can drop it -- the relay is off while the ESP32 boots.
  - Stopping it leaves the relay as it is (a held box is not let go), unless
    the ESP32 restarts when the port closes.
  - The port is opened exclusively: a second bridge on the same port is told so.
  - Lines the ESP32 sends back are logged ("esp32: ..."), each new one once.
"""

import glob
import os
import time

import rclpy
import serial
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

# the vacuum's ESP32 (2026-11-05): its native USB (303a:1001), not the board's
# CH343 port (usb-1a86_USB_Single_Serial_5959116712-if00, /dev/ttyACM1)
ESP32 = '/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_34:CD:B0:16:1D:30-if00'
BY_ID = '/dev/serial/by-id/*'
RETRY = 1.0          # s between attempts to open the port
BOOT_WAIT = 2.0      # s after opening: an ESP32 that restarts on open is booted by then


class VacuumBridge(Node):
    def __init__(self):
        super().__init__('vacuum_serial_bridge')
        self.port = str(self.declare_parameter('port', ESP32).value)
        self.baud = int(self.declare_parameter('baud', 115200).value)
        topic = str(self.declare_parameter('topic', '/beast_arm/vacuum').value)
        self.resend = float(self.declare_parameter('resend', 1.0).value)
        self.ser = None
        self.want = None              # the state asked for last (None: nothing heard yet)
        self.sent_at = 0.0
        self.next_try = 0.0
        self.why = None               # the last connection problem (each said once)
        self.rx, self.last_rx = b'', None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Bool, topic, self.on_vacuum, latched)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'{topic} -> {self.port} at {self.baud} baud')

    # -- the port ----------------------------------------------------------------

    def problem(self, text):
        if text != self.why:
            self.why = text
            self.get_logger().warn(text)

    def connect(self):
        """Open the port (at most once per RETRY s); True when open. The wanted
        state is written as soon as it is."""
        now = time.monotonic()
        if now < self.next_try:
            return False
        self.next_try = now + RETRY
        port = self.port
        if port == 'auto':
            found = sorted(glob.glob(BY_ID))
            if len(found) != 1:
                self.problem('no USB serial device -- plug the ESP32 in' if not found else
                             'several USB serial devices, say which: port:=' + ' | '.join(found))
                return False
            port = found[0]
        elif not os.path.exists(port):
            self.problem(f'{port} is not there -- is the ESP32 plugged in?')
            return False
        try:
            ser = serial.Serial(port, self.baud, timeout=0, write_timeout=1, exclusive=True)
        except (serial.SerialException, OSError, ValueError) as exc:
            self.problem(f'cannot open {port}: {exc}')
            return False
        time.sleep(BOOT_WAIT)
        try:
            ser.reset_input_buffer()          # its boot messages
        except (serial.SerialException, OSError):
            pass
        self.ser, self.why, self.rx = ser, None, b''
        self.get_logger().info(f'connected: {port}')
        if self.want is not None:
            self.write()
        return True

    def lost(self, exc):
        self.problem(f'serial port lost ({exc}); reopening')
        try:
            self.ser.close()
        except Exception:                     # noqa: BLE001 - it is gone anyway
            pass
        self.ser = None

    def write(self):
        try:
            self.ser.write(b'1' if self.want else b'0')
            self.ser.flush()
        except (serial.SerialException, OSError) as exc:
            self.lost(exc)
            return False
        self.sent_at = time.monotonic()
        return True

    # -- ROS -----------------------------------------------------------------------

    def on_vacuum(self, msg):
        on = bool(msg.data)
        if on != self.want:
            self.get_logger().info(f'vacuum -> {"ON" if on else "OFF"}' + (
                '' if self.ser is not None else
                '  (NOT CONNECTED: written when the ESP32 is back)'))
        self.want = on
        if self.ser is not None:
            self.write()
        else:
            self.next_try = 0.0               # try at once
            self.connect()

    def tick(self):
        if self.ser is None:
            self.connect()
            return
        try:
            n = self.ser.in_waiting
            if n:
                self.rx += self.ser.read(n)
        except (serial.SerialException, OSError) as exc:
            self.lost(exc)
            return
        *lines, self.rx = self.rx.split(b'\n')
        self.rx = self.rx[-1024:]
        for line in lines:
            text = line.decode(errors='replace').strip()
            if text and text != self.last_rx:
                self.last_rx = text
                self.get_logger().info(f'esp32: {text}')
        if self.want is not None and self.resend > 0 and \
                time.monotonic() - self.sent_at >= self.resend:
            self.write()

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:                 # noqa: BLE001
                pass
            self.ser = None


def main():
    rclpy.init()
    node = VacuumBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

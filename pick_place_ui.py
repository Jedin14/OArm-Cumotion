#!/usr/bin/env python3
"""Type what to pick, watch it happen.

A small control panel for the VLM pick-and-place cycle: an entry box for the
object, a Pick button, an Abort button, and the state machine's current step.
"pick up the screwdriver" becomes the detection prompt "detect screwdriver" --
PaliGemma is a pretrained checkpoint, so "detect X" is the well-formed prompt
shape and the conversational wrapper has to come off (see the note on grasp
verification in pick_place_orchestrator.py).

It drives the orchestrator over its normal interface and holds no robot state of
its own:

    /pick_place/prompt   what to pick, published before starting
    /pick_place/start    runs one cycle
    /pick_place/abort    stops after the current motion
    /pick_place/state    the step it is on, shown live

Started for you by pick_place_demo.launch.py. On its own:

    source native/setup.bash && python3 pick_place_ui.py

Tk is not thread-safe and rclpy callbacks arrive on an executor thread, so
callbacks only push onto a queue that the Tk main loop drains -- touching
widgets directly from a subscription is what makes these panels die with an
X error after a few minutes.
"""

import os
import queue
import signal
import sys
import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger

from vlm_prompt import to_detection_prompt

WS = os.path.dirname(os.path.realpath(__file__))
DEFAULT_STATES_FILE = os.path.join(WS, 'pick_place_states.yaml')

# States that mean the cycle is over, so the buttons can be re-enabled.
TERMINAL = ('DONE', 'FAILED', 'ABORTED', 'IDLE')


def recorded_states(path):
    """Names of the poses on file, so the panel can say if they are missing."""
    try:
        import yaml
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
        return sorted((data.get('states') or {}).keys())
    except Exception:                                # noqa: BLE001 - advisory
        return []


class UiNode(Node):
    """The ROS half. Everything it receives goes on the queue, nothing else."""

    def __init__(self, events):
        super().__init__('pick_place_ui')
        self.events = events
        cb = ReentrantCallbackGroup()

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.prompt_pub = self.create_publisher(String, '/pick_place/prompt', latched)

        self.create_subscription(String, '/pick_place/state', self._on_state, 10,
                                 callback_group=cb)
        self.start_client = self.create_client(Trigger, '/pick_place/start',
                                               callback_group=cb)
        self.abort_client = self.create_client(Trigger, '/pick_place/abort',
                                               callback_group=cb)

    def _on_state(self, msg):
        self.events.put(('state', msg.data))

    def _call(self, client, label):
        if not client.wait_for_service(timeout_sec=2.0):
            self.events.put(('log', f'{label}: orchestrator not running'))
            self.events.put(('done', ''))
            return
        future = client.call_async(Trigger.Request())

        def finished(fut):
            try:
                result = fut.result()
            except Exception as exc:                 # noqa: BLE001 - shown to user
                self.events.put(('log', f'{label} failed: {exc}'))
                self.events.put(('done', ''))
                return
            self.events.put(('log', f'{label}: {result.message}'))
            if not result.success:
                self.events.put(('done', ''))

        future.add_done_callback(finished)

    def start(self, prompt):
        self.prompt_pub.publish(String(data=prompt))
        self.events.put(('log', f'prompt "{prompt}"'))
        # The orchestrator applies a prompt only when idle and reads it at
        # LOCATE, so publishing immediately before starting is the ordering
        # that makes one button do both.
        self._call(self.start_client, 'start')

    def abort(self):
        self._call(self.abort_client, 'abort')


class Panel:

    def __init__(self, node, events, states_file):
        import tkinter as tk
        self.tk = tk
        self.node = node
        self.events = events
        self.states_file = states_file

        self.root = tk.Tk()
        self.root.title('OpenArm pick and place')
        self.root.minsize(460, 280)

        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill='both', expand=True)

        tk.Label(frame, text='What should the arm pick up?',
                 font=('TkDefaultFont', 11, 'bold')).pack(anchor='w')

        self.entry = tk.Entry(frame, font=('TkDefaultFont', 12))
        self.entry.insert(0, 'pick up the screwdriver')
        self.entry.pack(fill='x', pady=(6, 2))
        self.entry.bind('<Return>', lambda _e: self.on_pick())
        self.entry.focus_set()

        self.translated = tk.Label(frame, text='', fg='#666')
        self.translated.pack(anchor='w')
        self.entry.bind('<KeyRelease>', lambda _e: self.show_translation())

        buttons = tk.Frame(frame)
        buttons.pack(fill='x', pady=8)
        self.pick_button = tk.Button(buttons, text='Pick', width=12,
                                     command=self.on_pick)
        self.pick_button.pack(side='left')
        self.abort_button = tk.Button(buttons, text='Abort', width=12,
                                      state='disabled', command=self.on_abort)
        self.abort_button.pack(side='left', padx=6)

        self.state_label = tk.Label(frame, text='IDLE',
                                    font=('TkDefaultFont', 13, 'bold'))
        self.state_label.pack(anchor='w', pady=(4, 0))

        self.log = tk.Text(frame, height=8, state='disabled', wrap='word',
                           font=('TkFixedFont', 9))
        self.log.pack(fill='both', expand=True, pady=(8, 0))

        self.show_translation()
        self.check_states()
        self.root.after(100, self.drain)
        self.install_signal_handlers()

    def install_signal_handlers(self):
        """Make Ctrl-C actually close the panel.

        Tk's mainloop is C code, so Python's default SIGINT handler raises
        KeyboardInterrupt inside whichever after() callback happens to be
        running -- and Tk catches exceptions raised by callbacks, reports them
        and keeps looping. The process then ignores Ctrl-C entirely, and
        `ros2 launch` has to escalate SIGINT -> SIGTERM -> SIGKILL to bring the
        stack down, which is 15 seconds of a shutdown that looks hung.
        Quitting the loop explicitly is what makes it exit on the first one.

        This is safe from a handler: handlers run on the main thread between
        bytecodes, and the 100 ms drain timer guarantees the interpreter gets
        control regularly enough for that to happen promptly.
        """
        for received in (signal.SIGINT, signal.SIGTERM):
            signal.signal(received, lambda *_: self.root.quit())

    # -- helpers -----------------------------------------------------------

    def write(self, line):
        self.log.configure(state='normal')
        self.log.insert('end', line + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def show_translation(self):
        prompt = to_detection_prompt(self.entry.get())
        self.translated.configure(
            text=f'detector prompt:  {prompt}' if prompt else 'type an object')

    def check_states(self):
        names = recorded_states(self.states_file)
        missing = [n for n in ('pre_pick_state', 'drop_state') if n not in names]
        if missing:
            self.write(f'! {", ".join(missing)} not recorded yet -- run: '
                       f'python3 record_states.py {" ".join(missing)}')
        else:
            self.write(f'recorded poses: {", ".join(names)}')

    def busy(self, is_busy):
        self.pick_button.configure(state='disabled' if is_busy else 'normal')
        self.abort_button.configure(state='normal' if is_busy else 'disabled')

    # -- events ------------------------------------------------------------

    def on_pick(self):
        prompt = to_detection_prompt(self.entry.get())
        if not prompt:
            self.write('! type what to pick first')
            return
        self.busy(True)
        self.state_label.configure(text='starting...')
        self.node.start(prompt)

    def on_abort(self):
        self.write('abort requested')
        self.node.abort()

    def drain(self):
        """Pull anything the ROS side queued. The only thread touching widgets."""
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == 'state':
                    self.state_label.configure(text=payload)
                    self.write(payload)
                    if payload.split(':')[0].strip() in TERMINAL:
                        self.busy(False)
                elif kind == 'log':
                    self.write(payload)
                elif kind == 'done':
                    self.busy(False)
        except queue.Empty:
            pass
        self.root.after(100, self.drain)

    def run(self):
        self.root.mainloop()


def main():
    if not os.environ.get('DISPLAY'):
        print('error: no DISPLAY. This panel needs a desktop session -- the same '
              'one RViz and the octomap updater need.', file=sys.stderr)
        return 1

    events = queue.Queue()
    rclpy.init()
    node = UiNode(events)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    states_file = os.environ.get('PICK_PLACE_STATES_FILE', DEFAULT_STATES_FILE)
    try:
        Panel(node, events, states_file).run()
    finally:
        # Order matters: shutdown() makes spin() return, and the thread has to
        # be joined before the node goes away. Tearing the node down underneath
        # a still-spinning executor aborts in the C layer -- "terminate called
        # without an active exception", plus a core dump in the workspace.
        executor.shutdown()
        spin.join(timeout=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())

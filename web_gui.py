#!/usr/bin/env python3
"""WEB GUI: every tool of the arms in one browser page -- click_to_move,
auto_box, calibrate.py and the pick stack's ROS interface, together.

    native/run_web_gui.sh                 (the robot stack first: run_launch_everything.sh)
    then open http://beast:8080 (or http://localhost:8080 on beast itself)

One process: the ROS node and its topics (arm_state, arm_action, ... exactly
as pick_stack.py), the same touch, pick and calibration code as the OpenCV
windows, and a web server for the page:

  GET  /                 the page (web/)
  GET  /stream.mjpg      the camera with what is drawn on it (live markers,
                         the picked point, the boxes, the robot overlay, ...)
  GET  /api/events       the state, as server-sent events (~5 a second)
  POST /api/click        {x, y} in 0..1 of the picture: pick that point
  POST /api/action       {name, ...}: every button (see WebStack.actions)

The box model is the /box_model/detect service: run_web_gui.sh starts
box_model_node.py beside it. --click: click-only, like click_to_move -- no
model, no DETECT BOX / AUTO PICK (native/run_web_gui.sh --click).

Anyone who can open the page can move the arms: it listens on every
interface for the laptop on the network (--host 127.0.0.1 keeps it local).
"""

import asyncio
import collections
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String

import calibrate as calib_app
import click_to_move as ctm_app
import pick_stack
from ctm import armcal, board as bd
from ctm.common import WS, decode_image
from ctm.robot import ClickToMove
from pick_interfaces.msg import PickTarget
from pick_interfaces.srv import DetectTarget

WEB_DIR = os.path.join(WS, 'web')
MARKER_EVERY = 0.15            # s between marker detections (every frame kept 2+ cores busy)
FPS = 15


class WebStack(pick_stack.PickStack):
    """The pick stack's ROS interface plus every window tool, for the page."""

    CALIBRATION = True

    def __init__(self, node):
        super().__init__(node)
        self.anchor_mode = False
        self.view = 'live'            # 'live' camera or the 'captured' frame
        self.show_markers = True      # the live ArUco outlines + ids
        self.click_only = bool(node.args.click)   # --click: no box model
        self.box = None               # the last DETECT BOX result (PickTarget)
        self.boxes = []               # every box of that frame (pixels)
        self.log = collections.deque(maxlen=80)
        self.error = False
        self.aruco = cv2.aruco.DetectorParameters_create()
        self._found = ((), None)
        self._found_at = 0.0
        self.seen = []                # marker ids in view
        self.status = ('Ready. Click the picture to pick a point.' if self.click_only else
                       'Ready. Click the picture to pick a point, or detect a box.')

    # -- messages ---------------------------------------------------------------

    def say(self, text, error=False):
        super().say(text, error)
        self.error = error
        self.log.append({'t': time.time(), 'text': text, 'error': bool(error)})

    # -- calibrate.py's ANCHOR and REFIT ------------------------------------------

    refit = calib_app.CalibrateApp.refit
    _anchor_error = calib_app.CalibrateApp._anchor_error

    # -- the buttons ---------------------------------------------------------------

    def action(self, name, data):
        """One button. Returns (ok, message) for the page's toast."""
        motion = {'move', 'pick', 'return', 'navigation', 'pre_pick', 'cal_arms', 'calibrate',
                  'ready_to_pick', 'detect', 'recapture', 'overlay'}
        if name == 'stop':
            self.node.stop_motion()
            self.abort.set()
            if self.calibrating or self.teaching:
                self.answer('abort')
            self.say('STOP: motion cancelled; the arms hold where they are. Any move clears it',
                     True)
            return True, 'stopped'
        if name in motion:
            self.node.resume()
        if self.teaching:
            if name in ('left', 'right', 'up', 'down', 'in', 'out', 'save', 'skip'):
                self.answer(name)
                return True, name
            return False, 'a touch is held for TEACH / ANCHOR: nudge, SAVE or SKIP first'
        if self.calibrating and name in ('confirm', 'skip', 'abort'):
            self.answer(name)
            return True, name
        if name == 'view':
            self.view = 'captured' if data.get('value') == 'captured' and self.frame is not None \
                else 'live'
            return True, self.view
        if name == 'markers':
            self.show_markers = not self.show_markers
            return True, 'markers on' if self.show_markers else 'markers off'
        if name == 'tool':
            arm = data.get('arm')
            if arm not in ('left', 'right'):
                return False, 'which arm?'
            step = float(data.get('step') or 0.0)
            self.change_tool(arm, toggle=not step, step=step)
            return True, 'tool'
        if name == 'anchor':
            self.anchor_mode = not self.anchor_mode
            self.teach = self.anchor_mode
            self.say('ANCHOR on: click a precise point (marker 11 centre, a box top), MOVE; '
                     'nudge the tip exactly onto it, SAVE' if self.anchor_mode else 'ANCHOR off')
            return True, 'anchor'
        if name == 'teach':
            if self.anchor_mode:
                self.anchor_mode = False
            self.press('teach')
            return True, 'teach'
        if name == 'refit':
            return self._start(self.refit)
        if name in ('detect', 'ready_to_pick') and self.click_only:
            return False, 'click-only mode (--click): no box model - click the picture'
        if name == 'detect':
            return self._start(self.detect_box)
        if name == 'pick' or name == 'move':
            if self.pick is None:
                return False, 'pick a point or DETECT BOX first'
            if self.check is not None and self.check.get('ok') is False:
                return False, 'not reachable: ' + str(self.check.get('why') or '')
            self.move()
            return True, 'moving'
        if name == 'ready_to_pick':
            self._on_action(String(data='ready_to_pick ' + data.get('approach', 'auto')))
            return True, 'ready_to_pick'
        if name in ('navigation', 'pre_pick'):
            if self.busy:
                return False, 'busy'
            self.go_to(name)
            return True, name
        if name == 'cal_arms':
            # never without the folded pieces in the planners (they stick out 4 cm)
            self.node.pieces_on = True
            return self._start(self.calibrate_arms, 'CAL ARMS')
        names = {'recapture': 'recapture', 'calibrate': 'calibrate', 'apply': 'apply',
                 'overlay': 'overlay', 'return': 'return', 'topdown': 'topdown',
                 'pieces': 'pieces'}
        if name in names:
            if self.busy and name not in ('topdown',):
                return False, 'busy'
            self.press(names[name])
            if name in ('recapture', 'overlay'):
                self.view = 'captured'
            return True, name
        return False, f'unknown action {name}'

    def _on_action(self, msg):
        if self.click_only and msg.data.strip().lower().startswith('ready_to_pick'):
            self.report('pick_failed: click-only mode (--click), no box model')
            return
        super()._on_action(msg)

    def _start(self, work, label=None):
        if self.busy:
            return False, 'busy'
        self.in_background(work)
        return True, label or 'started'

    # -- a click on the picture ------------------------------------------------------

    def click(self, x, y):
        """x, y in 0..1 of the picture shown."""
        if self.teaching or self.calibrating or self.calib_view is not None:
            return False, 'not now: finish the step in progress'
        if self.busy:
            return False, 'busy'
        self.node.resume()
        if self.view == 'captured' and self.frame is not None:
            h, w = self.frame[0].shape[:2]
            check = self.select_point(x * w, y * h)
            if check is None:
                return False, self.status
            if self.mode != 'pick':
                self.in_background(lambda: self.ensure_pick_mode() and
                                   self.check_pick(check, in_job=True))
            else:
                threading.Thread(target=self.check_pick, args=(check,), daemon=True).start()
            return True, 'checking'

        # live: a fresh capture (arms to pre_pick first), then that point
        def job():
            if self.mode != 'pick' and not self.ensure_pick_mode():
                return
            self.capture()
            if self.frame is None:
                return
            self.view = 'captured'
            h, w = self.frame[0].shape[:2]
            check = self.select_point(x * w, y * h)
            if check is not None:
                self.check_pick(check, in_job=True)
        self.in_background(job)
        return True, 'capturing'

    # -- DETECT BOX: the model, then the reach check (nothing moves) -----------------

    def detect_box(self):
        if self.mode != 'pick' and not self.ensure_pick_mode():
            return
        if not self.detect_client.wait_for_service(timeout_sec=3.0):
            self.say(f'no box model ({self.node.args.model_service}) - is box_model_node.py '
                     f'running?', True)
            return
        self.say('DETECT BOX: the model is looking...')
        answer = self.node.wait(self.detect_client.call_async(
            DetectTarget.Request(approach='top' if self.top_down else self.node.args.approach)),
            30.0)
        if answer is None or not answer.found:
            self.box = None
            self.say('no box: ' + (answer.message if answer else 'the model did not answer'),
                     True)
            return
        self.box = self.target = answer.target
        self.capture()
        if self.frame is None:
            return
        self.view = 'captured'
        t = answer.target
        self.top_down = t.approach == PickTarget.APPROACH_TOP
        h, w = self.frame[0].shape[:2]
        check = self.select_point(t.u * w / float(t.image_width or w),
                                  t.v * h / float(t.image_height or h),
                                  world=np.array([t.position.x, t.position.y, t.position.z]))
        self.say(f'box: {answer.message}')
        if check is not None:
            self.check_pick(check, in_job=True)

    # -- the state, for the page -------------------------------------------------------

    def snapshot(self):
        check = self.check or {}
        plan = check.get('plan') or {}
        pick = None
        if self.pick is not None:
            _u, _v, p, arm, _o = self.pick
            pick = {'arm': arm, 'xyz': [round(float(v), 4) for v in p],
                    'ok': check.get('ok'), 'why': check.get('why'),
                    'label': plan.get('label'), 'edge': bool(plan.get('edge'))}
        models = {}
        raw = armcal.load_model_raw()
        for arm in ('left', 'right'):
            r = raw.get(arm) or {}
            models[arm] = {'has': self.arm_models.has(arm), 'model': r.get('model'),
                           'rms_px': r.get('rms_px'), 'anchor_mm': r.get('anchor_mm'),
                           'anchors': len(armcal.load_anchors(arm))}
        t = self.target
        color, _d, _i = self.node.latest()
        return {
            'status': self.status, 'error': self.error, 'detail': self.detail,
            'busy': self.busy, 'arm_state': self.current_state(), 'mode': self.mode,
            'result': self.result, 'halted': self.node.halt.is_set(),
            'top_down': self.top_down, 'teach': self.teach, 'teaching': self.teaching,
            'anchor_mode': self.anchor_mode, 'calibrating': self.calibrating,
            'confirming': bool(getattr(self, 'confirming', False)),
            'calib_view': self.calib_view is not None,
            'calibration_ready': self.calibration is not None,
            'overlay': self.overlay is not None, 'view': self.view,
            'captured': self.frame is not None, 'show_markers': self.show_markers,
            'markers': self.seen, 'pieces_on': self.node.pieces_on,
            'tools': {a: {'on': bool(v['on']), 'length': round(float(v['length']), 4)}
                      for a, v in self.node.args.tools.items()},
            'pick': pick, 'last_touch': self.last_touch is not None,
            'target': ({'approach': t.approach, 'confidence': round(float(t.confidence), 2),
                        'xyz': [round(t.position.x, 4), round(t.position.y, 4),
                                round(t.position.z, 4)], 'boxes': int(t.num_boxes)}
                       if t is not None else None),
            'models': models, 'model_when': raw.get('when'),
            'camera': color is not None, 'click_only': self.click_only,
            'log': list(self.log)[-40:],
        }

    # -- the picture -------------------------------------------------------------------

    def picture(self):
        """The frame the page shows, BGR."""
        if self.calib_view is not None:
            img, predicted, found, mask = self.calib_view
            img = img.copy()
            if mask is not None:
                img[mask] = (0.5 * img[mask] + 0.5 * np.array([255, 120, 0])).astype(np.uint8)
            if predicted is not None and np.all(np.isfinite(predicted)):
                cv2.circle(img, (int(predicted[0]), int(predicted[1])), 9, (0, 255, 0), 2)
            if found is not None:
                cv2.drawMarker(img, (int(found[0]), int(found[1])), (0, 0, 255),
                               cv2.MARKER_CROSS, 18, 2)
            return img
        if self.view == 'captured' and self.frame is not None:
            img = self.frame[0].copy()
            h, w = img.shape[:2]
            if self.unreachable is not None and self.unreachable.shape == (h, w):
                bad = self.unreachable
                img[bad] = (0.6 * img[bad] + 0.4 * np.array([40, 40, 200])).astype(np.uint8)
            t = self.box
            if t is not None:
                x1, y1, x2, y2 = (int(round(c)) for c in t.bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (80, 200, 80), 2)
            if self.pick:
                u, v = int(self.pick[0]), int(self.pick[1])
                ok = (self.check or {}).get('ok')
                colour = (90, 200, 60) if ok else (60, 60, 230) if ok is False \
                    else (0, 190, 255)
                cv2.circle(img, (u, v), 12, colour, 2, cv2.LINE_AA)
                cv2.drawMarker(img, (u, v), colour, cv2.MARKER_CROSS, 30, 1, cv2.LINE_AA)
            if self.overlay is not None:
                pts, agree = self.overlay
                ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
                p = pts[ok].astype(int)
                img[p[:, 1], p[:, 0]] = np.where(agree[ok][:, None], [0, 255, 0], [0, 0, 255])
        else:
            color, _d, _i = self.node.latest()
            img = decode_image(color).copy() if color is not None else \
                np.full((480, 640, 3), 24, np.uint8)
            if color is not None and self.show_markers:
                self._draw_markers(img)
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        return img

    def _draw_markers(self, img):
        """calibrate.py's live view: every marker outlined with its id and name
        (green: big enough to read well, orange: small)."""
        now = time.monotonic()
        if now - self._found_at > MARKER_EVERY:
            corners, ids, _ = cv2.aruco.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                      bd.dictionary(), parameters=self.aruco)
            self._found = (corners, ids)
            self._found_at = now
            self.seen = sorted(int(i) for i in ids.ravel()) if ids is not None else []
        corners, ids = self._found
        for c, i in zip(corners or [], ids.ravel() if ids is not None else []):
            c = c.reshape(4, 2)
            size = float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))
            colour = (90, 200, 60) if size >= 20 else (0, 170, 255)
            cv2.polylines(img, [np.round(c).astype(np.int32)], True, colour, 2, cv2.LINE_AA)
            label = f'{int(i)} {bd.name_of(int(i))}'.strip()
            x, y = int(c[:, 0].min()), max(14, int(c[:, 1].min()) - 6)
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1,
                        cv2.LINE_AA)


# -- the web server ----------------------------------------------------------------------

def make_app(stack):
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title='OpenArm')
    app.mount('/static', StaticFiles(directory=WEB_DIR), name='static')

    @app.get('/')
    def index():
        return FileResponse(os.path.join(WEB_DIR, 'index.html'),
                            headers={'Cache-Control': 'no-store'})

    @app.get('/stream.mjpg')
    async def stream():
        async def frames():
            loop = asyncio.get_running_loop()
            try:
                while True:
                    t0 = time.monotonic()
                    jpg = await loop.run_in_executor(None, lambda: cv2.imencode(
                        '.jpg', stack.picture(), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                           + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                    await asyncio.sleep(max(0.0, 1.0 / FPS - (time.monotonic() - t0)))
            except asyncio.CancelledError:               # the page closed, or Ctrl-C
                return
        return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame',
                                 headers={'Cache-Control': 'no-store'})

    @app.get('/snapshot.jpg')
    def snapshot():
        from fastapi.responses import Response
        jpg = cv2.imencode('.jpg', stack.picture(), [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
        return Response(jpg.tobytes(), media_type='image/jpeg',
                        headers={'Cache-Control': 'no-store'})

    @app.get('/api/events')
    async def events(request: Request):
        async def push():
            try:
                while not await request.is_disconnected():
                    yield f'data: {json.dumps(stack.snapshot(), default=str)}\n\n'
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                return
        return StreamingResponse(push(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-store'})

    @app.get('/api/state')
    def state():
        return JSONResponse(stack.snapshot())

    @app.post('/api/click')
    async def click(request: Request):
        body = await request.json()
        x, y = float(body.get('x', -1)), float(body.get('y', -1))
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return JSONResponse({'ok': False, 'message': 'outside the picture'})
        ok, message = stack.click(x, y)
        return JSONResponse({'ok': ok, 'message': message})

    @app.post('/api/action')
    async def action(request: Request):
        body = await request.json()
        try:
            ok, message = stack.action(str(body.get('name', '')), body)
        except Exception as exc:                     # noqa: BLE001 - shown on the page
            ok, message = False, str(exc)
        return JSONResponse({'ok': ok, 'message': message})

    return app


def make_parser():
    parser = pick_stack.make_parser()
    parser.description = __doc__.split('\n')[0]
    parser.add_argument('--host', default='0.0.0.0',
                        help='where the page listens (127.0.0.1: only on this machine)')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--click', action='store_true',
                        help='click-only, like click_to_move: no box model, no DETECT BOX '
                             'or AUTO PICK (ready_to_pick on arm_action is refused)')
    parser.set_defaults(gui='off')
    return parser


def main():
    args = make_parser().parse_args(remove_ros_args(sys.argv)[1:])
    ctm_app.finish_args(args)
    rclpy.init(args=sys.argv)
    node = ClickToMove(args)
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    stack = WebStack(node)
    deadline = time.monotonic() + 10.0
    while node.arm_positions('left') is None and time.monotonic() < deadline:
        time.sleep(0.2)
    stack.publish_state()
    import uvicorn
    node.get_logger().info(f'web GUI on http://{args.host}:{args.port} '
                           f'(arm_state: {stack.snapshot()["arm_state"]})')
    try:
        uvicorn.run(make_app(stack), host=args.host, port=args.port, log_level='warning',
                    lifespan='off')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
        except BaseException:                        # noqa: BLE001 - a second Ctrl-C
            pass
        rclpy.try_shutdown()
        os._exit(0)


if __name__ == '__main__':
    main()

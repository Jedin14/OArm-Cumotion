"""sensor_msgs/PointCloud2 <-> numpy without sensor_msgs_py (works with numpy 1.x and 2.x)."""

import array

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

_NP = {
    PointField.INT8: np.int8, PointField.UINT8: np.uint8,
    PointField.INT16: np.int16, PointField.UINT16: np.uint16,
    PointField.INT32: np.int32, PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64,
}
_PF = {np.dtype(v): k for k, v in _NP.items()}


def to_struct(msg: PointCloud2) -> np.ndarray:
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype not in _NP or f.count != 1:
            continue
        names.append(f.name)
        dt = np.dtype(_NP[f.datatype])
        formats.append(dt.newbyteorder(">") if msg.is_bigendian else dt.newbyteorder("<"))
        offsets.append(f.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                      "itemsize": msg.point_step})
    n = msg.width * msg.height
    if n == 0:
        return np.zeros(0, dtype=dtype)
    if msg.row_step == msg.width * msg.point_step:
        return np.frombuffer(memoryview(msg.data).cast("B"), dtype=dtype, count=n)
    rows = [np.frombuffer(bytes(msg.data[r * msg.row_step:(r * msg.row_step) + msg.width * msg.point_step]),
                          dtype=dtype, count=msg.width) for r in range(msg.height)]
    return np.concatenate(rows)


def xyz(arr: np.ndarray) -> np.ndarray:
    p = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    return p


def field_or(arr: np.ndarray, name: str, default: float = 0.0) -> np.ndarray:
    if arr.dtype.names and name in arr.dtype.names:
        return np.asarray(arr[name], dtype=np.float64)
    return np.full(arr.shape[0], default, dtype=np.float64)


def make_cloud(header, columns: dict) -> PointCloud2:
    """columns: ordered dict name -> 1-D array (dtype decides the PointField type)."""
    names = list(columns.keys())
    arrs = [np.ascontiguousarray(columns[k]) for k in names]
    n = arrs[0].shape[0] if arrs else 0
    dtype = np.dtype([(k, a.dtype.newbyteorder("<")) for k, a in zip(names, arrs)])
    rec = np.empty(n, dtype=dtype)
    for k, a in zip(names, arrs):
        rec[k] = a
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.is_bigendian = False
    msg.is_dense = True
    msg.point_step = dtype.itemsize
    msg.row_step = dtype.itemsize * n
    msg.fields = [PointField(name=k, offset=dtype.fields[k][1],
                             datatype=_PF[np.dtype(dtype.fields[k][0]).newbyteorder("=")], count=1)
                  for k in names]
    buf = array.array("B")
    buf.frombytes(rec.tobytes())
    msg.data = buf
    return msg

from __future__ import annotations

from typing import Optional

import numpy as np
from sensor_msgs.msg import Image, LaserScan


def laserscan_to_numpy(msg: LaserScan, *, clip: bool = True) -> np.ndarray:
    """
    LaserScan -> 1D float32 numpy array (ranges).
    - clip=True: 将 inf/nan 裁剪/替换到 [range_min, range_max]
    """
    ranges = np.asarray(msg.ranges, dtype=np.float32)
    if not clip:
        return ranges

    rmin = float(msg.range_min) if msg.range_min > 0 else 0.0
    rmax = float(msg.range_max) if msg.range_max > 0 else float(np.nanmax(ranges[np.isfinite(ranges)])) if np.any(np.isfinite(ranges)) else 0.0

    ranges = np.nan_to_num(ranges, nan=rmax, posinf=rmax, neginf=rmin)
    if rmax > rmin:
        ranges = np.clip(ranges, rmin, rmax)
    return ranges.astype(np.float32, copy=False)


def image_to_numpy(msg: Image) -> np.ndarray:
    """
    sensor_msgs/Image -> numpy
    支持常见 encoding：rgb8/bgr8/rgba8/bgra8/mono8/16UC1/32FC1
    """
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    data = msg.data

    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        if enc == "bgr8":
            arr = arr[..., ::-1]
        return arr
    if enc in ("rgba8", "bgra8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))[..., :3]
        if enc == "bgra8":
            arr = arr[..., ::-1]
        return arr
    if enc == "mono8":
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w))
    if enc == "16uc1":
        return np.frombuffer(data, dtype=np.uint16).reshape((h, w))
    if enc == "32fc1":
        return np.frombuffer(data, dtype=np.float32).reshape((h, w))

    return np.frombuffer(data, dtype=np.uint8)


def image_to_torch(arr: np.ndarray, *, device: Optional[str] = None):
    """
    numpy -> torch tensor（延迟依赖 torch；没装 torch 会抛 ImportError）
    默认输出 CHW float32，范围 [0,1]（若输入为 uint8 彩色图）
    """
    import torch  # type: ignore

    t = torch.from_numpy(arr)
    if t.dtype == torch.uint8 and t.ndim == 3:
        t = t.permute(2, 0, 1).contiguous().float() / 255.0
    elif t.ndim == 2:
        t = t.unsqueeze(0).contiguous()

    if device is not None:
        t = t.to(device)
    return t


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import numpy as np


def _wrap_pi(a: float) -> float:
    """wrap angle to (-pi, pi]."""
    x = (float(a) + math.pi) % (2.0 * math.pi) - math.pi
    return x


@dataclass(frozen=True)
class RoadMap:
    """
    Polyline centerline road map (in odom/world frame).

    - centerline_xy: (M,2) points in meters
    - half_width_m: lane half width (for in-road check)
    """

    name: str
    world: str
    centerline_xy: np.ndarray
    half_width_m: float

    # precomputed
    s_of_wp: np.ndarray  # (M,) cumulative arc length

    @staticmethod
    def load(path: str | Path) -> "RoadMap":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"road map not found: {p}")
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "缺少依赖 PyYAML，无法读取 road_map yaml。请先 `pip install pyyaml`。"
            ) from e

        data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
        name = str(data.get("name", p.stem))
        world = str(data.get("world", ""))
        road = data.get("road", {}) if isinstance(data.get("road"), dict) else {}
        hw = float(road.get("half_width_m", 0.6))
        pts = road.get("centerline_xy", [])
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] < 2:
            raise ValueError("centerline_xy 至少需要 2 个点")
        s = np.zeros((arr.shape[0],), dtype=np.float32)
        for i in range(1, arr.shape[0]):
            s[i] = s[i - 1] + float(np.linalg.norm(arr[i] - arr[i - 1]))
        return RoadMap(name=name, world=world, centerline_xy=arr, half_width_m=hw, s_of_wp=s)

    @property
    def s_end(self) -> float:
        return float(self.s_of_wp[-1])

    def project(self, *, xy: tuple[float, float]) -> tuple[float, float, float]:
        """
        Project a point to centerline polyline.
        Returns: (s, cte, tangent_yaw)
        cte sign: + means point is on the left of tangent direction.
        """
        x, y = float(xy[0]), float(xy[1])
        p = np.asarray([x, y], dtype=np.float32)
        pts = self.centerline_xy
        best_d2 = float("inf")
        best_s = 0.0
        best_cte = 0.0
        best_yaw = 0.0
        for i in range(pts.shape[0] - 1):
            a = pts[i]
            b = pts[i + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 < 1e-8:
                continue
            t = float(np.dot(p - a, ab) / ab2)
            t = max(0.0, min(1.0, t))
            q = a + t * ab
            d = p - q
            d2 = float(np.dot(d, d))
            if d2 < best_d2:
                best_d2 = d2
                seg_len = float(np.linalg.norm(ab))
                s0 = float(self.s_of_wp[i])
                best_s = s0 + t * seg_len
                # tangent yaw
                best_yaw = math.atan2(float(ab[1]), float(ab[0]))
                # signed cte via 2D cross (ab x (p-a))
                cross = float(ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0]))
                sign = 1.0 if cross > 0.0 else (-1.0 if cross < 0.0 else 0.0)
                best_cte = sign * math.sqrt(max(0.0, d2))
        return float(best_s), float(best_cte), float(best_yaw)

    def lookahead_points(
        self, *, s: float, n: int, ds: float
    ) -> np.ndarray:
        """
        Sample N points ahead along centerline, returning (N,2) in map frame.
        """
        n = int(max(0, n))
        ds = float(ds)
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        s_targets = np.asarray([float(s) + (i + 1) * ds for i in range(n)], dtype=np.float32)
        s_targets = np.clip(s_targets, 0.0, float(self.s_end))
        out = np.zeros((n, 2), dtype=np.float32)
        pts = self.centerline_xy
        s_wp = self.s_of_wp
        j = 0
        for i in range(n):
            st = float(s_targets[i])
            while (j + 1) < s_wp.shape[0] and float(s_wp[j + 1]) < st:
                j += 1
            if j >= pts.shape[0] - 1:
                out[i] = pts[-1]
                continue
            s0 = float(s_wp[j])
            s1 = float(s_wp[j + 1])
            a = pts[j]
            b = pts[j + 1]
            if s1 <= s0 + 1e-8:
                out[i] = a
                continue
            t = (st - s0) / (s1 - s0)
            out[i] = a + float(t) * (b - a)
        return out

    @staticmethod
    def road_feat_to_body_frame(
        *,
        ego_xy: tuple[float, float],
        ego_yaw: float,
        points_xy: np.ndarray,
    ) -> np.ndarray:
        """
        Transform points from map frame to body frame.
        Returns (N,2).
        """
        if points_xy.size == 0:
            return points_xy.astype(np.float32, copy=False)
        px = points_xy.astype(np.float32, copy=False)
        ex, ey = float(ego_xy[0]), float(ego_xy[1])
        dx = px[:, 0] - ex
        dy = px[:, 1] - ey
        cy = math.cos(-float(ego_yaw))
        sy = math.sin(-float(ego_yaw))
        bx = cy * dx - sy * dy
        by = sy * dx + cy * dy
        return np.stack([bx, by], axis=1).astype(np.float32, copy=False)

    @staticmethod
    def heading_error(*, ego_yaw: float, tangent_yaw: float) -> float:
        return float(_wrap_pi(float(tangent_yaw) - float(ego_yaw)))


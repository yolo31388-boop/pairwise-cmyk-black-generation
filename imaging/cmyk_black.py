"""RGB -> CMYK 分色与黑版生成管线。

修复的六类印刷缺陷：

1. ``tic_constraint``      —— 总墨量约束（TIC）。不再逐通道硬裁，而是
   等比缩放 CMY、保持灰平衡，仅当 K 单通道超限时才压 K。
2. ``region_gcr``          —— 分区 GCR。按像素分为肤色 / 金属色 / 暗调 /
   普通区，各自使用不同 GCR 比例：肤色低 GCR 不发灰，金属色低 GCR 保光泽，
   暗调高 GCR 省墨、保证中性灰。
3. ``sync_sharpen``        —— 通道同步锐化。锐化核与偏移对 C/M/Y/K 完全
   相同，黑版边缘不再比 CMY 早一像素，杜绝套印彩边。
4. ``gray_balance_rebuild`` —— 灰平衡重建。对近中性像素按 C:M:Y 灰平衡
   配比（C 略高于 M、M 略高于 Y）重建，TIC 压缩后仍保持中性灰。
5. ``paper_white_adapt``   —— 纸白适配。Bradford 色适应把源纸白点映射到
   目标纸白点（相对比色意图下媒体白自动对齐），不同纸张不再偏色。
6. ``page_order_bind``     —— 页序绑定。页码在提交时即冻结到输出文件名与
   清单，并发完成顺序不影响最终页序，杜绝页码与输出错位。

所有量约定：RGB、CMYK 通道取值均为 [0, 1] 浮点（uint8 输入自动归一化），
CMYK 用各通道墨量百分比的小数形式。
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 全局工艺常量
# ---------------------------------------------------------------------------

DEFAULT_TIC = 3.0
"""默认总墨量上限（300%），胶印典型值。"""

# 分区 GCR 比例：黑版吸收中性灰成分的比例。
GCR_SKIN = 0.25      # 肤色：低 GCR，避免发灰
GCR_METALLIC = 0.30  # 金属色：低 GCR，保留 CMY 叠印光泽
GCR_SHADOW = 0.90    # 暗调：高 GCR，省墨并稳定中性灰
GCR_DEFAULT = 0.60   # 普通区：中等 GCR

REGION_SKIN = 0
REGION_METALLIC = 1
REGION_SHADOW = 2
REGION_DEFAULT = 3
REGION_NAMES = ("skin", "metallic", "shadow", "default")

# 中性灰灰平衡配比（C:M:Y），C 略重以补偿胶印中 C 墨密度偏低的特性。
GRAY_BALANCE_C = 1.00
GRAY_BALANCE_M = 0.90
GRAY_BALANCE_Y = 0.82
NEUTRAL_CHROMA_TOL = 0.12   # max(C,M,Y)-min <= 该值视为近中性
NEUTRAL_MIN_DEPTH = 0.08    # 中性灰重建的最小墨量，纸白不处理

# D50（印刷标准观察环境）与 sRGB/D65 白点。
_D50_XYZ = np.array([0.9642, 1.0, 0.8249], dtype=np.float64)
_D65_XYZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

# 保护 manifest.json 读-改-写的进程内锁，多页并发落盘时避免清单撕裂。
_MANIFEST_LOCK = threading.Lock()

_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)
_BRADFORD_INV = np.linalg.inv(_BRADFORD)

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _as_float_image(arr: Any, channels: int) -> Optional[np.ndarray]:
    """把输入规整为 HxWxC、取值 [0,1] 的 float64 数组；不可识别返回 None。"""
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
    except Exception:
        return None
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[2] != channels:
        return None
    if not np.issubdtype(a.dtype, np.floating):
        if np.issubdtype(a.dtype, np.integer):
            a = a.astype(np.float64) / np.iinfo(a.dtype).max
        else:
            return None
    else:
        a = a.astype(np.float64)
    return np.clip(a, 0.0, 1.0)


def _is_image_like(obj: Any, channels: int) -> bool:
    return _as_float_image(obj, channels) is not None


def _as_ink_image(arr: Any, channels: int = 4, cap: float = 4.0) -> Optional[np.ndarray]:
    """墨量数组规整化。

    与 :func:`_as_float_image` 的区别：TIC 约束是兜底环节，必须能接收
    临时超过 100% 的单通道墨量，因此上限放宽到 ``cap``（默认 400%），
    只剔除负值与 NaN，不能在入口处把超限像素裁掉。
    """
    try:
        a = np.asarray(arr, dtype=np.float64)
    except Exception:
        return None
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[2] != channels:
        return None
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=cap, neginf=0.0), 0.0, cap)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _gaussian_kernel1d(radius: int, sigma: float) -> np.ndarray:
    """确定性一维高斯核（纯 numpy，无随机、无平台差异）。"""
    if radius <= 0:
        return np.array([1.0], dtype=np.float64)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _separable_blur(plane: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    """反射边界的可分离高斯模糊，所有通道调用时参数完全一致。"""
    if radius <= 0:
        return plane.copy()
    k = _gaussian_kernel1d(radius, sigma)
    pad = radius
    padded = np.pad(plane, ((pad, pad), (pad, pad)), mode="reflect")
    # 水平方向相关：输出宽度还原为原平面宽度
    acc = np.zeros((padded.shape[0], plane.shape[1]), dtype=plane.dtype)
    for i, w in enumerate(k):
        acc += w * padded[:, i:i + plane.shape[1]]
    # 垂直方向相关：输出高度还原为原平面高度
    out = np.zeros_like(plane)
    for i, w in enumerate(k):
        out += w * acc[i:i + plane.shape[0], :]
    return out


# ---------------------------------------------------------------------------
# 1) 分区 GCR
# ---------------------------------------------------------------------------

def classify_regions(rgb: np.ndarray) -> np.ndarray:
    """像素分区图，取值为 REGION_* 常量。

    判定优先级：肤色 > 金属色 > 暗调 > 普通。
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = _luminance(rgb)
    chroma = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)

    # 肤色：R 主导、G 居中、B 最弱，亮度中等偏亮，低饱和暖色。
    skin = (
        (r > 0.35)
        & (r > g)
        & (g > b)
        & (r - b > 0.08)
        & (lum > 0.25)
        & (chroma < 0.45)
    )
    # 金属色：高亮、低饱和、非中性（带轻微色相）。
    metallic = (~skin) & (lum > 0.55) & (chroma > 0.03) & (chroma < 0.30)
    # 暗调：亮度低，无论饱和度。
    shadow = (~skin) & (~metallic) & (lum < 0.25)

    regions = np.full(rgb.shape[:2], REGION_DEFAULT, dtype=np.uint8)
    regions[shadow] = REGION_SHADOW
    regions[metallic] = REGION_METALLIC
    regions[skin] = REGION_SKIN
    return regions


def region_gcr(rgb_or_cmy: Any, regions: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """分区灰成分替代（GCR），输出初始 C/M/Y/K 墨量。

    参数
    ----
    rgb_or_cmy:
        HxWx3 的 RGB 图像（uint8 或 [0,1] 浮点）。也接受已算好的 CMY。
    regions:
        可选的预计算分区图；缺省时用 :func:`classify_regions` 现场计算。

    返回
    ----
    dict，含 ``cmyk``（HxWx4）、``regions``、``gcr_map``（逐像素实际 GCR 比例）。
    """
    rgb = _as_float_image(rgb_or_cmy, 3)
    if rgb is None:
        return _unavailable("region_gcr")

    if regions is None:
        regions = classify_regions(rgb)
    regions = np.asarray(regions, dtype=np.uint8)

    gcr_table = np.array(
        [GCR_SKIN, GCR_METALLIC, GCR_SHADOW, GCR_DEFAULT], dtype=np.float64
    )
    gcr_map = gcr_table[np.clip(regions, 0, 3)]

    # 朴素补色：CMY = 1 - RGB
    cmy = 1.0 - rgb
    c, m, y = cmy[..., 0], cmy[..., 1], cmy[..., 2]
    gray = np.minimum(np.minimum(c, m), y)

    # 黑版只吸收各分区比例内的中性成分 —— 不再全额换黑。
    k = gcr_map * gray
    under = k  # UCR：从 CMY 中等量移除的灰成分
    cmyk = np.stack([c - under, m - under, y - under, k], axis=-1)
    cmyk = np.clip(cmyk, 0.0, 1.0)
    return {"cmyk": cmyk, "regions": regions, "gcr_map": gcr_map}


# ---------------------------------------------------------------------------
# 2) 总墨量约束（TIC）
# ---------------------------------------------------------------------------

def tic_constraint(cmyk: Any, tic_limit: float = DEFAULT_TIC) -> Dict[str, np.ndarray]:
    """总墨量约束，保持灰平衡的等比压缩，而非逐通道裁剪。

    规则：
      * C+M+Y+K <= tic_limit 的像素不动；
      * 超限时 K 保持不变，CMY 三通道按**同一比例**缩小（保持色相与灰平衡）；
      * 若 K 本身就超过 tic_limit（极端情况），先把 K 压到 tic_limit。
    """
    data = _as_ink_image(cmyk, 4)
    if data is None:
        return _unavailable("tic_constraint")

    out = data.copy()
    k = out[..., 3]

    # K 单通道兜底压缩。
    k_over = k > tic_limit
    if np.any(k_over):
        out[k_over, 3] = tic_limit
        k = out[..., 3]

    total = out.sum(axis=-1)
    over = total > tic_limit
    if np.any(over):
        cmy_sum = out[..., :3].sum(axis=-1)
        # 目标 CMY 总量 = tic_limit - K；同比例因子作用于三个通道。
        target_cmy = np.maximum(tic_limit - k[over], 0.0)
        factor = np.ones_like(total)
        denom = np.where(cmy_sum > 1e-12, cmy_sum, 1.0)
        factor[over] = np.clip(target_cmy / denom[over], 0.0, 1.0)
        out[..., :3] *= factor[..., None]

    # 不用 [0,1] 裁剪：墨量允许单通道到 100%，且 K 兜底压缩可能产生
    # tic_limit 以内但大于 1 的中间值；真正的工艺约束是上面的 TIC 逻辑。
    out = np.clip(out, 0.0, max(1.0, tic_limit))
    return {
        "cmyk": out,
        "tic": out.sum(axis=-1),
        "violation_mask": over,
        "tic_limit": float(tic_limit),
    }


# ---------------------------------------------------------------------------
# 3) 灰平衡重建
# ---------------------------------------------------------------------------

def gray_balance_rebuild(cmyk: Any, tol: float = NEUTRAL_CHROMA_TOL) -> Dict[str, np.ndarray]:
    """对近中性像素按 C:M:Y 灰平衡配比重建 CMY。

    TIC 等比压缩会同时缩小 CMY 与 K 的比例关系，本函数在压缩之后执行，
    只调整近中性像素，把其中性部分重新分配为标准灰平衡配比；
    彩色像素（色度超差）原样保留，避免污染色相。
    """
    data = _as_ink_image(cmyk, 4)
    if data is None:
        return _unavailable("gray_balance_rebuild")

    out = data.copy()
    c, m, y, k = out[..., 0], out[..., 1], out[..., 2], out[..., 3]
    cmy = out[..., :3]
    cmy_max = cmy.max(axis=-1)
    cmy_min = cmy.min(axis=-1)
    neutral = ((cmy_max - cmy_min) <= tol) & (cmy_max >= NEUTRAL_MIN_DEPTH)

    if np.any(neutral):
        # 中性部分的目标总量保持不变（不额外增加墨量），只改三通道配比。
        neutral_amount = cmy[neutral].mean(axis=-1)
        balance = np.array(
            [GRAY_BALANCE_C, GRAY_BALANCE_M, GRAY_BALANCE_Y], dtype=np.float64
        )
        rebuilt = neutral_amount[..., None] * (balance / balance.mean())
        out[neutral, :3] = np.clip(rebuilt, 0.0, 1.0)

    return {
        "cmyk": np.clip(out, 0.0, 1.0),
        "neutral_mask": neutral,
        "balance_ratio": (GRAY_BALANCE_C, GRAY_BALANCE_M, GRAY_BALANCE_Y),
    }


# ---------------------------------------------------------------------------
# 4) 通道同步锐化
# ---------------------------------------------------------------------------

def _unsharp_plane(plane: np.ndarray, blurred: np.ndarray, amount: float,
                   edge: np.ndarray) -> np.ndarray:
    # 四个通道共用同一 blurred 计算参数与同一 edge 权重，偏移天然一致。
    sharpened = plane + amount * edge * (plane - blurred)
    return np.clip(sharpened, 0.0, 1.0)


def sync_sharpen(cmyk: Any, radius: int = 2, sigma: float = 1.0,
                 amount: float = 0.8) -> Dict[str, np.ndarray]:
    """四通道同步 USM 锐化。

    关键点：
      * 同一个高斯核（radius/sigma 相同）、同一套反射边界、同一组像素偏移
        作用于 C/M/Y/K，黑版不会比 CMY 早或晚一像素；
      * 锐化权重取自全图亮度边缘（而非各通道各自的边缘），进一步保证
        四通道在同一批像素上做同强度增强，套印不出现彩边。
    """
    data = _as_float_image(cmyk, 4)
    if data is None:
        return _unavailable("sync_sharpen")

    # 边缘权重由四通道的平均墨量（近似中性密度）统一计算，所有通道共享。
    density = data.mean(axis=-1)
    blurred_density = _separable_blur(density, radius, sigma)
    edge = np.clip(np.abs(density - blurred_density) * 4.0, 0.0, 1.0)

    out = np.empty_like(data)
    for ch in range(4):
        blurred = _separable_blur(data[..., ch], radius, sigma)
        out[..., ch] = _unsharp_plane(data[..., ch], blurred, amount, edge)

    return {"cmyk": np.clip(out, 0.0, 1.0), "edge_map": edge,
            "kernel_radius": int(radius), "kernel_sigma": float(sigma)}


# ---------------------------------------------------------------------------
# 5) 纸白适配（Bradford 色适应）
# ---------------------------------------------------------------------------

def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )


def _srgb_decode(srgb: np.ndarray) -> np.ndarray:
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power(np.maximum((srgb + 0.055) / 1.055, 0.0), 2.4),
    )


def _chromatic_adaptation_matrix(src_white: np.ndarray,
                                 dst_white: np.ndarray) -> np.ndarray:
    """Bradford CAT：把 src 白点下的 XYZ 变换到 dst 白点下。"""
    src_cone = _BRADFORD @ src_white
    dst_cone = _BRADFORD @ dst_white
    scale = np.diag(dst_cone / src_cone)
    return _BRADFORD_INV @ scale @ _BRADFORD


def paper_white_adapt(rgb: Any,
                      src_white: Sequence[float] = (0.95047, 1.0, 1.08883),
                      dst_white: Optional[Sequence[float]] = None,
                      intent: str = "relative-colorimetric") -> Dict[str, Any]:
    """纸白适配：把源纸白上的 sRGB 像素色适应到目标纸白。

    相对比色意图下媒体白必须对齐（源白点 -> 目标纸白点），这正是旧实现
    “固定 ICC 意图、不处理纸白” 导致跨纸张偏色的根因。dst_white 缺省为
    D50 胶印纸白；矩阵奇异等异常场景退回到每通道增益法，保证不崩。
    """
    data = _as_float_image(rgb, 3)
    if data is None:
        return _unavailable("paper_white_adapt")

    src = np.asarray(src_white, dtype=np.float64)
    dst = _D50_XYZ if dst_white is None else np.asarray(dst_white, dtype=np.float64)
    intent = intent or "relative-colorimetric"

    linear = _srgb_decode(data)
    xyz = linear @ _SRGB_TO_XYZ.T

    try:
        cat = _chromatic_adaptation_matrix(src, dst)
        if not np.all(np.isfinite(cat)) or np.linalg.matrix_rank(cat) < 3:
            raise np.linalg.LinAlgError("singular CAT")
        xyz_adapted = xyz @ cat.T
        method = "bradford"
    except np.linalg.LinAlgError:
        gain = dst / np.where(src > 1e-9, src, 1.0)
        xyz_adapted = xyz * gain
        method = "channel-gain-fallback"

    xyz_to_rgb = np.linalg.inv(_SRGB_TO_XYZ)
    linear_out = xyz_adapted @ xyz_to_rgb.T
    if intent.startswith("relative"):
        # 相对比色的媒体白对齐在输出线性 RGB 域归一化：目标纸白对应的
        # 线性 RGB 被拉到 (1,1,1)，保证输出编码中纸白精确为纯白，
        # 同时有色像素完整保留 Bradford 色适应带来的色相/密度变化。
        white_linear = dst @ xyz_to_rgb.T
        linear_out = linear_out / np.where(np.abs(white_linear) > 1e-9,
                                           white_linear, 1.0)
    out = np.clip(_srgb_encode(np.clip(linear_out, 0.0, None)), 0.0, 1.0)

    return {
        "rgb": out,
        "src_white": src,
        "dst_white": dst,
        "intent": intent,
        "method": method,
    }


# ---------------------------------------------------------------------------
# 6) 页序绑定
# ---------------------------------------------------------------------------

@dataclass
class _PageRecord:
    page: int
    payload: Any
    name: str
    digest: str


class PageBinder:
    """线程安全的页序绑定器。

    页码在 :meth:`submit` 时即冻结进文件名与摘要，工作线程无论以什么顺序
    完成，:meth:`manifest` / :meth:`pages` 永远按页码升序输出。
    """

    def __init__(self, prefix: str = "page", digits: int = 4):
        self.prefix = prefix
        self.digits = digits
        self._records: Dict[int, _PageRecord] = {}
        self._lock = threading.Lock()

    def page_name(self, page: int) -> str:
        return f"{self.prefix}-{int(page):0{self.digits}d}.tif"

    def submit(self, page: int, payload: Any) -> Dict[str, Any]:
        page = int(page)
        token = repr(payload).encode("utf-8", errors="replace")
        digest = hashlib.sha256(token).hexdigest()[:16]
        name = self.page_name(page)
        with self._lock:
            if page in self._records:
                raise ValueError(f"duplicate page number: {page}")
            self._records[page] = _PageRecord(page, payload, name, digest)
        return {"page": page, "name": name, "digest": digest}

    def pages(self) -> List[Dict[str, Any]]:
        with self._lock:
            records = sorted(self._records.values(), key=lambda r: r.page)
            return [
                {"page": r.page, "name": r.name, "digest": r.digest,
                 "payload": r.payload}
                for r in records
            ]

    def manifest(self) -> List[Dict[str, Any]]:
        return [{"page": r["page"], "name": r["name"], "digest": r["digest"]}
                for r in self.pages()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def page_order_bind(pages: Any, prefix: str = "page",
                    digits: int = 4) -> Dict[str, Any]:
    """批量页序绑定。

    接受：dict（页码 -> 页面数据）、可迭代的 ``(page, payload)``、或单页
    ``(page, payload)``。内部用 :class:`PageBinder` 冻结页码与文件名的关系，
    并校验缺页 / 重号；返回按页序排好的清单，与完成先后无关。
    """
    binder = PageBinder(prefix=prefix, digits=digits)

    if isinstance(pages, Mapping):
        items = [(int(k), v) for k, v in pages.items()]
    elif isinstance(pages, tuple) and len(pages) == 2 and np.isscalar(pages[0]):
        items = [pages]
    elif isinstance(pages, Iterable):
        items = []
        for item in pages:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                return _unavailable("page_order_bind")
            items.append((int(item[0]), item[1]))
    else:
        return _unavailable("page_order_bind")

    seen: set = set()
    duplicates: List[int] = []
    bound: List[Dict[str, Any]] = []
    for page, payload in items:
        if page in seen:
            duplicates.append(page)
            continue
        seen.add(page)
        bound.append(binder.submit(page, payload))

    ordered = sorted(seen)
    expected = list(range(min(ordered), max(ordered) + 1)) if ordered else []
    missing = [p for p in expected if p not in seen]

    return {
        "manifest": binder.manifest(),
        "pages": binder.pages(),
        "duplicates": duplicates,
        "missing": missing,
        "bound": len(bound),
    }


# ---------------------------------------------------------------------------
# 端到端管线
# ---------------------------------------------------------------------------

def load_rgb(source: Any) -> Dict[str, Any]:
    """加载 RGB 图像。

    支持 numpy 数组、文件路径、类文件对象、PIL Image。无法识别的输入
    返回 ``ok=False`` 的结构化结果而不是抛异常，供 fallback 环节接管。
    """
    arr = _as_float_image(source, 3)
    if arr is not None:
        return {"ok": True, "rgb": arr, "meta": {"source": "ndarray",
                                                 "shape": list(arr.shape)}}
    try:
        from PIL import Image
    except ImportError:
        return _unavailable("load_rgb", reason="Pillow unavailable")

    try:
        if isinstance(source, (str, os.PathLike)):
            img = Image.open(os.fspath(source))
            meta = {"source": os.fspath(source)}
        elif hasattr(source, "read"):
            img = Image.open(io.BytesIO(source.read()))
            meta = {"source": "stream"}
        else:
            img = source  # 当作 PIL Image
            meta = {"source": "pil-image"}
        rgb_img = img.convert("RGB")
        arr = np.asarray(rgb_img, dtype=np.float64) / 255.0
        meta.update({"size": list(rgb_img.size), "mode": "RGB"})
        return {"ok": True, "rgb": arr, "meta": meta}
    except Exception as exc:  # 任何加载失败都走结构化降级
        return _unavailable("load_rgb", reason=str(exc))


def convert_cmyk(rgb: Any, tic_limit: float = DEFAULT_TIC,
                 sharpen_amount: float = 0.8,
                 src_white: Optional[Sequence[float]] = None,
                 dst_white: Optional[Sequence[float]] = None,
                 intent: str = "relative-colorimetric") -> Dict[str, Any]:
    """完整分色管线：纸白适配 -> 分区 GCR -> TIC -> 灰平衡 -> 同步锐化。

    顺序刻意如此安排：先在 RGB 域做纸白适配；GCR 产生黑版；TIC 压缩后
    立刻重建灰平衡；最后四通道同步锐化，锐化输出再过一次 TIC 兜底，
    保证 USM 振铃不会重新冲破墨量上限。
    """
    arr = _as_float_image(rgb, 3)
    if arr is None:
        loaded = load_rgb(rgb)
        if not loaded.get("ok"):
            return _unavailable("convert_cmyk", reason=loaded.get("reason"))
        arr = loaded["rgb"]

    adapt = paper_white_adapt(
        arr,
        src_white=tuple(src_white) if src_white is not None else tuple(_D65_XYZ),
        dst_white=dst_white,
        intent=intent,
    )
    gcr = region_gcr(adapt["rgb"])
    tic = tic_constraint(gcr["cmyk"], tic_limit)
    bal = gray_balance_rebuild(tic["cmyk"])
    sharp = sync_sharpen(bal["cmyk"], amount=sharpen_amount)
    # 锐化兜底：振铃可能抬升局部墨量。
    final = tic_constraint(sharp["cmyk"], tic_limit)

    return {
        "ok": True,
        "cmyk": final["cmyk"],
        "regions": gcr["regions"],
        "gcr_map": gcr["gcr_map"],
        "tic_map": final["tic"],
        "neutral_mask": bal["neutral_mask"],
        "edge_map": sharp["edge_map"],
        "paper_adapt": {"method": adapt["method"], "intent": adapt["intent"],
                        "dst_white": adapt["dst_white"]},
        "tic_limit": float(tic_limit),
    }


def proof_preview(cmyk: Any) -> Dict[str, Any]:
    """CMYK 软打样：朴素叠印回 RGB，仅用于屏幕预览。"""
    data = _as_float_image(cmyk, 4)
    if data is None:
        return _unavailable("proof_preview")
    c, m, y, k = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    rgb = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    return {"rgb": rgb, "cmyk": data}


def present_print(result: Any, out_dir: Optional[str] = None, page: int = 1,
                  prefix: str = "page", digits: int = 4) -> Dict[str, Any]:
    """把分色结果落盘为 TIFF（每页一个文件，文件名绑定页码）并写清单。"""
    cmyk = result.get("cmyk") if isinstance(result, Mapping) else result
    data = _as_float_image(cmyk, 4)
    if data is None:
        return _unavailable("present_print")
    try:
        from PIL import Image
    except ImportError:
        return _unavailable("present_print", reason="Pillow unavailable")

    os.makedirs(out_dir, exist_ok=True)
    binder = PageBinder(prefix=prefix, digits=digits)
    binding = binder.submit(page, data.tobytes())
    path = os.path.join(out_dir, binding["name"])

    img = Image.fromarray(np.round(data * 255.0).astype(np.uint8), mode="CMYK")
    img.save(path, format="TIFF")
    import json
    manifest_path = os.path.join(out_dir, "manifest.json")
    # 整段读-改-写加锁：并发 worker 不会互相覆盖或写出撕裂的 JSON。
    with _MANIFEST_LOCK:
        manifest = []
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
            except (ValueError, OSError):
                manifest = []
        if not any(entry.get("page") == page for entry in manifest):
            manifest.append(binding)
        manifest.sort(key=lambda e: e["page"])
        tmp_path = manifest_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, manifest_path)

    return {"ok": True, "path": path, "manifest": manifest_path,
            "page": page, "name": binding["name"], "digest": binding["digest"]}


def validate_print(result: Any, tic_limit: float = DEFAULT_TIC) -> Dict[str, Any]:
    """印前校验：墨量范围、TIC、NaN、页绑定一致性。"""
    if not isinstance(result, Mapping) or "cmyk" not in result:
        return _unavailable("validate_print")
    data = _as_float_image(result["cmyk"], 4)
    if data is None:
        return _unavailable("validate_print")

    finite = bool(np.all(np.isfinite(data)))
    in_range = bool(np.all((data >= 0.0) & (data <= 1.0)))
    tic = data.sum(axis=-1)
    tic_max = float(tic.max())
    tic_ok = tic_max <= tic_limit + 1e-9
    # 中性灰检查：近中性像素 CMY 配比是否贴近灰平衡。只统计有足够墨量的
    # 像素（近纸白处分母趋零，逐像素相除没有意义），并用总量配比而非
    # 逐像素比值的平均，避免被个别浅像素拉偏。
    neutral_ok = True
    if result.get("neutral_mask") is not None:
        mask = np.asarray(result["neutral_mask"], dtype=bool)
        deep = mask & (data[..., :3].max(axis=-1) >= 0.15)
        if deep.any():
            sums = data[..., :3][deep].sum(axis=0)
            share = sums / sums.sum()
            target = np.array(
                [GRAY_BALANCE_C, GRAY_BALANCE_M, GRAY_BALANCE_Y], dtype=np.float64
            )
            target_share = target / target.sum()
            neutral_ok = bool(np.all(np.abs(share - target_share) <= 0.06))

    ok = finite and in_range and tic_ok and neutral_ok
    return {
        "ok": ok,
        "finite": finite,
        "in_range": in_range,
        "tic_ok": tic_ok,
        "tic_max": tic_max,
        "tic_limit": float(tic_limit),
        "neutral_gray_ok": neutral_ok,
    }


def metrics_print(result: Any) -> Dict[str, Any]:
    """工艺度量：TIC 分布、分区占比、平均墨量。"""
    if not isinstance(result, Mapping) or "cmyk" not in result:
        return _unavailable("metrics_print")
    data = _as_float_image(result["cmyk"], 4)
    if data is None:
        return _unavailable("metrics_print")

    tic = data.sum(axis=-1)
    regions = result.get("regions")
    region_share: Dict[str, float] = {}
    if regions is not None:
        regions = np.asarray(regions)
        total = regions.size
        for idx, name in enumerate(REGION_NAMES):
            region_share[name] = float(np.count_nonzero(regions == idx) / total)

    return {
        "tic_mean": float(tic.mean()),
        "tic_p95": float(np.percentile(tic, 95)),
        "tic_max": float(tic.max()),
        "ink_mean": {ch: float(data[..., i].mean())
                     for i, ch in enumerate(("c", "m", "y", "k"))},
        "region_share": region_share,
        "gcr_mean": float(np.asarray(result["gcr_map"]).mean())
        if result.get("gcr_map") is not None else None,
    }


def audit_print(result: Any) -> Dict[str, Any]:
    """审计：对分色结果做内容哈希并固化关键工艺参数，供追溯。"""
    if not isinstance(result, Mapping) or "cmyk" not in result:
        return _unavailable("audit_print")
    data = _as_float_image(result["cmyk"], 4)
    if data is None:
        return _unavailable("audit_print")

    digest = hashlib.sha256(np.round(data * 10000).astype(np.int64).tobytes()).hexdigest()
    return {
        "digest": digest,
        "shape": list(data.shape),
        "tic_limit": result.get("tic_limit", DEFAULT_TIC),
        "paper_adapt": result.get("paper_adapt"),
        "region_names": list(REGION_NAMES),
        "gcr_levels": [GCR_SKIN, GCR_METALLIC, GCR_SHADOW, GCR_DEFAULT],
        "gray_balance": [GRAY_BALANCE_C, GRAY_BALANCE_M, GRAY_BALANCE_Y],
    }


def fallback_print(rgb: Any, tic_limit: float = DEFAULT_TIC) -> Dict[str, Any]:
    """降级路径：完整管线失败时使用的安全朴素分色。

    固定低 GCR、强制 TIC 等比压缩、跳过锐化，保证任何可识别的 RGB 输入
    都能产出可印刷结果；输入本身不可识别时返回结构化失败。
    """
    arr = _as_float_image(rgb, 3)
    if arr is None:
        loaded = load_rgb(rgb)
        if not loaded.get("ok"):
            return _unavailable("fallback_print", reason=loaded.get("reason"))
        arr = loaded["rgb"]

    cmy = 1.0 - arr
    gray = np.min(cmy, axis=-1, keepdims=True)
    k = 0.35 * gray  # 保守固定 GCR
    cmyk = np.concatenate([cmy - k, k], axis=-1)
    safe = tic_constraint(np.clip(cmyk, 0.0, 1.0), tic_limit)
    return {"ok": True, "cmyk": safe["cmyk"], "fallback": True,
            "tic_map": safe["tic"], "tic_limit": float(tic_limit)}


def determinism_print(rgb: Any, runs: int = 3, **kwargs: Any) -> Dict[str, Any]:
    """确定性验证：同一输入多次运行（含并发）结果必须逐位一致。"""
    arr = _as_float_image(rgb, 3)
    if arr is None:
        loaded = load_rgb(rgb)
        if not loaded.get("ok"):
            return _unavailable("determinism_print", reason=loaded.get("reason"))
        arr = loaded["rgb"]

    first = convert_cmyk(arr, **kwargs)
    if not first.get("ok"):
        return _unavailable("determinism_print", reason="conversion failed")
    baseline = first["cmyk"]
    digest0 = hashlib.sha256(baseline.tobytes()).hexdigest()

    identical = True
    digests = [digest0]
    for _ in range(max(1, runs - 1)):
        r = convert_cmyk(arr, **kwargs)
        d = hashlib.sha256(r["cmyk"].tobytes()).hexdigest()
        digests.append(d)
        identical &= (d == digest0) and np.array_equal(r["cmyk"], baseline)

    # 并发页序绑定测试：乱序提交，清单仍按页码升序。
    import random
    pages = list(range(10))
    shuffled = pages.copy()
    random.Random(42).shuffle(shuffled)
    binder = PageBinder()
    errors: List[str] = []
    lock = threading.Lock()

    def worker(pg: int) -> None:
        try:
            binder.submit(pg, arr.tobytes()[:64])
        except Exception as exc:  # pragma: no cover - 防御性
            with lock:
                errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(p,)) for p in shuffled]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    order_ok = [entry["page"] for entry in binder.manifest()] == pages

    return {
        "ok": bool(identical and order_ok and not errors),
        "identical_across_runs": bool(identical),
        "digests": digests,
        "runs": runs,
        "concurrent_page_order_ok": bool(order_ok),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# 内部：不可识别输入的统一结构化返回
# ---------------------------------------------------------------------------

def _unavailable(stage: str, reason: str = "unrecognized input") -> Dict[str, Any]:
    return {"ok": False, "stage": stage, "reason": reason}

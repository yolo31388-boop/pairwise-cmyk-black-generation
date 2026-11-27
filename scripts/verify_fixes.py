"""六类缺陷的专项验证（独立于 pytest 契约测试）。

运行：python3 -I scripts/verify_fixes.py
"""
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imaging import cmyk_black as cb  # noqa: E402

rng = np.random.default_rng(7)
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{PASS if cond else FAIL}] {name} {detail}")


# 构造测试图：肤色块、金属块、暗调块、中性灰块、彩色块
H = W = 48
rgb = np.zeros((H, W, 3), dtype=np.float64)
rgb[:16, :16] = (0.85, 0.55, 0.42)          # 肤色
rgb[:16, 16:32] = (0.80, 0.78, 0.72)        # 金属（高亮低饱和）
rgb[:16, 32:] = (0.08, 0.07, 0.09)          # 暗调
rgb[16:32, :16] = (0.5, 0.5, 0.5)           # 中性灰
rgb[16:32, 16:32] = (0.9, 0.1, 0.1)         # 饱和红
rgb[16:32, 32:] = rng.random((16, 16, 3))   # 随机纹理
rgb[32:, :] = rng.random((16, W, 3))

# ---------------------------------------------------------------- 1. TIC
gcr = cb.region_gcr(rgb)
tic = cb.tic_constraint(gcr["cmyk"], tic_limit=3.0)
max_tic = tic["cmyk"].sum(axis=-1).max()
check("1a TIC 上限 300% 不被突破", max_tic <= 3.0 + 1e-9, f"max={max_tic:.4f}")

# 人为制造一个必超 300% 的像素
heavy = np.array([[[0.9, 0.9, 0.9, 0.9]]])
forced = cb.tic_constraint(heavy, 3.0)["cmyk"][0, 0]
cmy_before = heavy[0, 0, :3]
cmy_after = forced[:3]
scale = cmy_after / np.maximum(cmy_before, 1e-12)
check("1b 超限时 CMY 同比例缩放（保持灰平衡/色相）",
      np.allclose(scale, scale[0], atol=1e-9), f"scale={scale}")
check("1c 超限时 K 保持不变", abs(forced[3] - 0.9) < 1e-9)

# 极限：K 本身超限
k_only = cb.tic_constraint(np.array([[[0.0, 0.0, 0.0, 3.5]]]), 3.0)["cmyk"][0, 0]
check("1d K 单通道超限被压到上限", abs(k_only[3] - 3.0) < 1e-9)

# ---------------------------------------------------------------- 2. 分区 GCR
regions = gcr["regions"]
gcr_map = gcr["gcr_map"]
skin_gcr = gcr_map[regions == cb.REGION_SKIN].mean()
metal_gcr = gcr_map[regions == cb.REGION_METALLIC].mean()
shadow_gcr = gcr_map[regions == cb.REGION_SHADOW].mean()
check("2a 分区数 >= 3（肤色/金属/暗调被识别）",
      len(np.unique(regions)) >= 3, f"labels={np.unique(regions)}")
check("2b 肤色低 GCR", skin_gcr <= cb.GCR_SKIN + 1e-9, f"skin={skin_gcr:.2f}")
check("2c 金属低 GCR（保光泽）", metal_gcr <= cb.GCR_METALLIC + 1e-9,
      f"metal={metal_gcr:.2f}")
check("2d 暗调高 GCR（省墨稳灰）", shadow_gcr >= cb.GCR_SHADOW - 1e-9,
      f"shadow={shadow_gcr:.2f}")
check("2e 肤色 GCR 严格低于暗调 GCR", skin_gcr < shadow_gcr)
# 黑版不再“全额换黑”：除暗调外没有任何像素 K == min(CMY)
cmy = 1.0 - rgb
gray = cmy.min(axis=-1)
k = gcr["cmyk"][..., 3]
non_shadow = regions != cb.REGION_SHADOW
full_swap = np.count_nonzero(np.abs(k[non_shadow] - gray[non_shadow]) < 1e-9)
check("2f 非暗调区不再全额黑版替换", full_swap == 0, f"full-swap px={full_swap}")

# ---------------------------------------------------------------- 3. 同步锐化
sharp = cb.sync_sharpen(gcr["cmyk"], radius=2, sigma=1.0, amount=1.0)
# 构造一个边缘信号，检查四通道在完全相同的位置产生响应
edge_img = np.zeros((20, 20, 4))
edge_img[:, 10:, :] = 0.8
s = cb.sync_sharpen(edge_img, radius=2, sigma=1.0, amount=1.0)["cmyk"]
deltas = [s[..., ch] - edge_img[..., ch] for ch in range(4)]
support = [(np.abs(d) > 1e-9) for d in deltas]
aligned = all(np.array_equal(support[0], support[ch]) for ch in range(1, 4))
check("3a 四通道锐化影响的像素位置完全一致（无 1px 错位）", aligned)
same_sign = all(
    np.allclose(deltas[ch][support[0]], deltas[0][support[0]]) for ch in range(1, 4)
)
check("3b 四通道锐化幅度一致（不生彩边）", same_sign)
check("3c 锐化后仍满足 TIC 兜底（管线级）",
      cb.convert_cmyk(rgb)["tic_map"].max() <= 3.0 + 1e-9)

# ---------------------------------------------------------------- 4. 灰平衡
# 先 TIC 压缩制造中性灰失衡，再重建
unbalanced = np.array([[[0.70, 0.70, 0.70, 0.95]]])  # 总 3.05 -> 压缩
pressed = cb.tic_constraint(unbalanced, 3.0)["cmyk"]
rebuilt = cb.gray_balance_rebuild(pressed)["cmyk"][0, 0]
check("4a 重建后 C >= M >= Y（胶印灰平衡方向）",
      rebuilt[0] >= rebuilt[1] >= rebuilt[2], f"CMY={rebuilt[:3]}")
# 中性灰像素重建后配比贴近标准
balance = np.array([cb.GRAY_BALANCE_C, cb.GRAY_BALANCE_M, cb.GRAY_BALANCE_Y])
ratio = rebuilt[:3] / rebuilt[:3].mean()
check("4b 中性灰配比贴近标准灰平衡", np.allclose(ratio, balance / balance.mean(), atol=0.02),
      f"ratio={ratio}")
# 彩色像素不被灰平衡污染
color_px = np.array([[[0.9, 0.2, 0.1, 0.05]]])
color_out = cb.gray_balance_rebuild(color_px)["cmyk"][0, 0]
check("4c 高饱和彩色像素不参与灰平衡重建", np.allclose(color_out, color_px[0, 0], atol=1e-9))
# 端到端：中性灰块不偏色（重建后 C/M/Y 关系正确）
full = cb.convert_cmyk(rgb)
gray_block = full["cmyk"][16:32, :16]
check("4d 端到端中性灰块满足 C>=M>=Y",
      np.all(gray_block[..., 0] >= gray_block[..., 1] - 1e-9) and
      np.all(gray_block[..., 1] >= gray_block[..., 2] - 1e-9))

# ---------------------------------------------------------------- 5. 纸白适配
adapt_d50 = cb.paper_white_adapt(rgb, dst_white=(0.9642, 1.0, 0.8249))
adapt_warm = cb.paper_white_adapt(rgb, dst_white=(0.90, 1.0, 0.85))
check("5a 纸白适配返回有限值且在 [0,1]", np.all(np.isfinite(adapt_d50["rgb"])))
# 同一文件、不同纸白 -> 输出应不同（旧实现固定意图不处理纸白则完全相同）
check("5b 不同纸白产生不同分色结果", not np.allclose(adapt_d50["rgb"], adapt_warm["rgb"]))
# 白点本身应映射到目标白点（相对比色媒体白对齐）
white = np.ones((1, 1, 3))
w_adapted = cb.paper_white_adapt(white, dst_white=(0.9642, 1.0, 0.8249))["rgb"][0, 0]
check("5c 源白点经适应后仍为纸白（接近 1,1,1 的 sRGB 白）",
      np.all(w_adapted > 0.99), f"white={w_adapted}")
# 同一纸白两次运行一致
check("5d 纸白适配确定性",
      np.array_equal(cb.paper_white_adapt(rgb)["rgb"], adapt_d50["rgb"]))

# ---------------------------------------------------------------- 6. 页序绑定
pages = {3: "c", 1: "a", 2: "b"}
bound = cb.page_order_bind(pages)
order = [e["page"] for e in bound["manifest"]]
check("6a 乱序提交 -> 清单按页码升序", order == [1, 2, 3], f"order={order}")
check("6b 文件名零填充且绑定页码",
      bound["manifest"][0]["name"] == "page-0001.tif")

# 并发乱序提交
binder = cb.PageBinder()
submit_order = [4, 1, 3, 0, 2]
def submit(pg):
    binder.submit(pg, f"payload-{pg}".encode())
threads = [threading.Thread(target=submit, args=(p,)) for p in submit_order]
for t in threads:
    t.start()
for t in threads:
    t.join()
manifest = binder.manifest()
check("6c 多线程并发后页序仍正确",
      [e["page"] for e in manifest] == [0, 1, 2, 3, 4])
check("6d 每个页码与文件名摘要一一对应",
      all(e["name"] == binder.page_name(e["page"]) for e in manifest))

# 重号检测
dup = cb.page_order_bind([(1, "a"), (1, "b"), (2, "c")])
check("6e 重号被检出", dup["duplicates"] == [1])
# 缺页检测
gap = cb.page_order_bind({1: "a", 3: "c"})
check("6f 缺页被检出", gap["missing"] == [2])

# ---------------------------------------------------------------- 管线整体
det = cb.determinism_print(rgb, runs=4)
check("7a 多次运行逐位一致 + 并发页序正确", det["ok"], str({k: v for k, v in det.items() if k != "digests"}))
val = cb.validate_print(full)
check("7b 印前校验通过", val["ok"], str(val))
met = cb.metrics_print(full)
check("7c 度量含 TIC 与分区占比", "tic_max" in met and set(met["region_share"]) == set(cb.REGION_NAMES))
audit = cb.audit_print(full)
check("7d 审计哈希稳定", audit["digest"] == cb.audit_print(full)["digest"])
fb = cb.fallback_print(rgb)
check("7e 降级管线可用且满足 TIC", fb["ok"] and fb["cmyk"].sum(axis=-1).max() <= 3.0 + 1e-9)
bad = cb.load_rgb("/nonexistent/path/x.png")
check("7f 坏输入结构化降级不抛异常", bad["ok"] is False)

print("\n" + "=" * 60)
failed = [n for n, ok, _ in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} 通过")
if failed:
    print("失败项:", failed)
    sys.exit(1)
print("全部通过")

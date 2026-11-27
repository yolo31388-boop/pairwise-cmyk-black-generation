"""六类印刷缺陷修复的回归测试。

对应需求：总墨量约束、分区 GCR、通道同步锐化、灰平衡重建、纸白适配、页序绑定。
"""
import threading

import numpy as np
import pytest

import imaging.cmyk_black as cb


@pytest.fixture(scope="module")
def sample_rgb():
    rng = np.random.default_rng(7)
    rgb = np.zeros((48, 48, 3), dtype=np.float64)
    rgb[:16, :16] = (0.85, 0.55, 0.42)       # 肤色
    rgb[:16, 16:32] = (0.80, 0.78, 0.72)     # 金属
    rgb[:16, 32:] = (0.08, 0.07, 0.09)       # 暗调
    rgb[16:32, :16] = (0.5, 0.5, 0.5)        # 中性灰
    rgb[16:32, 16:32] = (0.9, 0.1, 0.1)      # 饱和红
    rgb[16:32, 32:] = rng.random((16, 16, 3))
    rgb[32:, :] = rng.random((16, 48, 3))
    return rgb


# ---------------------------------------------------------------- 1. TIC
def test_tic_never_exceeds_limit(sample_rgb):
    out = cb.tic_constraint(cb.region_gcr(sample_rgb)["cmyk"], 3.0)
    assert out["cmyk"].sum(axis=-1).max() <= 3.0 + 1e-9


def test_tic_scales_cmy_proportionally_keeps_hue():
    px = np.array([[[0.9, 0.9, 0.9, 0.9]]])
    out = cb.tic_constraint(px, 3.0)["cmyk"][0, 0]
    scale = out[:3] / px[0, 0, :3]
    assert np.allclose(scale, scale[0])          # CMY 同一缩放因子
    assert out[3] == pytest.approx(0.9)          # K 不动


def test_tic_clamps_k_alone_when_over_limit():
    out = cb.tic_constraint(np.array([[[0.0, 0.0, 0.0, 3.5]]]), 3.0)["cmyk"][0, 0]
    assert out[3] == pytest.approx(3.0)


# ---------------------------------------------------------------- 2. 分区 GCR
def test_region_gcr_levels(sample_rgb):
    gcr = cb.region_gcr(sample_rgb)
    regions, gmap = gcr["regions"], gcr["gcr_map"]
    assert set(np.unique(regions)) >= {cb.REGION_SKIN, cb.REGION_METALLIC,
                                       cb.REGION_SHADOW}
    assert gmap[regions == cb.REGION_SKIN].mean() <= cb.GCR_SKIN + 1e-9
    assert gmap[regions == cb.REGION_METALLIC].mean() <= cb.GCR_METALLIC + 1e-9
    assert gmap[regions == cb.REGION_SHADOW].mean() >= cb.GCR_SHADOW - 1e-9


def test_black_plate_never_full_swap_outside_shadow(sample_rgb):
    gcr = cb.region_gcr(sample_rgb)
    gray = (1.0 - sample_rgb).min(axis=-1)
    k = gcr["cmyk"][..., 3]
    non_shadow = gcr["regions"] != cb.REGION_SHADOW
    assert not np.any(np.abs(k[non_shadow] - gray[non_shadow]) < 1e-9)


# ---------------------------------------------------------------- 3. 同步锐化
def test_sharpen_aligned_across_channels():
    edge = np.zeros((20, 20, 4))
    edge[:, 10:, :] = 0.8
    out = cb.sync_sharpen(edge, radius=2, sigma=1.0, amount=1.0)["cmyk"]
    supports = [np.abs(out[..., ch] - edge[..., ch]) > 1e-9 for ch in range(4)]
    assert all(np.array_equal(supports[0], s) for s in supports[1:])
    # 同位置四通道增量一致 -> 不生彩边
    base = (out[..., 0] - edge[..., 0])[supports[0]]
    for ch in range(1, 4):
        assert np.allclose((out[..., ch] - edge[..., ch])[supports[0]], base)


# ---------------------------------------------------------------- 4. 灰平衡
def test_gray_balance_rebuild_direction():
    pressed = cb.tic_constraint(np.array([[[0.7, 0.7, 0.7, 0.95]]]), 3.0)["cmyk"]
    c, m, y = cb.gray_balance_rebuild(pressed)["cmyk"][0, 0, :3]
    assert c >= m >= y


def test_gray_balance_skips_saturated_colors():
    px = np.array([[[0.9, 0.2, 0.1, 0.05]]])
    out = cb.gray_balance_rebuild(px)["cmyk"]
    assert np.allclose(out, px, atol=1e-9)


def test_pipeline_neutral_gray_not_cast(sample_rgb):
    cmyk = cb.convert_cmyk(sample_rgb)["cmyk"][16:32, :16]
    assert np.all(cmyk[..., 0] >= cmyk[..., 1] - 1e-9)
    assert np.all(cmyk[..., 1] >= cmyk[..., 2] - 1e-9)


# ---------------------------------------------------------------- 5. 纸白适配
def test_paper_white_media_white_mapping():
    white = np.ones((1, 1, 3))
    out = cb.paper_white_adapt(white, dst_white=tuple(cb._D50_XYZ))["rgb"][0, 0]
    assert np.all(out > 0.999)


def test_different_paper_produces_different_result(sample_rgb):
    a = cb.paper_white_adapt(sample_rgb, dst_white=tuple(cb._D50_XYZ))["rgb"]
    b = cb.paper_white_adapt(sample_rgb, dst_white=(0.90, 1.0, 0.85))["rgb"]
    assert not np.allclose(a, b)


def test_paper_white_deterministic(sample_rgb):
    a = cb.paper_white_adapt(sample_rgb)["rgb"]
    b = cb.paper_white_adapt(sample_rgb)["rgb"]
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- 6. 页序绑定
def test_page_order_binding_sorts_by_page():
    out = cb.page_order_bind({3: "c", 1: "a", 2: "b"})
    assert [e["page"] for e in out["manifest"]] == [1, 2, 3]
    assert out["manifest"][0]["name"] == "page-0001.tif"


def test_page_order_binding_concurrent():
    binder = cb.PageBinder()
    order = [4, 1, 3, 0, 2]
    threads = [threading.Thread(target=binder.submit, args=(p, f"p{p}")) for p in order]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [e["page"] for e in binder.manifest()] == [0, 1, 2, 3, 4]


def test_page_order_detects_duplicate_and_gap():
    assert cb.page_order_bind([(1, "a"), (1, "b"), (2, "c")])["duplicates"] == [1]
    assert cb.page_order_bind({1: "a", 3: "c"})["missing"] == [2]


# ---------------------------------------------------------------- 管线整体
def test_full_pipeline_validates(sample_rgb):
    res = cb.convert_cmyk(sample_rgb)
    assert cb.validate_print(res)["ok"]
    assert res["tic_map"].max() <= 3.0 + 1e-9


def test_determinism_across_runs_and_threads(sample_rgb):
    assert cb.determinism_print(sample_rgb, runs=4)["ok"]


def test_bad_input_never_raises():
    for fn in (cb.load_rgb, cb.convert_cmyk, cb.fallback_print,
               cb.proof_preview, cb.validate_print):
        out = fn("/no/such/file.png")
        assert out is not None and out["ok"] is False

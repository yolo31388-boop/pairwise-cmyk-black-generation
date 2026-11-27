# pairwise-cmyk-black-generation

## 领域

游戏、图形与多媒体

## 主题

CMYK 分色黑版生成

## 修复的六类印刷缺陷

| # | 旧实现问题 | 修复方案 | 入口 |
|---|-----------|---------|------|
| 1 | `min(CMY)` 全额换黑，深色区总墨量超 300% 糊版 | 总墨量约束：超限时 **K 不动、CMY 同一比例因子压缩**，保持色相与灰平衡；K 单通道超限才压 K | `tic_constraint` |
| 2 | 全局单一 GCR 比例，肤色发灰、金属失光 | **分区 GCR**：像素分为肤色(0.25)/金属(0.30)/暗调(0.90)/普通(0.60)，黑版只吸收对应比例的中性成分 | `region_gcr` / `classify_regions` |
| 3 | 各通道独立锐化，黑版边缘早 1px 起彩边 | **同步 USM**：同一高斯核、同一反射边界、同一亮度边缘权重作用于 C/M/Y/K，四通道像素偏移完全一致 | `sync_sharpen` |
| 4 | 逐通道裁剪破坏灰平衡，中性灰偏色 | TIC 压缩后对近中性像素按 **C:M:Y = 1.00:0.90:0.82** 重建配比，彩色像素不参与 | `gray_balance_rebuild` |
| 5 | ICC 意图固定、不处理纸白，跨纸张偏色 | **Bradford 色适应 + 相对比色媒体白对齐**（线性 RGB 域归一化，纸白精确映射为纯白），支持每文件指定源/目标白点与意图 | `paper_white_adapt` |
| 6 | 多页并发时页码与输出文件错位 | **页序绑定**：页码在提交时冻结进零填充文件名与 SHA-256 摘要，清单按页码升序输出，带锁原子写，重号/缺页可检 | `page_order_bind` / `PageBinder` |

## 分色管线顺序

```
RGB
 └─ paper_white_adapt   纸白适配（Bradford CAT，媒体白对齐）
     └─ region_gcr      分区 GCR 产生初始 C/M/Y/K
         └─ tic_constraint      总墨量等比压缩（保灰平衡）
             └─ gray_balance_rebuild  中性灰配比重建
                 └─ sync_sharpen      四通道同步锐化
                     └─ tic_constraint 锐化振铃兜底，保证出口仍 ≤ TIC
```

辅助入口：`load_rgb`、`proof_preview`、`present_print`（TIFF+manifest 落盘）、
`validate_print`、`metrics_print`、`audit_print`、`fallback_print`、
`determinism_print`（多次运行逐位一致 + 并发页序校验）。

所有不可识别输入走结构化降级（`{"ok": False, ...}`），不抛异常。

## 运行测试

```bash
python -m pytest -q
```

六类缺陷的专项验证脚本：

```bash
python -P scripts/verify_fixes.py
```

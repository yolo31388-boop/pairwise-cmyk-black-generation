# pairwise-cmyk-black-generation

## 领域

游戏、图形与多媒体

## 主题

CMYK 分色黑版生成

## 说明

总墨量约束、分区 GCR、通道同步锐化。

初始状态下 `imaging/cmyk_black.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

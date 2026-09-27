# 09 宏观 PIT 上下文

安装 financial-data extra 后：
```python
from quant_regime.macro import macro_context
context = macro_context(observations, "2024-02-01T13:30:00Z", required_series=["CPI"], max_age_days=90)
```
输入使用 QDK financial.macro.COLUMNS，保留 observation_date、vintage_date、released_at、available_at 和单位/证据。未来发布时间不可见，修订仅从新获知时点生效。返回 values、unavailable、complete；陈旧或缺失不填零。

这是独立上下文 API，不改变旧技术指标 regime 分类器，也未把宏观序列自动解释为仓位信号。调用者应显式制定 incomplete 时的仓位政策并另做样本外验证。测试见 tests/test_macro_context.py。

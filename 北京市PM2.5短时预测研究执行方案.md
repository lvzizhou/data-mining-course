# 北京市国控监测站 PM2.5 短时预测研究执行方案

> 对应课题：**基于时空图神经网络的北京市国控监测站 PM2.5 短时预测研究**  
> 文档定位：从空目录开始，逐步执行到成果冻结的完整操作手册  
> 最终成果：可复现实验工程、课程论文、成果展示平台  
> 当前范围：详细规定数据与模型实施方式；论文写作和展示平台的单独建设方案后续另行编制

### 本文档的使用方式

- 第 1 至 24 节解释各项技术的原理、规则和推荐实现；
- 第 25 节是实际执行时使用的逐步操作手册；
- 每一步均包含前置输入、详细操作、输出文件、量化验收和失败处理；
- 必须按步骤编号顺序推进，当前一步未通过验收时不得直接进入下一步；
- 若前文与第 25 节存在执行口径差异，以第 25 节为准；
- 论文写作和展示平台开发将在后续文档中单独设计，本方案只确保实验阶段产出它们必需且可复用的数据、模型、图表和元数据。

---

## 1. 执行目标

本项目要完成的不是一次孤立的模型训练，而是一条可以重复运行、能够支撑论文结论，并可直接向展示平台提供数据的完整流水线：

```text
站点元数据
    │
    ├── 空气质量小时数据
    ├── 气象再分析小时数据
    │
    ▼
数据审计 → 清洗对齐 → 特征构造 → 时间窗口 → 图结构
    │                                      │
    ├─────────────── 基线模型 ──────────────┤
    └────────────── 时空图模型 ─────────────┘
                                             │
                                             ▼
                         统一评估 → 消融实验 → 案例分析
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
             论文图表与实验记录                              展示平台数据产品
```

项目完成时必须能够回答：

1. 北京各监测站之间是否存在可利用的空间依赖？
2. 时空图模型是否优于不使用图的时间模型和传统机器学习模型？
3. 污染物与历史气象特征分别贡献了多少预测能力？
4. 固定距离图、历史相关图和风向动态图各自适用于什么场景？
5. 模型在哪些站点、季节、污染等级和预测时长上有效或失效？

---

## 2. 固定研究设置

以下设置作为第一版实验协议。只有在验证集结果或数据审计明确表明不合理时才能修改，且必须记录修改原因。

### 2.1 研究对象

- 空间范围：北京市；
- 节点：研究期内具有稳定记录和可靠坐标的国控空气质量监测站；
- 目标变量：各站点 PM2.5 小时浓度；
- 时间粒度：1 小时；
- 输入窗口：过去 24 小时；
- 预测目标：未来第 1、3、6 小时 PM2.5；
- 主要模型：STGCN；
- 动态图扩展：风向加权的动态图时空模型；
- 主要指标：MAE、RMSE；
- 辅助指标：\(R^2\)、SMAPE、高污染时段 MAE。

### 2.2 建议研究期

第一轮尝试下载：

```text
2023-01-01 至 2025-12-31
```

正式研究期由数据审计决定。选择规则依次为：

1. 尽可能保留连续三个完整年度；
2. 所选站点 PM2.5 有效率原则上不低于 90%；
3. 其他主要污染物和气象特征有效率原则上不低于 85%；
4. 若三年无法满足要求，则选择最近的连续两年；
5. 不允许把多个不连续时间段拼成一条连续序列而不做标记。

### 2.3 关于“短时预测”的操作定义

以某时刻 \(t\) 为预测起点：

\[
\mathbf{X}_{t-23:t}
\longrightarrow
\left(
\mathbf{Y}_{t+1},
\mathbf{Y}_{t+3},
\mathbf{Y}_{t+6}
\right)
\]

其中：

- \(\mathbf{X}_{t-23:t}\) 是所有站点过去 24 小时的可用特征；
- \(\mathbf{Y}_{t+h}\) 是所有站点在未来第 \(h\) 小时的 PM2.5；
- 模型采用直接多步预测，一次输出三个时距，不进行递归滚动；
- 预测时不得使用 \(t\) 时刻之后的真实污染或真实气象观测。

---

## 3. 工程环境

### 3.1 推荐软件

- Python 3.11；
- PyTorch 2.x；
- pandas、NumPy、PyArrow；
- scikit-learn；
- LightGBM；
- requests、httpx 或 aiohttp；
- tenacity；
- PyYAML；
- matplotlib、seaborn、Plotly、Folium；
- MLflow 或本地 JSON/CSV 实验登记；
- pytest；
- ruff；
- JupyterLab，仅用于探索，不承载正式数据流水线。

图神经网络可选两种实现方式：

1. 使用 PyTorch 编写基础图卷积，依赖少、便于解释；
2. 使用 PyTorch Geometric，但必须锁定版本并记录数据张量格式。

第一版 STGCN 建议直接用 PyTorch 实现图卷积和时间卷积，避免为一个小图引入过重框架。

### 3.2 建议目录

```text
DataMining/
├─ README.md
├─ pyproject.toml
├─ configs/
│  ├─ data.yaml
│  ├─ features.yaml
│  ├─ graph.yaml
│  ├─ baseline.yaml
│  └─ stgcn.yaml
├─ data/
│  ├─ raw/
│  │  ├─ air_quality/
│  │  ├─ weather/
│  │  └─ stations/
│  ├─ interim/
│  ├─ processed/
│  └─ manifests/
├─ src/
│  ├─ data/
│  │  ├─ download_air_quality.py
│  │  ├─ download_weather.py
│  │  ├─ validate_raw.py
│  │  ├─ clean_air_quality.py
│  │  ├─ clean_weather.py
│  │  └─ build_dataset.py
│  ├─ features/
│  │  ├─ time_features.py
│  │  ├─ graph_distance.py
│  │  ├─ graph_correlation.py
│  │  └─ graph_wind.py
│  ├─ models/
│  │  ├─ persistence.py
│  │  ├─ historical_mean.py
│  │  ├─ lightgbm_model.py
│  │  ├─ gru.py
│  │  ├─ stgcn.py
│  │  └─ wind_gcn_gru.py
│  ├─ training/
│  │  ├─ dataset.py
│  │  ├─ losses.py
│  │  ├─ train.py
│  │  └─ predict.py
│  ├─ evaluation/
│  │  ├─ metrics.py
│  │  ├─ stratified.py
│  │  ├─ significance.py
│  │  └─ cases.py
│  └─ visualization/
│     ├─ maps.py
│     ├─ curves.py
│     └─ export_platform.py
├─ tests/
├─ notebooks/
│  └─ exploration/
├─ outputs/
│  ├─ runs/
│  ├─ checkpoints/
│  ├─ predictions/
│  ├─ metrics/
│  ├─ figures/
│  ├─ cases/
│  └─ platform/
├─ reports/
└─ docs/
```

### 3.3 环境锁定

项目开始时生成并提交：

- `pyproject.toml` 或 `requirements.txt`；
- 锁文件；
- Python 版本；
- CUDA、显卡和操作系统信息；
- `pip freeze` 快照；
- 全局随机种子配置。

建议固定：

```yaml
seed: 42
deterministic: true
device: auto
float_precision: float32
timezone: Asia/Shanghai
```

---

## 4. 数据获取方案

## 4.1 空气质量数据

### 4.1.1 数据来源

原始观测来源为中国环境监测总站全国城市空气质量实时发布平台，项目采用 QuotSoft 整理的历史归档获取逐站小时数据。

- 数据说明页：<https://quotsoft.net/air/>
- 单日全国站点文件：

```text
https://quotsoft.net/air/data/china_sites_YYYYMMDD.csv
```

例如：

```text
https://quotsoft.net/air/data/china_sites_20250101.csv
```

该来源是第三方历史归档，不应在论文中描述为官方开放 API。论文的数据来源应表述为：

> 原始空气质量观测来自中国环境监测总站公开发布平台，历史文件由 QuotSoft 整理归档。

### 4.1.2 原始格式

单日文件通常是宽表：

```text
date,hour,type,1001A,1002A,1003A,...
20250101,0,PM2.5,43,51,38,...
20250101,0,PM10,71,80,63,...
...
```

主要 `type`：

- `PM2.5`：PM2.5 1 小时均值，\(\mu g/m^3\)；
- `PM10`：PM10 1 小时均值，\(\mu g/m^3\)；
- `SO2`：SO2 1 小时均值，\(\mu g/m^3\)；
- `NO2`：NO2 1 小时均值，\(\mu g/m^3\)；
- `O3`：O3 1 小时均值，\(\mu g/m^3\)；
- `CO`：CO 1 小时均值，\(mg/m^3\)；
- `AQI`：空气质量指数；
- 带 `_24h` 或 `_8h` 的派生统计量不作为模型输入。

第一版只保留：

```python
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]
```

原因是 24 小时或 8 小时滚动指标由历史观测重复计算而来，容易造成冗余，也可能把数据发布规则引入模型。

### 4.1.3 下载程序要求

下载器按日期生成 URL，必须具备：

- 连接超时和读取超时；
- 最多 3 次指数退避重试；
- 限速，例如每次请求间隔 0.2 至 0.5 秒；
- 已存在且校验通过的文件跳过；
- HTTP 状态码记录；
- 空响应、HTML 错误页和异常小文件识别；
- SHA-256 校验；
- 失败日期清单；
- 可从失败清单断点续传。

伪代码：

```python
for date in date_range(start_date, end_date):
    url = f"https://quotsoft.net/air/data/china_sites_{date:%Y%m%d}.csv"
    target = RAW_AIR_DIR / f"china_sites_{date:%Y%m%d}.csv"

    if target.exists() and validate_csv(target):
        continue

    response = get_with_retry(url, timeout=(10, 60))
    if response.status_code == 404:
        log_missing(date, url)
        continue

    response.raise_for_status()
    assert looks_like_csv(response.content)
    atomic_write(target, response.content)
    write_manifest(date, url, target, sha256(target))
```

禁止直接把响应写入最终文件。应先写入 `.part` 临时文件，校验后原子重命名，避免中断产生半文件。

### 4.1.4 下载后的原始校验

每个文件至少检查：

1. 存在 `date`、`hour`、`type`；
2. `date` 与文件名日期相同；
3. `hour` 可转换为整数且位于 0 至 23；
4. 至少包含一个目标站点列；
5. 至少包含 `PM2.5`；
6. 文件不是服务器返回的 HTML；
7. 编码可被 UTF-8 或 UTF-8-SIG 正确读取；
8. 同一 `(date, hour, type)` 不应出现多个冲突行。

若某日文件缺失，不立即使用插值补整天，而是登记为数据缺口，在构造滑动窗口时排除跨越该缺口的样本。

---

## 4.2 监测站元数据

### 4.2.1 来源

优先使用与空气质量归档同版本的站点列表：

- QuotSoft 数据说明及网盘中的“站点列表”；
- 核验来源：<https://docs.datastation.skyviewor.com/data-query/china-pollution/pollution-stations-list/>

候选字段：

```text
station_id,station_name,province,city,district,longitude,latitude,
valid_from,valid_to,source
```

北京市传统国控站代码通常包括 `1001A` 至 `1012A` 中的若干站点，但站点会新增、撤销、改名或迁址，不能直接假设整个研究期始终固定为 12 个站。

### 4.2.2 版本处理

站点元数据必须版本化。若同一站点代码在不同时期对应不同坐标：

1. 根据 `valid_from` 和 `valid_to` 建立有效期；
2. 若迁址距离较大，将迁址前后视为不同节点，或缩短研究期；
3. 不得用最新坐标解释所有历史记录；
4. 在数据审计报告中列出站点集合变化。

正式站点集合按以下步骤产生：

```text
北京市站点候选集
  ∩ 研究期内空气质量文件实际出现的站点
  ∩ 具有可靠经纬度的站点
  ∩ PM2.5 有效率满足阈值的站点
```

### 4.2.3 坐标核验

检查：

- 经度位于约 115°E 至 118°E；
- 纬度位于约 39°N 至 42°N；
- 地图上位置应落在北京或合理边界附近；
- 重名站点不得仅靠名称连接，必须使用 `station_id`；
- 坐标统一为 WGS84。

---

## 4.3 气象数据

### 4.3.1 数据来源

采用 Open-Meteo Historical Weather API：

- 官方文档：<https://open-meteo.com/en/docs/historical-weather-api>
- 接口：

```text
https://archive-api.open-meteo.com/v1/archive
```

底层为 ERA5/ERA5-Land 等再分析数据。它是网格化再分析产品，不是监测站现场实测气象数据，论文和展示平台必须明确标注。

### 4.3.2 请求参数

每个站点按经纬度请求：

```text
latitude
longitude
start_date
end_date
hourly
timezone=Asia/Shanghai
wind_speed_unit=ms
```

小时变量：

```text
temperature_2m
relative_humidity_2m
dew_point_2m
surface_pressure
precipitation
wind_speed_10m
wind_direction_10m
```

示例：

```text
https://archive-api.open-meteo.com/v1/archive
?latitude=39.982
&longitude=116.3968
&start_date=2025-01-01
&end_date=2025-01-31
&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,precipitation,wind_speed_10m,wind_direction_10m
&wind_speed_unit=ms
&timezone=Asia%2FShanghai
```

实际程序应使用请求参数字典，不要手工拼接转义后的长 URL。

### 4.3.3 请求切片

建议以“站点 × 年”为请求单位：

- 单个文件大小适中；
- 失败后只重试一个站点的一年；
- 易于增量更新；
- 便于记录网格坐标和海拔等响应元数据。

保存格式：

```text
data/raw/weather/{station_id}/{year}.json
```

同时保存原始 JSON 和标准化 Parquet。原始响应至少保留：

- 请求参数；
- 返回的经纬度；
- `utc_offset_seconds`；
- `timezone`；
- `hourly_units`；
- `hourly.time`；
- 所有请求变量。

### 4.3.4 气象数据的因果约束

本研究的主实验模拟“在当前时刻，根据已经观测到的信息预测未来 PM2.5”。因此：

- 输入只能包含 \(t\) 及之前的气象记录；
- 不得把历史数据中的 \(t+1\)、\(t+3\)、\(t+6\) 真实气象值直接作为输入；
- 动态风向图使用输入窗口内最后一个有效时刻的风向，或输入窗口内最近若干小时风向的加权平均；
- 若未来要研究“已知天气预报条件下的 PM2.5 预测”，必须作为独立实验，并使用历史天气预报产品，而不能用再分析真值代替天气预报。

这项约束是论文可信度的关键。

---

## 4.4 数据清单和可追溯性

每次下载生成 `manifest.csv`：

```text
source_type
source_url
request_params
local_path
date_start
date_end
downloaded_at
http_status
size_bytes
sha256
validation_status
error_message
```

处理后的数据集生成 `dataset_manifest.json`：

```json
{
  "dataset_version": "v1.0.0",
  "created_at": "ISO-8601 timestamp",
  "air_quality_source": "QuotSoft/CNEMC archive",
  "weather_source": "Open-Meteo Historical Weather API",
  "date_range": ["2023-01-01", "2025-12-31"],
  "stations": ["..."],
  "features": ["..."],
  "raw_manifest_sha256": "...",
  "preprocessing_config_sha256": "...",
  "split_definition_sha256": "..."
}
```

论文中的所有数字都必须能追溯到某个 `dataset_version`。

---

## 5. 数据清洗与对齐

## 5.1 空气质量宽表转长表

对每个单日文件：

1. 读取 `date`、`hour`、`type` 和最终候选站点列；
2. 只保留六种小时污染物；
3. 使用 `melt` 把站点列转为 `station_id`；
4. 将空字符串、`NA`、`—` 等转换为缺失；
5. 将数值列强制转换为浮点型；
6. 生成北京时间时间戳；
7. 输出长表。

目标结构：

```text
timestamp
station_id
pollutant
value
source_file
quality_flag
```

时间戳生成：

```python
timestamp = (
    pd.to_datetime(date.astype(str), format="%Y%m%d")
    + pd.to_timedelta(hour, unit="h")
).dt.tz_localize("Asia/Shanghai")
```

若原始说明表明 `hour=24` 代表次日 0 时，则必须先转换为下一日 0 时，并编写测试；不能直接把 24 当作非法值删除。正式实现前应对多个日期样本核验小时编码规则。

## 5.2 去重

主键：

```text
(timestamp, station_id, pollutant)
```

处理规则：

- 重复值完全相同：保留一条，记录重复数量；
- 一条缺失、一条非缺失：保留非缺失值；
- 多个非缺失值冲突：不任意平均，设为缺失并写入冲突日志；
- 同一主键跨文件出现：根据文件日期、来源优先级和下载时间核验。

## 5.3 单位和范围

统一单位：

- PM2.5、PM10、SO2、NO2、O3：\(\mu g/m^3\)；
- CO：\(mg/m^3\)；
- 温度：°C；
- 湿度：%；
- 气压：hPa；
- 降水：mm；
- 风速：m/s；
- 风向：角度。

基础合法性规则：

- 所有污染物浓度小于 0：设为缺失；
- 相对湿度不在 0 至 100：设为缺失；
- 风速小于 0：设为缺失；
- 风向不在 0 至 360：先检查是否可取模修复，否则设为缺失；
- 时间、站点代码或坐标非法：整条记录进入隔离区。

高浓度不自动视为错误。PM2.5 污染事件正是重要研究对象，不能简单按 99% 分位数删除。

对于极端正值，采用以下流程：

1. 标记训练集内超出中位数 \(\pm 8 \times MAD\) 的点；
2. 检查相邻小时、邻近站点和其他污染物；
3. 若具有时间连续性或空间一致性，保留为真实事件；
4. 若是孤立尖峰且伴随数据格式异常，设为缺失；
5. 所有人工判定写入 `quality_events.csv`。

测试集极端值不得依据测试误差删除。

## 5.4 污染物透视

把长表透视为：

```text
timestamp,station_id,PM2.5,PM10,SO2,NO2,O3,CO
```

使用明确的站点—小时完整索引重新索引，确保缺失小时真实存在，而不是被表格结构隐藏。

## 5.5 气象清洗

将 Open-Meteo 数组展开为：

```text
timestamp
station_id
temperature_2m
relative_humidity_2m
dew_point_2m
surface_pressure
precipitation
wind_speed_10m
wind_direction_10m
```

检查：

- `hourly.time` 与每个变量数组长度一致；
- 时区为 `Asia/Shanghai`；
- 每个自然日通常有 24 条记录；
- API 返回坐标与请求坐标差异被记录；
- 站点可能映射到同一网格，不把重复网格误判为下载错误。

## 5.6 多源时间对齐

使用以下键左连接：

```text
(timestamp, station_id)
```

以空气质量完整时间网格为主，连接气象数据和站点元数据。

连接后必须检查：

- 行数是否等于 `小时数 × 站点数`；
- 主键是否唯一；
- 气象匹配率；
- 经纬度是否稳定；
- 连接前后 PM2.5 非缺失数量是否一致。

## 5.7 缺失值处理

必须同时保存：

1. 原始观测值；
2. 缺失掩码；
3. 插补后的模型输入值。

掩码：

```text
mask_feature = 1：原始有观测
mask_feature = 0：原始缺失
```

因果插补规则：

1. 按站点和变量排序；
2. 对连续不超过 3 小时的缺失执行前向填充；
3. 更长缺失使用训练集计算的“站点 × 月份 × 小时”中位数；
4. 若分组样本不足，退化为“站点 × 小时”训练中位数；
5. 再不足时使用该变量训练集全局中位数；
6. 目标标签 PM2.5 不插补，标签缺失的损失位置直接屏蔽；
7. 测试期插补只能调用训练期统计量和当前时刻之前的数据。

不用双向线性插值作为默认方案，因为它可能把预测起点之后的信息带入输入。

## 5.8 风向编码

Open-Meteo 的风向通常表示风从哪个方向吹来。神经网络输入使用：

\[
\text{wind\_sin}=\sin(\theta\pi/180)
\]

\[
\text{wind\_cos}=\cos(\theta\pi/180)
\]

原始角度仍保留，用于构造动态图。

## 5.9 时间特征

构造：

\[
\sin(2\pi \cdot hour/24),\quad
\cos(2\pi \cdot hour/24)
\]

\[
\sin(2\pi \cdot weekday/7),\quad
\cos(2\pi \cdot weekday/7)
\]

\[
\sin(2\pi \cdot dayofyear/365.25),\quad
\cos(2\pi \cdot dayofyear/365.25)
\]

可增加：

- 是否周末；
- 月份；
- 季节。

节假日特征只有在能获得可靠、逐年维护的节假日表时才加入。

## 5.10 清洗后数据结构

推荐保存两个数据产品。

长表 `observations.parquet`：

```text
timestamp
station_id
PM2.5
PM10
SO2
NO2
O3
CO
temperature_2m
relative_humidity_2m
dew_point_2m
surface_pressure
precipitation
wind_speed_10m
wind_direction_10m
各变量缺失掩码
各时间特征
split
```

模型张量 `dataset.npz` 或分片 Parquet：

```text
X: [time, node, feature]
Y: [time, node]
M: [time, node, target_mask]
timestamps: [time]
station_ids: [node]
feature_names: [feature]
```

长表用于审计、论文和平台，张量用于训练。两者必须共享相同的站点顺序和时间索引。

---

## 6. 数据质量门槛

进入建模前必须生成数据审计报告并通过以下检查。

### 6.1 必查统计

- 每个原始日期文件是否存在；
- 每个站点每个变量的有效率；
- 连续缺失长度分布；
- 每月记录数量；
- 每小时记录数量；
- 各污染物基本分布；
- 站点间 PM2.5 相关性；
- 气象站点网格重复情况；
- 站点元数据版本变化；
- 目标时间段是否存在整日或整站缺失。

### 6.2 站点保留规则

第一版规则：

- PM2.5 总有效率不低于 90%；
- 单次连续缺失原则上不超过 72 小时；
- 有可靠坐标；
- 在训练、验证和测试期均有观测；
- 若移除一个站点，应记录具体原因。

如果保留下来的站点少于 8 个，应停止建模并重新检查研究期或数据来源，因为节点过少会削弱图模型研究意义。

### 6.3 自动测试

至少编写以下测试：

```text
test_timestamp_is_hourly_and_sorted
test_station_time_key_is_unique
test_required_features_exist
test_no_future_values_in_imputer
test_scaler_fit_on_train_only
test_correlation_graph_fit_on_train_only
test_window_does_not_cross_split
test_station_order_is_consistent
test_adjacency_shape_and_finiteness
test_prediction_shape
```

---

## 7. 数据划分和样本构造

## 7.1 时间划分

默认按完整时间顺序：

- 训练集：前 70%；
- 验证集：中间 10%；
- 测试集：最后 20%。

划分边界向自然日 0 时对齐，并保存到：

```yaml
train_end: YYYY-MM-DDTHH:00:00+08:00
val_end: YYYY-MM-DDTHH:00:00+08:00
test_end: YYYY-MM-DDTHH:00:00+08:00
```

禁止随机划分小时样本。

## 7.2 窗口构造

对预测起点 \(t\)：

```text
输入索引：t-23 ... t
标签索引：t+1、t+3、t+6
```

样本张量：

```text
x: [24, N, F]
y: [3, N]
y_mask: [3, N]
timestamp_origin: scalar
```

样本保留条件：

- 输入时间轴完整连续；
- 输入 PM2.5 总体观测率达到设定阈值，例如 80%；
- 至少存在一个有效目标标签；
- 窗口不跨越数据集分割边界；
- 不跨越整日缺档。

## 7.3 标准化

仅在训练集上拟合：

\[
z=(x-\mu_{\text{train}})/\sigma_{\text{train}}
\]

建议按“特征全局”标准化，即每个特征在全部训练时间和站点上计算一组均值和标准差。这样保留站点间相对水平，便于图消息传递。

目标 PM2.5 使用单独的训练集均值和标准差。指标计算前必须逆变换到 \(\mu g/m^3\)。

对降水等偏态变量可先做：

\[
x'=\log(1+x)
\]

但必须在配置中记录，并保持训练、评估、平台导出一致。

---

## 8. 图结构构建

## 8.1 图的基本定义

\[
\mathcal{G}=(\mathcal{V},\mathcal{E},A)
\]

- \(\mathcal{V}\)：监测站；
- \(\mathcal{E}\)：站点间连接；
- \(A\in\mathbb{R}^{N\times N}\)：邻接矩阵；
- 节点顺序由 `station_ids.json` 唯一确定。

## 8.2 地理距离图

### 8.2.1 距离计算

使用 Haversine 距离：

\[
d_{ij}=2R\arcsin\sqrt{
\sin^2\left(\frac{\phi_j-\phi_i}{2}\right)+
\cos\phi_i\cos\phi_j
\sin^2\left(\frac{\lambda_j-\lambda_i}{2}\right)}
\]

其中 \(R=6371\) km。

### 8.2.2 连边

默认每个节点连接最近的 3 个其他节点：

```yaml
graph_type: distance_knn
k: 3
symmetrize: max
self_loop: true
```

边权：

\[
A^{dist}_{ij}=
\begin{cases}
\exp(-d_{ij}^{2}/\sigma^{2}), & j\in kNN(i)\\
0, & \text{otherwise}
\end{cases}
\]

\(\sigma\) 取所有保留边距离的中位数。

随后：

1. 对称化；
2. 加自环；
3. 归一化。

归一化：

\[
\hat{A}=\tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}
\]

## 8.3 历史相关图

只使用训练集 PM2.5：

1. 按站点计算“月份 × 小时”训练均值；
2. 从原序列减去该季节性均值；
3. 对残差序列计算 Spearman 相关；
4. 每个节点保留最高的 3 个正相关邻居；
5. 负相关边第一版不保留；
6. 加自环并归一化。

去除共同季节性后再计算相关性，可以减少所有站点仅因日周期或季节变化而呈现虚高相关的问题。

## 8.4 风向动态图

### 8.4.1 方向约定

气象风向 \(\theta_i^t\) 表示风从哪个方向吹来，污染传播方向为：

\[
\psi_i^t=(\theta_i^t+180^\circ)\bmod 360^\circ
\]

计算站点 \(i\) 指向站点 \(j\) 的地理方位角 \(\beta_{ij}\)。

方向一致性：

\[
c_{ij}^{t}=
\max\left(0,\cos(\beta_{ij}-\psi_i^t)\right)
\]

动态边权：

\[
A_{ij}^{wind,t}
=A_{ij}^{dist}
\cdot(c_{ij}^{t})^p
\cdot g(v_i^t)
\]

其中：

- \(p\) 默认取 2，使偏离风向的边快速衰减；
- \(v_i^t\) 是站点 \(i\) 的风速；
- \(g(v)=\min(v/v_0,1)\)；
- \(v_0\) 默认取 5 m/s，在验证集上比较 3、5、8。

当风速很低时，风向不稳定，动态边不应主导。最终融合：

\[
A^t=\alpha A^{dist}+(1-\alpha)A^{wind,t}
\]

默认 \(\alpha=0.5\)，仅在验证集上从：

```text
0.25、0.50、0.75
```

中选择。

### 8.4.2 预测时可用信息

对预测起点 \(t\)，动态图只能由 \(t\) 或之前的风场生成。建议使用最近 3 小时向量平均：

\[
\bar{u}=\frac{1}{3}\sum_{\tau=t-2}^{t}v_\tau\cos\psi_\tau,\quad
\bar{v}=\frac{1}{3}\sum_{\tau=t-2}^{t}v_\tau\sin\psi_\tau
\]

再由 \((\bar{u},\bar{v})\) 计算平均传播方向和有效风速。

## 8.5 图质量检查

必须输出：

- 节点和边地图；
- 每个节点的度；
- 边距离分布；
- 邻接矩阵热力图；
- 动态边随典型风向变化的示例；
- 孤立节点检查；
- 行列中 NaN 和无穷值检查。

---

## 9. 基线模型

所有基线与主模型必须使用相同的时间切分、目标时距和评估脚本。

## 9.1 历史均值

对每个站点、星期类型和小时计算训练集均值：

```text
station_id × hour_of_day × is_weekend
```

若分组样本不足，退化为：

```text
station_id × hour_of_day
```

## 9.2 持久性预测

\[
\hat{Y}_{t+h}=Y_t
\]

分别评估 \(h=1,3,6\)。这是短时预测必须超过的基线。

## 9.3 LightGBM

每个“预测起点—目标站点—预测时距”形成一条表格样本。

特征：

- 目标站点 PM2.5 的 1、2、3、6、12、24 小时滞后；
- 目标站点其他污染物的 1、3、6 小时滞后；
- 当前及历史气象；
- 最近 3 小时和 6 小时均值、标准差、斜率；
- 最近 3 个图邻居 PM2.5 的加权均值；
- 时间周期特征；
- 缺失掩码；
- `station_id` 类别编码或独热编码；
- `horizon`。

建议先训练一个联合模型，把站点和时距作为特征；若效果明显较差，再按时距训练三个模型。

默认参数起点：

```yaml
objective: regression_l1
n_estimators: 1000
learning_rate: 0.03
num_leaves: 31
max_depth: -1
subsample: 0.8
colsample_bytree: 0.8
reg_lambda: 1.0
early_stopping_rounds: 50
```

只用验证集调参。

## 9.4 GRU

输入形状：

```text
[batch, time=24, node*feature]
```

或共享节点 GRU：

```text
[batch*node, time=24, feature]
```

推荐采用共享节点 GRU，并加入可学习的站点嵌入。这样模型利用时间特征，但不显式聚合邻居，可作为公平的无图深度基线。

默认设置：

```yaml
hidden_size: 64
num_layers: 2
dropout: 0.1
station_embedding_dim: 8
output_horizons: [1, 3, 6]
```

---

## 10. STGCN 理论核心

## 10.1 为什么需要同时建模时间和空间

令：

\[
X_t\in\mathbb{R}^{N\times F}
\]

表示时刻 \(t\) 的 \(N\) 个站点和 \(F\) 个特征。

只做时间模型，相当于学习：

\[
\hat{Y}=f(X_{t-23},\ldots,X_t)
\]

但没有明确告诉模型哪些站点在地理上相邻。

图卷积显式使用邻接矩阵：

\[
H^{(l+1)}
=\sigma\left(
\hat{A}H^{(l)}W^{(l)}
\right)
\]

每个站点的新表示由自身与邻居共同决定。时间卷积再对连续小时进行建模，从而形成时空联合表示。

## 10.2 时间卷积

沿时间维使用一维因果卷积。建议采用门控线性单元：

\[
\operatorname{TCN}(X)
=(X*W_f+b_f)
\odot
\sigma(X*W_g+b_g)
\]

其中：

- `*` 表示时间卷积；
- 第一部分产生候选信息；
- 第二部分是 0 至 1 的门；
- \(\odot\) 为逐元素乘法。

卷积必须采用左侧填充，确保时刻 \(t\) 的表示不读取未来。

## 10.3 ST-Conv 块

一个基础块：

```text
Temporal Conv → Graph Conv → Temporal Conv → LayerNorm → Dropout
```

堆叠两个块：

```text
Input
 → ST-Conv Block 1
 → ST-Conv Block 2
 → Temporal Pooling / Last-step representation
 → Fully Connected Head
 → [1h, 3h, 6h] × all stations
```

推荐张量约定：

```text
[batch, channels, time, nodes]
```

整个项目始终使用同一约定，避免时间维和节点维混淆。

## 10.4 第一版网络配置

```yaml
input_length: 24
num_nodes: auto
num_features: auto
hidden_channels: 64
temporal_kernel: 3
num_st_blocks: 2
dropout: 0.1
output_horizons: [1, 3, 6]
activation: glu
graph: distance_knn
```

如果样本量或显存不足，先把隐藏通道降至 32，不优先缩短研究期。

## 10.5 输出头

采用直接多时距输出：

```text
[batch, 3, nodes]
```

不使用递归预测，原因是：

- 避免误差逐步累积；
- 三个目标时距固定；
- 更便于按时距赋权和评估；
- 展示平台可以直接读取三个标准预测层。

## 10.6 损失函数

默认使用掩码 MAE：

\[
\mathcal{L}_{MAE}
=
\frac{
\sum_{b,h,n}M_{bhn}
\left|\hat{Y}_{bhn}-Y_{bhn}\right|
}{
\sum_{b,h,n}M_{bhn}
}
\]

其中 \(M\) 表示真实目标是否存在。

若高污染极端值训练不稳定，可比较 Huber Loss，但主结果仍应报告原尺度 MAE 和 RMSE。

可选时距权重：

```text
1h: 1.0
3h: 1.0
6h: 1.0
```

第一版等权，避免人为偏向某一预测时距。

---

## 11. 风向动态图模型

标准 STGCN 通常使用固定邻接矩阵。动态风向图不应只是训练前求一个平均矩阵，否则无法体现“动态”。

推荐扩展模型为 Wind-GCN-GRU：

1. 对输入窗口每个时刻构造 \(A^t\)；
2. 使用 \(A^t\) 对当时节点特征进行图卷积；
3. 得到每个节点的空间表示；
4. 对 24 个时间步的空间表示使用共享 GRU；
5. 直接输出未来 1、3、6 小时。

动态空间层：

\[
H_t=\sigma(\hat{A}^{t}X_tW_s)
\]

时间层：

\[
Z_t=\operatorname{GRU}(H_t,Z_{t-1})
\]

输出：

\[
\hat{Y}_{t+\{1,3,6\}}=W_oZ_t+b_o
\]

对照关系：

- GRU：无图；
- STGCN-distance：固定距离图；
- GCN-GRU-distance：统一架构中的固定图；
- Wind-GCN-GRU：统一架构中的动态图。

比较固定图与动态图时，优先使用相同的 GCN-GRU 主体，只改变邻接矩阵，避免把架构差异误判为动态图收益。

---

## 12. 模型训练

## 12.1 训练配置

默认：

```yaml
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 64
max_epochs: 200
early_stopping_patience: 20
lr_scheduler: ReduceLROnPlateau
lr_factor: 0.5
lr_patience: 5
gradient_clip_norm: 5.0
loss: masked_mae
seeds: [42, 3407, 2026]
```

如显存不足，将批量降到 32 或 16；不得改变测试集。

## 12.2 单轮流程

```python
for batch in train_loader:
    model.train()
    optimizer.zero_grad()

    prediction = model(batch.x, graph=batch.graph)
    loss = masked_mae(prediction, batch.y, batch.y_mask)

    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
```

每轮结束：

1. 在验证集计算原尺度 MAE；
2. 学习率调度器根据验证 MAE 更新；
3. 保存最优检查点；
4. 写入训练日志；
5. 达到早停条件后结束。

## 12.3 超参数选择

按以下顺序调整，避免无目的搜索：

1. 输入窗口：12、24、48；
2. 隐藏通道：32、64；
3. 图邻居数：2、3、4；
4. 学习率：\(3\times10^{-4}\)、\(10^{-3}\)、\(3\times10^{-3}\)；
5. Dropout：0、0.1、0.3；
6. 动态图融合系数：0.25、0.5、0.75。

超参数全部根据验证集选择。测试集在配置冻结后只运行最终评估。

不建议一开始使用大规模自动调参。先确保数据、基线和模型实现正确，再进行有限网格搜索。

## 12.4 训练正确性检查

正式训练前完成三个小实验：

### 检查 A：单批过拟合

用一个小批次训练到损失明显接近 0。若不能过拟合，优先检查：

- 张量维度；
- 标签对齐；
- 掩码；
- 输出逆变换；
- 图矩阵归一化。

### 检查 B：时间打乱

打乱输入时间顺序后，性能应变差。若几乎不变，模型可能没有利用时间结构。

### 检查 C：图置换

随机置换邻接矩阵后，图模型性能应下降或发生可解释变化。若完全不变，检查图卷积是否真正进入前向传播。

---

## 13. 实验矩阵

## 13.1 必做实验

```text
B0  Historical Mean
B1  Persistence
B2  LightGBM
B3  GRU
M1  STGCN + Distance Graph
M2  GCN-GRU + Distance Graph
M3  Wind-GCN-GRU
```

M2 用于与 M3 做严格的固定图—动态图比较；M1 是标准时空图卷积主模型。

## 13.2 特征消融

```text
F0  仅 PM2.5
F1  PM2.5 + 时间特征
F2  F1 + 其他污染物
F3  F1 + 气象变量
F4  PM2.5 + 时间 + 其他污染物 + 气象 + 缺失掩码
```

## 13.3 图消融

```text
G0  无图
G1  距离图
G2  训练集历史相关图
G3  距离图 + 历史相关图
G4  距离图 + 风向动态图
```

## 13.4 缺失处理消融

至少比较：

- 仅前向填充；
- 前向填充 + 训练集季节中位数；
- 前向填充 + 季节中位数 + 缺失掩码。

## 13.5 实验命名

每个实验 ID 唯一：

```text
{model}_{graph}_{feature_set}_in24_h136_seed42
```

例如：

```text
stgcn_dist_f4_in24_h136_seed42
```

每个运行目录保存：

```text
config.yaml
environment.json
dataset_manifest.json
graph_manifest.json
train_log.csv
best.ckpt
predictions.parquet
metrics.json
```

---

## 14. 评估方法

## 14.1 指标

### MAE

\[
MAE=\frac{1}{K}\sum|\hat{y}-y|
\]

作为主指标，单位为 \(\mu g/m^3\)。

### RMSE

\[
RMSE=\sqrt{\frac{1}{K}\sum(\hat{y}-y)^2}
\]

用于反映大误差。

### \(R^2\)

\[
R^2=1-\frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}
\]

### SMAPE

\[
SMAPE=
\frac{100\%}{K}
\sum
\frac{2|\hat{y}-y|}{|\hat{y}|+|y|+\epsilon}
\]

不把普通 MAPE 作为主指标，因为 PM2.5 接近 0 时比值会失真。

## 14.2 统计层级

必须输出：

- 全体样本总体指标；
- 每个预测时距；
- 每个站点；
- 每个月和季节；
- 工作日与周末；
- PM2.5 浓度区间；
- 风速区间；
- 数据缺失率区间。

污染浓度分层不在代码中硬编码标准名称。先保留原始浓度分箱，并在论文撰写时根据采用的国家标准版本给出规范解释。

## 14.3 高污染时段

建议同时使用：

1. 训练集 PM2.5 第 90 分位数；
2. 所采用空气质量标准中的明确阈值。

第一种便于数据驱动比较，第二种便于现实解释。阈值与标准版本必须写入配置和论文。

## 14.4 显著性

不能把每个高度相关的小时—站点误差都视为独立样本。

建议：

1. 先按自然日计算各模型 MAE；
2. 对两个模型的逐日 MAE 差值做配对分析；
3. 使用按日块自助法计算 95% 置信区间；
4. 同时报告效应量，即 MAE 绝对下降和相对下降。

## 14.5 预测结果标准表

`predictions.parquet`：

```text
run_id
model
graph_type
origin_time
target_time
horizon
station_id
y_true
y_pred
y_mask
split
season
is_weekend
pm25_bin
wind_speed_bin
```

该表同时服务于：

- 评估脚本；
- 论文表格和图；
- 典型案例筛选；
- 展示平台。

---

## 15. 解释和案例分析

## 15.1 误差地图

按站点聚合：

```text
station_id, horizon, MAE, RMSE, bias, sample_count
```

连接站点坐标后输出 GeoJSON，地图颜色表示误差。

## 15.2 图结构解释

对每个站点展示：

- 地理邻居；
- 相关图邻居；
- 典型风向下的上风向邻居；
- 移除邻居前后的误差变化。

## 15.3 遮挡实验

分别遮挡：

- 最近 3 小时；
- 更早 4 至 12 小时；
- 更早 13 至 24 小时；
- 目标站自身历史；
- 空间邻居历史；
- 气象特征。

用误差增量判断模型依赖哪些信息。注意力权重如果存在，只作为辅助，不能代替遮挡验证。

## 15.4 案例选择

通过固定规则选取，不手工挑最好看的样本：

1. 模型显著优于持久性的高污染事件；
2. 模型显著差于持久性的失败事件；
3. 强风条件下动态图优于固定图的事件；
4. 弱风条件下动态图无优势的事件；
5. 数据缺失较多但模型仍稳定的事件。

每个案例保存：

```text
case_id
selection_rule
time_range
stations
observations
predictions
weather
graph_edges
error_summary
```

---

## 16. 为论文产出预留的实现约束

本文件不展开论文写作步骤，但工程从一开始必须保存以下材料，否则后期难以形成可信论文。

### 16.1 方法可复现

- 数据来源和下载日期；
- 原始文件哈希；
- 数据清洗配置；
- 固定的数据划分边界；
- 站点顺序；
- 邻接矩阵生成参数；
- 模型配置；
- 随机种子；
- 最优检查点；
- 完整预测明细。

### 16.2 可自动生成的论文图

至少预留脚本生成：

- 研究区域与站点地图；
- 整体技术路线图；
- 数据缺失率图；
- PM2.5 时间和空间分布图；
- 图结构示意图；
- 模型总体对比图；
- 分预测时距对比图；
- 站点误差地图；
- 特征消融图；
- 图结构消融图；
- 典型成功和失败案例。

### 16.3 可自动生成的论文表

以 CSV 或 LaTeX 兼容格式导出：

- 数据集统计；
- 模型参数量和训练时间；
- 总体结果；
- 分时距结果；
- 消融结果；
- 分层结果；
- 显著性结果。

禁止从终端手工抄数字到论文。所有表格由固定脚本从 `metrics.json` 和 `predictions.parquet` 生成。

---

## 17. 为成果展示平台预留的实现约束

本文件不展开平台技术选型和页面开发计划，但训练流水线必须输出稳定、与模型代码解耦的数据接口。

### 17.1 平台需要的数据

站点：

```text
station_id,name,district,longitude,latitude
```

时间序列：

```text
timestamp,station_id,observed_pm25
```

预测：

```text
origin_time,target_time,horizon,station_id,predicted_pm25,model
```

误差：

```text
station_id,horizon,MAE,RMSE,bias
```

图边：

```text
timestamp,source_station,target_station,weight,graph_type
```

案例：

```text
case_id,title,start_time,end_time,summary
```

### 17.2 统一导出格式

训练结束后执行一个独立导出步骤：

```text
python -m src.visualization.export_platform --run-id <RUN_ID>
```

生成：

```text
outputs/platform/
├─ metadata.json
├─ stations.geojson
├─ observations.parquet
├─ forecasts.parquet
├─ station_metrics.json
├─ graph_static.geojson
├─ graph_dynamic/
└─ cases/
```

展示平台只读取这些稳定产物，不直接加载训练检查点，也不现场重新训练。

### 17.3 平台演示所需的时间快照

为降低平台运行压力，预先生成：

- 每小时站点浓度；
- 每个预测起点的 1、3、6 小时结果；
- 典型事件的动态图边；
- PM2.5 色阶；
- 站点弹窗摘要；
- 模型切换所需的同一格式预测。

这样后续无论选择 Streamlit、FastAPI 加前端，还是纯静态可视化，都无需重做数据层。

---

## 18. 执行顺序与阶段验收

## 阶段 A：工程初始化

任务：

- 建立目录；
- 创建环境和锁文件；
- 创建配置读取、日志和随机种子模块；
- 建立测试框架。

通过条件：

- 新环境可安装；
- 能运行空测试；
- 配置和日志路径统一。

## 阶段 B：小样本数据验证

先只下载：

```text
2025-01-01 至 2025-01-07
```

任务：

- 验证空气质量文件结构；
- 核对北京站点列；
- 请求一个站点气象数据；
- 打通宽表转长表和连接。

通过条件：

- 七天数据可生成规范长表；
- 时间、单位、站点坐标正确；
- 无主键重复。

## 阶段 C：全量数据构建

任务：

- 下载候选三年；
- 完成文件校验和清单；
- 下载逐站气象；
- 清洗、连接和数据审计；
- 确定最终研究期和站点集合。

通过条件：

- 数据质量门槛通过；
- 生成数据集版本；
- 审计结果已冻结。

## 阶段 D：基线

任务：

- 固定划分；
- 生成窗口；
- 实现 Historical Mean、Persistence、LightGBM、GRU；
- 验证指标和预测明细导出。

通过条件：

- 四个基线均有可复现结果；
- 模型优劣符合基本常识；
- 持久性预测计算正确。

## 阶段 E：固定图模型

任务：

- 距离图；
- STGCN；
- GCN-GRU 固定图版本；
- 图置换和单批过拟合检查。

通过条件：

- 模型稳定训练；
- 图确实参与前向计算；
- 完成三随机种子主结果。

## 阶段 F：动态图与消融

任务：

- 风向传播方向计算；
- 动态图生成；
- Wind-GCN-GRU；
- 特征、图和缺失处理消融；
- 分层评估。

通过条件：

- 固定图和动态图主体架构一致；
- 所有核心假设有实验对应；
- 结果无论正负均可解释。

## 阶段 G：论文和平台数据产品

任务：

- 冻结最终运行；
- 自动生成图表数据；
- 导出平台数据；
- 固定典型案例；
- 创建最终数据和模型卡。

通过条件：

- 论文数字可自动重生；
- 平台无需读取训练代码即可展示结果；
- 最终运行目录不可再被覆盖。

---

## 19. 建议命令接口

最终工程应提供类似命令：

```powershell
# 1. 下载空气质量
python -m src.data.download_air_quality `
  --start 2023-01-01 `
  --end 2025-12-31

# 2. 下载气象
python -m src.data.download_weather `
  --stations data/raw/stations/stations.csv `
  --start 2023-01-01 `
  --end 2025-12-31

# 3. 校验原始数据
python -m src.data.validate_raw --config configs/data.yaml

# 4. 构建数据集
python -m src.data.build_dataset `
  --data-config configs/data.yaml `
  --feature-config configs/features.yaml

# 5. 构建图
python -m src.features.graph_distance --config configs/graph.yaml
python -m src.features.graph_correlation --config configs/graph.yaml

# 6. 训练
python -m src.training.train --config configs/stgcn.yaml

# 7. 评估
python -m src.training.predict --run-id <RUN_ID> --split test
python -m src.evaluation.metrics --run-id <RUN_ID>
python -m src.evaluation.stratified --run-id <RUN_ID>

# 8. 导出论文图表和平台数据
python -m src.visualization.curves --run-id <RUN_ID>
python -m src.visualization.maps --run-id <RUN_ID>
python -m src.visualization.export_platform --run-id <RUN_ID>
```

这些是目标接口，不要求第一天全部实现。每个阶段按接口逐步补齐。

---

## 20. 关键决策记录

在 `docs/decisions.md` 中记录每个会改变研究结论的决定：

```text
日期
决策
备选方案
证据
最终选择
影响范围
```

必须记录：

- 最终研究期；
- 移除的站点；
- 缺失率阈值；
- 插补规则；
- 输入窗口；
- 图邻居数；
- 是否使用相关图；
- 动态图公式；
- 最终主模型；
- 高污染阈值；
- 最终论文使用的运行 ID。

---

## 21. 终止和回退条件

出现以下情况时不要继续堆叠模型。

### 数据回退

若三年数据完整性不足：

1. 缩短为最近连续两年；
2. 再评估站点；
3. 仍不足时更换起止年份；
4. 不用大规模插补伪造连续数据。

### 图模型回退

若 STGCN 不稳定：

1. 单批过拟合；
2. 检查邻接矩阵；
3. 减少隐藏通道；
4. 暂停动态图；
5. 先用 GCN-GRU 固定图打通。

### 动态图回退

若动态图无提升：

1. 确认风向“来自方向”已转为传播方向；
2. 检查低风速抑制；
3. 比较最近 1 小时和 3 小时向量平均；
4. 做强风子集分析；
5. 若仍无提升，将其作为负结果，不继续增加复杂模块。

---

## 22. 最终交付清单

### 数据

- [ ] 原始空气质量文件及清单；
- [ ] 原始气象响应及清单；
- [ ] 版本化站点元数据；
- [ ] 清洗后长表；
- [ ] 模型张量；
- [ ] 数据质量报告；
- [ ] 数据集卡和数据来源说明。

### 模型

- [ ] Historical Mean；
- [ ] Persistence；
- [ ] LightGBM；
- [ ] GRU；
- [ ] STGCN；
- [ ] GCN-GRU 固定图；
- [ ] Wind-GCN-GRU；
- [ ] 最终检查点和配置。

### 实验

- [ ] 三个预测时距；
- [ ] 三个随机种子；
- [ ] 特征消融；
- [ ] 图结构消融；
- [ ] 缺失处理消融；
- [ ] 分站点和分场景评估；
- [ ] 显著性分析；
- [ ] 成功与失败案例。

### 论文准备

- [ ] 自动生成论文图；
- [ ] 自动生成论文表；
- [ ] 实验运行与论文数字对应；
- [ ] 方法、数据和模型可复现。

### 平台准备

- [ ] 站点 GeoJSON；
- [ ] 观测时间序列；
- [ ] 多模型预测结果；
- [ ] 静态和动态图边；
- [ ] 站点误差；
- [ ] 典型案例；
- [ ] 平台元数据。

---

## 23. 参考数据入口

1. QuotSoft 中国空气质量历史数据说明：  
   <https://quotsoft.net/air/>
2. 全国站点单日 CSV 地址模板：  
   `https://quotsoft.net/air/data/china_sites_YYYYMMDD.csv`
3. Open-Meteo Historical Weather API 文档：  
   <https://open-meteo.com/en/docs/historical-weather-api>
4. Open-Meteo 历史天气接口：  
   <https://archive-api.open-meteo.com/v1/archive>
5. 国控监测站点列表核验页：  
   <https://docs.datastation.skyviewor.com/data-query/china-pollution/pollution-stations-list/>
6. 北京市公共数据开放平台空气质量数据条目，可作为辅助核验：  
   <https://data.beijing.gov.cn/zyml/hcsjj/2979bdd1ec5f43b48305bb670a69ff98.htm>

---

## 24. 本方案的最终执行原则

1. **数据先于模型**：数据质量审计未通过前，不进行正式训练。
2. **基线先于改进**：不超过持久性和 LightGBM 时，不宣称图模型有效。
3. **因果先于精度**：任何未来信息泄漏都会使高精度失去意义。
4. **固定图先于动态图**：先建立可靠主线，再检验风向机制。
5. **明细先于汇总**：保存逐时、逐站预测，不能只保存一个平均指标。
6. **复现先于美化**：论文图和平台数据均由流水线自动生成。
7. **负结果也可报告**：动态图或复杂模型无提升时，用消融和场景分析解释，而不是隐藏结果。
8. **一次建模、两类交付**：同一套实验产物同时支撑论文和成果展示平台，避免后期重复加工。

---

## 25. 从零开始的逐步执行手册

本节是项目的主执行清单。建议在 `docs/progress.md` 中复制所有步骤标题，并为每一步维护以下状态：

```text
NOT_STARTED：尚未开始
IN_PROGRESS：执行中
BLOCKED：存在阻塞
REVIEW：等待验收
DONE：已通过验收
INVALIDATED：因上游变化而失效，必须重做
```

每一步完成后记录：

```text
执行日期
执行人
代码提交哈希
数据集版本
运行 ID
输出目录
验收结果
已知问题
下一步
```

### 步骤 0：冻结研究协议

#### 目的

在下载和建模前固定研究问题、目标和评测方式，防止看到测试结果后修改任务定义。

#### 前置输入

- 课程要求；
- 最终课题题目；
- 可用周期、计算设备和磁盘空间；
- 本执行方案。

#### 具体操作

1. 创建 `docs/research_protocol.md`。
2. 写入唯一主任务：

   ```text
   使用北京市国控监测站过去 24 小时的数据，
   预测全部保留站点未来第 1、3、6 小时的 PM2.5。
   ```

3. 固定四个研究问题：
   - 图结构是否优于无图模型；
   - 气象和其他污染物是否有效；
   - 风向动态图是否优于固定距离图；
   - 模型在哪些场景有效或失效。
4. 固定主指标为测试集原尺度 MAE，辅助指标为 RMSE、宏平均 MAE、\(R^2\)、SMAPE。
5. 固定主基线为 Persistence、LightGBM 和共享节点 GRU。
6. 固定主模型为 STGCN-distance。
7. 固定扩展模型为 Wind-GCN-GRU。
8. 固定测试集规则：模型、超参数和特征确定前不可查看测试指标。
9. 固定最低成果：
   - 一套可复现数据；
   - 至少 4 个基线；
   - 1 个固定图模型；
   - 1 个动态图模型；
   - 特征和图消融；
   - 分层评估；
   - 论文所需图表数据；
   - 平台所需标准数据包。
10. 创建 `docs/decisions.md`，后续所有协议变化必须登记。

#### 输出

```text
docs/research_protocol.md
docs/decisions.md
```

#### 验收

- 每个研究问题都能对应至少一个对照实验；
- 指标、时距和主要模型不存在两种解释；
- 测试集使用规则已明确；
- 预计工作量不超过实际课程周期。

#### 失败处理

若范围过大，按以下顺序缩减：

1. 保留固定图，暂缓动态图；
2. 减少辅助模型，不减少强基线；
3. 保留 1、3、6 小时，不扩展更多时距；
4. 不删除数据质量、泄漏检查和分层评估。

---

### 步骤 1：评估资源并建立预算

#### 目的

提前确定磁盘、内存、显存和时间边界，避免全量下载后才发现无法训练。

#### 具体操作

1. 记录 CPU、内存、GPU 型号和显存、可用磁盘空间。
2. 为项目预留：
   - 原始 CSV/JSON：建议至少 10 GB；
   - 中间 Parquet 与张量：建议至少 10 GB；
   - 模型、预测和图表：建议至少 5 GB；
   - 总安全余量：至少 30 GB。
3. 记录每周可投入时间。
4. 设定训练预算：
   - 单次主模型最长 2 小时；
   - 单个超参数搜索最多 20 次 trial；
   - 主结果使用 3 个随机种子；
   - 非候选配置只使用 1 个随机种子。
5. 无 GPU 时启用 CPU 最小方案：
   - 隐藏通道 32；
   - 批量 16；
   - 先用一年数据验证；
   - 完整数据只训练最终候选。

#### 输出

```text
docs/resources.md
```

#### 验收

- 磁盘余量达到预算；
- 至少能在 CPU 上运行七天小样本；
- 训练预算与课程剩余时间相容。

#### 失败处理

资源不足时先减小模型和搜索次数，不通过降低数据质量标准解决。

---

### 步骤 2：初始化工程

#### 目的

建立可安装、可测试、可追踪的项目骨架。

#### 具体操作

1. 在项目根目录初始化 Git：

   ```powershell
   git init
   ```

2. 创建 `.gitignore`，至少排除：

   ```text
   .venv/
   __pycache__/
   .pytest_cache/
   .ruff_cache/
   data/raw/
   data/interim/
   data/processed/
   outputs/
   *.ckpt
   *.pt
   *.part
   ```

3. 创建 `.gitattributes`：

   ```text
   * text=auto
   *.py text eol=lf
   *.md text eol=lf
   *.yaml text eol=lf
   *.csv text eol=lf
   ```

4. 创建虚拟环境：

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

5. 建立第 3.2 节规定的目录。
6. 创建 `pyproject.toml`，配置依赖、ruff、pytest 和包信息。
7. 安装可编辑包：

   ```powershell
   pip install -e ".[dev]"
   ```

8. 创建统一日志模块：
   - 控制台输出；
   - 日志写入 `outputs/runs/{run_id}/run.log`；
   - 每条日志包含时间、级别、模块和运行 ID。
9. 创建统一配置加载器：
   - YAML 读取；
   - 缺少必填项立即报错；
   - 最终解析配置复制到运行目录。
10. 创建随机种子函数，同时设置 Python、NumPy、PyTorch 和 CUDA。
11. 创建 `scripts/smoke.ps1`，执行格式检查、单元测试和七天样例流程。
12. 完成初始提交。

#### 输出

```text
pyproject.toml
.gitignore
.gitattributes
src/
tests/
configs/
scripts/smoke.ps1
```

#### 验收

在新 PowerShell 中执行：

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import src"
ruff check .
pytest -q
```

全部成功，且退出码为 0。

#### 失败处理

- PowerShell 禁止激活脚本时，在当前用户范围调整执行策略或直接使用 `.venv\Scripts\python.exe`；
- CUDA 依赖失败时先安装 CPU 版 PyTorch；
- 不允许在系统 Python 中继续安装依赖。

---

### 步骤 3：建立统一配置和数据契约

#### 目的

先定义每类数据“应该是什么”，再编写下载和清洗代码。

#### 具体操作

1. 创建 `configs/data.yaml`：

   ```yaml
   timezone: Asia/Shanghai
   candidate_start: 2023-01-01
   candidate_end: 2025-12-31
   pollutants: [PM2.5, PM10, SO2, NO2, O3, CO]
   station_pm25_min_valid_ratio: 0.90
   auxiliary_min_valid_ratio: 0.85
   max_forward_fill_hours: 3
   ```

2. 创建 `docs/data_dictionary.md`。
3. 为每个字段记录：
   - 字段名；
   - 中文含义；
   - 数据类型；
   - 单位；
   - 来源；
   - 主键关系；
   - 可用时刻；
   - 合法范围；
   - 缺失表示；
   - 变换规则；
   - 是否可用作模型输入。
4. 固定时间类型为带 `Asia/Shanghai` 时区的 ISO 8601。
5. 固定主键：
   - 空气质量长表：`timestamp + station_id + pollutant`；
   - 最终观测表：`timestamp + station_id`；
   - 预测表：`run_id + origin_time + horizon + station_id`。
6. 为 raw、clean、feature、prediction 分别创建 Pandera 或等价 schema。
7. 制作一个合法样例和多个故意错误样例。

#### 输出

```text
configs/data.yaml
docs/data_dictionary.md
src/data/schemas.py
tests/fixtures/
tests/test_schemas.py
```

#### 验收

- 合法样例通过；
- 错误时区、重复主键、非法湿度和缺列样例均失败；
- 字典中的每个模型特征都注明“预测时何时可获得”。

#### 失败处理

字段含义或单位无法确认时，将其列为 `QUARANTINED`，不得进入后续流程。

---

### 步骤 4：执行数据源准入验证

#### 目的

正式下载前确认地址、格式、许可、时间含义和稳定性。

#### 具体操作

1. 空气质量归档至少抽查以下日期：
   - 每个候选年份的 1 月 1 日；
   - 每个候选年份的 7 月 1 日；
   - 闰年 2 月 28 日和 29 日；
   - 月末与次月 1 日；
   - 随机 10 天。
2. 对每个样例记录：
   - HTTP 状态；
   - 响应类型；
   - 文件大小；
   - 编码；
   - 列名；
   - `hour` 最小值、最大值和数量；
   - `type` 集合；
   - 北京站点列数量。
3. 专门核验 `hour=0` 和可能的 `hour=24`：
   - 比较相邻日期；
   - 确认时间代表统计开始还是统计结束；
   - 把结论写入 `source_validation_report.md`。
4. 从北京市公共数据或其他可靠页面随机核验 3 个站点、3 个时刻，允许因数据发布时间和修订出现差异，但必须记录。
5. 核验站点元数据版本和坐标。
6. Open-Meteo 至少抽查 3 个城区站和 3 个郊区站，检查：
   - 时区；
   - 返回数组长度；
   - 单位；
   - 返回网格坐标；
   - 多个站点是否映射到同一网格；
   - 请求年份间 schema 是否一致。
7. 保存各来源条款、引用方式和访问日期。
8. 固定 Open-Meteo 的模型参数；若接口不能稳定固定底层产品，则在 manifest 中保存 API 返回元数据并披露这一限制。

#### 输出

```text
docs/source_validation_report.md
data/raw/source_probe/
data/manifests/source_probe.csv
```

#### 量化验收

- 抽查空气质量文件成功率至少 95%；
- 成功文件的必需字段一致率 100%；
- 小时编码规则得到唯一解释；
- 气象返回时间与北京时间逐小时对应；
- 数据来源和引用信息完整。

#### 失败处理

- 单日缺失：登记，继续抽查；
- 系统性无法访问：暂停全量下载，启用归档包或更换经验证的数据源；
- 字段随年份改变：为不同 schema 编写版本解析器；
- 不得静默改用不同来源并拼接。

---

### 步骤 5：建立站点版本表

#### 目的

确定研究期内每个节点的身份、名称、坐标和有效期。

#### 具体操作

1. 收集各版本站点列表。
2. 统一字段为：

   ```text
   station_id,name,province,city,district,longitude,latitude,
   valid_from,valid_to,source,source_version
   ```

3. 按 `station_id` 比较不同版本：
   - 名称变化；
   - 区县变化；
   - 坐标变化；
   - 新增或撤销。
4. 计算同一代码前后坐标距离。
5. 若坐标变化小于 1 km，保留同一节点并记录；
6. 若大于等于 1 km，人工核验迁址：
   - 可缩短研究期；
   - 或将迁址前后拆为不同内部节点；
   - 不允许无说明地沿用最新坐标。
7. 输出候选北京站点，不在此步骤按测试期完整率筛选。

#### 输出

```text
data/raw/stations/stations_versioned.csv
docs/station_changes.md
```

#### 验收

- `station_id + valid_from` 唯一；
- 所有候选站都有坐标和来源；
- 坐标落在合理地理范围；
- 所有迁址和撤站均有处理决定。

#### 失败处理

无法可靠确认坐标或身份的站点先排除，并保留排除理由。

---

### 步骤 6：打通七天端到端小样本

#### 目的

在全量下载前验证整条流水线的接口和形状。

#### 数据范围

```text
2025-01-01 至 2025-01-07
至少 3 个有效站点
六种污染物和全部气象变量
```

#### 具体操作

1. 下载七天空气质量文件。
2. 下载对应站点气象数据。
3. 完成宽表转长表。
4. 完成时间和站点连接。
5. 生成最简缺失掩码和前向填充值。
6. 构造距离图。
7. 生成输入 24 小时、预测 1/3/6 小时的窗口。
8. 运行 Persistence。
9. 运行一个最小 GRU 或单层 GCN。
10. 导出标准预测表。
11. 保存并重新加载模型，再次推理。

#### 输出

```text
data/interim/smoke/
outputs/runs/smoke_*/
```

#### 验收

- 最终长表主键唯一；
- 张量形状符合 `[sample, time, node, feature]`；
- 标签时间严格晚于预测起点；
- 模型可在一个批次上过拟合；
- 保存前后同一输入预测差异小于 \(10^{-6}\)；
- 标准预测表通过 schema。

#### 失败处理

任何一项失败都停留在七天样例，不得启动全量下载。

---

### 步骤 7：下载全量原始数据

#### 目的

获取不可修改、可校验、可断点恢复的原始数据。

#### 具体操作

1. 运行空气质量下载器，按天请求。
2. 每个响应先写 `.part`。
3. 校验 CSV 后原子重命名。
4. 为文件计算 SHA-256。
5. 记录 404、超时、无效内容和重试次数。
6. 下载气象数据，按“站点 × 年”存储。
7. 原始响应一旦通过校验不得覆盖。
8. 若来源后来修订，保存为新版本并保留旧文件。
9. 下载结束后重跑失败清单。
10. 生成日期覆盖日历。

#### 输出

```text
data/raw/air_quality/
data/raw/weather/
data/manifests/raw_files.csv
data/manifests/missing_dates.csv
```

#### 量化验收

- 候选时间范围日期文件覆盖率至少 98%；
- 已下载文件哈希覆盖率 100%；
- 通过结构校验的文件比例 100%；
- 所有失败都有明确状态；
- 原始目录不存在残留 `.part` 文件。

#### 失败处理

- 覆盖率低于 98%：按月定位集中缺档；
- 缺档集中在研究期边缘：考虑收缩研究期；
- 缺档位于中间且连续多日：不得插补整段，后续排除跨缺档窗口；
- 数据源整体不可用：回到步骤 4。

---

### 步骤 8：原始数据审计与探索

#### 目的

在制定清洗规则前认识真实数据问题和规律。

#### 具体操作

1. 输出每天文件数、记录数、站点数和类型数。
2. 绘制日期覆盖热力图。
3. 为每个站点绘制启用—停用时间轴。
4. 统计每个站点和变量：
   - 总记录数；
   - 非缺失数；
   - 缺失率；
   - 最大连续缺失；
   - 重复数；
   - 冲突数；
   - 最小值、最大值、分位数。
5. 绘制：
   - 站点 × 日期缺失热力图；
   - PM2.5 分布和对数分布；
   - 月度、季节、小时分布；
   - 典型站点时间曲线；
   - 自相关函数；
   - 站点间相关矩阵；
   - 距离与相关性散点图；
   - 污染物与气象的滞后相关图；
   - 风向、风速与下风向站点时滞关系。
6. 比较候选年份的分布，检查数据漂移。
7. 每个异常都写明数量、位置、可能原因和拟采用规则。

#### 输出

```text
outputs/data_audit/raw/
docs/raw_data_audit.md
```

#### 验收

- 每个候选站都有覆盖和缺失结论；
- 研究期内每个整日缺档均被识别；
- 清洗规则均有 EDA 证据；
- 研究期和站点选择不依赖测试模型结果。

#### 失败处理

无法解释的异常不得直接删除；先隔离并标记为待核验。

---

### 步骤 9：冻结研究期和站点集合

#### 目的

把候选数据转化为固定研究对象。

#### 具体操作

1. 根据原始审计选择最长连续两至三年。
2. 站点筛选主要依据研究期总体可用性和部署稳定性；同时单独报告训练、验证、测试各期覆盖率，避免隐藏测试期退化。
3. 应用门槛：
   - PM2.5 总有效率至少 90%；
   - 任一划分期 PM2.5 有效率至少 80%；
   - 单次连续缺失原则上不超过 72 小时；
   - 坐标可靠；
   - 研究期内节点身份稳定。
4. 若站点不足 8 个，按顺序处理：
   - 更换连续年份；
   - 缩短为两年；
   - 检查站点版本；
   - 重新评估数据源。
5. 生成固定节点顺序。
6. 对每个排除站点给出机器可读原因。

#### 输出

```text
configs/study_period.yaml
data/processed/station_ids.json
data/processed/stations.csv
data/processed/excluded_stations.csv
```

#### 验收

- 节点数不少于 8；
- 节点顺序唯一且后续不可隐式改变；
- 所有划分期均可计算指标；
- 排除规则可复现。

#### 失败处理

研究期或站点改变后，步骤 10 以后全部标记为 `INVALIDATED`。

---

### 步骤 10：解析和清洗空气质量数据

#### 目的

从每日宽表得到带质量标记的规范长表。

#### 具体操作

1. 逐文件读取，只加载必要列。
2. 根据步骤 4 确认的规则解析日期和小时。
3. 只保留六种小时污染物。
4. 宽表转长表。
5. 数值解析失败时保留原字符串到错误日志。
6. 执行去重规则：
   - 完全相同：保留一条；
   - 缺失与非缺失：保留非缺失；
   - 非缺失冲突：值设为缺失，写冲突表。
7. 处理非法负值。
8. 对极端正值只打标，不自动删除。
9. 使用 MAD、时间连续性和邻站一致性生成异常候选。
10. 人工复核只产生一个版本化覆盖表：

    ```text
    timestamp,station_id,pollutant,decision,reason,reviewer
    ```

11. 任何人工处理必须由代码读取覆盖表执行，不直接编辑清洗后文件。
12. 输出清洗前后对账：
    - 输入行数；
    - 展开后理论行数；
    - 重复处理数；
    - 冲突数；
    - 非法数；
    - 最终非缺失数。

#### 输出

```text
data/interim/air_quality_long.parquet
data/interim/air_quality_quality_flags.parquet
data/interim/manual_overrides.csv
outputs/data_audit/air_quality_reconciliation.json
```

#### 验收

- 主键唯一；
- 时间单调且整点化；
- 单位一致；
- 所有数值变化都有原因代码；
- 清洗前后数量可以对账。

#### 失败处理

对账不守恒时立即停止，不得通过删除重复索引绕过。

---

### 步骤 11：解析和清洗气象数据

#### 具体操作

1. 读取每个站点每年的原始 JSON。
2. 验证所有变量数组与 `hourly.time` 等长。
3. 解析 `timezone` 和 `utc_offset_seconds`。
4. 展开为逐站逐小时长表。
5. 统一单位。
6. 执行合法范围检查。
7. 对相同网格返回的多个站点保留各自 `station_id`，并新增 `grid_id`。
8. 计算：
   - `wind_sin`；
   - `wind_cos`；
   - 传播方向；
   - 风的东西、南北向量分量。
9. 保存 API 返回坐标、请求坐标及两者距离。
10. 输出清洗对账。

#### 输出

```text
data/interim/weather_long.parquet
data/interim/weather_grid_mapping.csv
outputs/data_audit/weather_reconciliation.json
```

#### 验收

- 主键 `timestamp + station_id` 唯一；
- 数组展开无丢失；
- 时区与空气质量一致；
- 湿度、风速和风向范围合法；
- 网格映射关系明确。

#### 失败处理

某站点气象缺失时优先重新请求；仍失败才使用最近网格，并记录替代距离。

---

### 步骤 12：多源连接并重建完整时间网格

#### 具体操作

1. 创建研究期内完整的：

   ```text
   小时 × 固定站点
   ```

   笛卡尔索引。
2. 左连接污染物数据。
3. 左连接气象数据。
4. 左连接站点有效期内的元数据。
5. 为每个原始变量创建观测掩码。
6. 检查同一时间站点坐标是否符合版本表。
7. 计算连接匹配率。
8. 保存未插补的“事实表”。

#### 输出

```text
data/interim/observations_unimputed.parquet
outputs/data_audit/join_report.json
```

#### 量化验收

- 行数严格等于 `小时数 × 站点数`；
- 主键唯一率 100%；
- 连接前后 PM2.5 非缺失数完全一致；
- 气象匹配率至少 99%；
- 无未识别站点。

#### 失败处理

匹配率不足时按站点、年份定位，不允许直接对整个气象表前向填充后掩盖连接错误。

---

### 步骤 13：冻结时间划分

#### 目的

先划分，再拟合任何会学习数据分布的处理器。

#### 具体操作

1. 按完整自然日计算 70%/10%/20% 边界。
2. 将边界写死在配置中。
3. 每一行增加 `split`。
4. 窗口归属按标签时间决定：
   - 所有目标时距都落在同一 split，样本才属于该 split；
   - 验证和测试样本可以读取边界之前的过去 24 小时作为上下文；
   - 不允许读取任何目标时间之后的信息。
5. 生成边界附近的人工核验样例。

#### 输出

```text
configs/splits.yaml
tests/fixtures/split_boundary_cases.csv
```

#### 验收

- 划分无时间重叠；
- 测试时间严格晚于验证；
- 边界样本标签归属正确；
- 修改随机种子不会改变划分。

#### 失败处理

边界改变后，插补器、标准化器、相关图、样本和全部实验均失效。

---

### 步骤 14：拟合并执行缺失处理

#### 具体操作

1. 保留原值和掩码。
2. 对每站每变量按时间执行最多 3 小时的因果前向填充。
3. 仅用训练集拟合回退中位数：
   - `station × month × hour`；
   - 样本不足时退化为 `station × hour`；
   - 再不足时使用特征全局训练中位数。
4. 将插补统计量序列化。
5. 标签 PM2.5 永不插补。
6. 为每个插补值记录 `imputation_method`。
7. 在验证和测试上只执行 transform，不得 fit。

#### 输出

```text
data/processed/imputers.joblib
data/interim/observations_imputed.parquet
outputs/data_audit/imputation_report.json
```

#### 验收

- 测试证明处理器只见过训练期；
- 所有输入特征无 NaN，或明确由模型支持；
- 标签掩码与原始 PM2.5 缺失完全一致；
- 各插补方法数量可统计；
- 长缺失没有被错误地视为真实连续观测。

#### 失败处理

发现未来信息泄漏后，步骤 14 及以后所有产物作废并删除运行登记。

---

### 步骤 15：执行清洗后 EDA 和质量门禁

#### 具体操作

1. 比较清洗前后每个变量的分布。
2. 绘制插补比例地图和时间图。
3. 比较训练、验证、测试：
   - 均值；
   - 标准差；
   - 分位数；
   - 缺失率；
   - 高污染比例；
   - 季节构成。
4. 计算 PSI 或 Wasserstein 距离辅助判断漂移。
5. 核验插补是否削弱高污染峰值。
6. 生成最终数据审计结论。

#### 输出

```text
docs/final_data_audit.md
outputs/data_audit/final/
```

#### 验收

- PM2.5 目标未被插补；
- 输入插补比例在报告中披露；
- 训练和测试漂移被量化；
- 不存在无法解释的分布突变；
- 数据质量结论为 `PASS`。

#### 失败处理

质量门禁失败时回到步骤 9、10、11 或 14，并提升数据集版本。

---

### 步骤 16：建立特征注册表和生成特征

#### 具体操作

1. 创建 `docs/feature_registry.csv`：

   ```text
   feature_name,source,unit,formula,available_at,
   transform,imputation,scaling,used_by
   ```

2. 生成时间周期特征。
3. 生成风向正余弦。
4. 为 LightGBM 生成滞后：
   - 1、2、3、6、12、24 小时；
   - 滞后必须使用 `shift` 后再滚动。
5. 滚动统计只使用当前及过去：
   - 最近 3、6、12、24 小时均值；
   - 标准差至少需要 2 个有效值；
   - 线性斜率使用最小二乘；
   - 有效点不足时设为缺失并走训练期插补。
6. 神经网络保留逐时原始特征，不提前展开滞后列。
7. 生成特征可用时间自动检查：每个特征最大依赖时间不得晚于预测起点。
8. 对偏态降水执行 `log1p`，并记录变换顺序。
9. 序列化类别编码器、特征顺序和标准化器。

#### 输出

```text
docs/feature_registry.csv
data/processed/features_tabular.parquet
data/processed/feature_order.json
data/processed/preprocessors.joblib
```

#### 验收

- 每个特征都能追溯到公式和输入列；
- 泄漏审计通过；
- 特征顺序固定；
- 训练和离线推理对同一输入产生完全相同特征。

#### 失败处理

定义不明、可用时间不明或无法复现的特征直接移除。

---

### 步骤 17：构建并验证图

#### 具体操作

1. 固定矩阵语义：

   ```text
   A[i, j] 表示节点 i 接收节点 j 的信息。
   ```

2. 距离图：
   - Haversine 距离；
   - 每节点取 3 个最近邻；
   - 对称化后实际度数可以大于 3；
   - 高斯核边权；
   - 加自环；
   - 对称归一化。
3. 相关图：
   - 只用训练集；
   - 先去除 `month × hour` 季节均值；
   - 计算 Spearman；
   - 成对有效样本至少 1000；
   - 相关低于 0.3 不强制连边；
   - 节点不足 1 条边时保留距离图回退边。
4. 距离与相关融合：

   \[
   A^{mix}=\lambda A^{dist}+(1-\lambda)A^{corr}
   \]

   \(\lambda\) 仅在验证集选择。
5. 风向图按第 8.4 节公式构建为有向图。
6. 有向图采用行归一化：

   \[
   \hat{A}_{ij}=\frac{A_{ij}}{\sum_j A_{ij}+\epsilon}
   \]

7. 风向缺失时退回距离图；风速低于 0.5 m/s 时提高固定图权重。
8. 构造 3 节点手算测试：
   - 西风时，西侧站点应向东侧站点传播；
   - 反向边权应接近 0；
   - 无风时接近固定图。
9. 每个图保存节点顺序、参数和哈希。

#### 输出

```text
data/processed/graphs/distance.npz
data/processed/graphs/correlation.npz
data/processed/graphs/dynamic/
data/processed/graphs/graph_manifest.json
outputs/figures/graph_*.png
```

#### 验收

- 形状为 `N × N`；
- 无 NaN 和 Inf；
- 节点顺序哈希与数据一致；
- 固定图无意外孤点；
- 手算方向测试通过；
- 相关图未读取验证和测试数据。

#### 失败处理

动态图失败时不阻塞主线，先使用距离图继续固定图实验。

---

### 步骤 18：生成模型样本

#### 具体操作

1. 按固定站点和特征顺序构造：

   ```text
   X[time,node,feature]
   Y[time,node]
   M[time,node]
   ```

2. 训练集拟合全局逐特征标准化器。
3. 标签使用独立 PM2.5 标准化器。
4. 对每个预测起点建立：

   ```text
   x = t-23 ... t
   y = t+1, t+3, t+6
   ```

5. 输入必须是连续 24 个小时。
6. 输入中原始 PM2.5 观测率低于 80% 的样本丢弃。
7. 若部分站点标签缺失，保留样本并通过 `y_mask` 屏蔽。
8. 若全部标签缺失，丢弃样本。
9. 样本按标签时间归属 split。
10. 保存每个样本的原始索引，支持反查。

#### 输出

```text
data/processed/windows/train.*
data/processed/windows/val.*
data/processed/windows/test.*
data/processed/windows/sample_index.parquet
data/processed/scalers.joblib
data/processed/dataset_manifest.json
```

#### 验收

- 随机抽查至少 100 个样本时间对齐；
- 不跨整日缺档；
- 不跨 split 泄漏标签；
- 标准化器只拟合训练集；
- 逆变换误差小于 \(10^{-6}\)；
- 数据加载器批次形状固定。

#### 失败处理

发现一个标签错位即视为系统性错误，全部窗口重建。

---

### 步骤 19：实现统一评估器

#### 目的

在训练模型前固定评价口径，避免不同模型使用不同实现。

#### 具体操作

1. 定义：
   - 微平均 MAE：全部有效站点时刻共同平均；
   - 宏平均 MAE：先算每站 MAE，再对站点平均；
   - RMSE；
   - Bias：\(\operatorname{mean}(\hat{y}-y)\)；
   - \(R^2\)；
   - SMAPE，\(\epsilon=10^{-6}\)；
   - 相对 Persistence Skill：

     \[
     Skill=1-\frac{MAE_{model}}{MAE_{persistence}}
     \]

2. 所有模型在相同有效标签交集上比较。
3. 负预测：
   - 保存原始值；
   - 展示与业务输出可裁剪到 0；
   - 主论文指标同时记录裁剪前后，默认使用裁剪后现实值；
   - 必须固定口径，不能按模型选择。
4. 分层样本少于 100 时标记 `LOW_SUPPORT`。
5. 块自助法：
   - 按自然日重采样；
   - 2000 次；
   - 种子 42；
   - 百分位 95% 置信区间。
6. 为手工小数组编写指标测试。

#### 输出

```text
src/evaluation/metrics.py
src/evaluation/stratified.py
src/evaluation/significance.py
tests/test_metrics.py
```

#### 验收

- 手算结果一致；
- 掩码行为正确；
- 不同模型使用相同样本数；
- 聚合口径写入指标元数据。

---

### 步骤 20：实现和验收基线

#### 执行顺序

1. Historical Mean；
2. Persistence；
3. Seasonal Persistence：使用前 24 小时同站点值；
4. LightGBM；
5. 共享节点 GRU。

#### 详细规则

- Historical Mean 按目标时刻的站点、小时和周末状态查表；
- Persistence 使用预测起点 \(t\) 的值；该值原始缺失时，该位置不参加该基线比较；
- Seasonal Persistence 使用 \(t+h-24\) 对应值；
- LightGBM 第一版训练三个独立时距模型，避免 `horizon` 联合模型掩盖时距差异；
- 站点 ID 使用类别特征或固定整数编码，编码器只拟合训练集；
- GRU 采用共享节点结构和站点嵌入，不使用空间邻接；
- 所有深度模型使用相同输入特征和标签掩码。

#### 每个模型必须输出

```text
config.yaml
model_file
predictions.parquet
metrics.json
resource_usage.json
run.log
```

资源记录包括参数量、训练时长、峰值显存或内存、批量推理耗时。

#### 验收

- 手工抽样 10 条验证 Historical Mean、Persistence；
- 1 小时 Persistence 通常应是强基线；
- LightGBM 和 GRU 预测无大面积常数或 NaN；
- 模型重新加载后预测一致；
- 全部预测通过统一 schema。

#### 失败处理

若复杂模型明显差于历史均值，依次检查标签位移、逆标准化、损失掩码和时间划分，不直接调参掩盖问题。

---

### 步骤 21：实现固定图模型

#### STGCN 的明确张量契约

```text
输入：[B, F, T=24, N]
静态图：[N, N]
输出：[B, H=3, N]
```

#### 具体实现

1. 左侧因果时间卷积，卷积后裁剪到原时间长度。
2. 第一层时间卷积：

   ```text
   [B,F,24,N] → [B,64,24,N]
   ```

3. 图卷积沿节点维：

   ```text
   einsum("ij,bctj->bcti", A_hat, X)
   ```

4. 第二层时间卷积保持：

   ```text
   [B,64,24,N]
   ```

5. 通道不同时，残差支路用 \(1\times1\) 卷积投影。
6. LayerNorm 明确作用在通道维，或转换维度后对特征维归一化。
7. 堆叠两个 ST-Conv Block。
8. 取最后时间步表示：

   ```text
   [B,64,N]
   ```

9. 对每个节点共享线性输出头：

   ```text
   [B,64,N] → [B,3,N]
   ```

10. 掩码作为额外输入特征，不在隐藏层中直接覆盖数值。
11. 使用 Xavier 初始化线性和卷积权重。
12. checkpoint 保存：
    - `model_state_dict`；
    - 模型配置；
    - optimizer 和 scheduler；
    - epoch；
    - 最佳验证指标；
    - 数据集和图哈希；
    - 特征顺序；
    - 代码提交哈希。

#### 必须通过的测试

- 每层形状测试；
- 因果测试：改变未来填充值不影响当前输出；
- 图乘法手算测试；
- 单批过拟合；
- 图置换影响测试；
- 保存加载一致性；
- CPU 前向测试；
- GPU 可用时 CPU/GPU 误差容限测试。

#### 失败处理

按以下顺序简化定位：

1. 单层 GCN；
2. GCN + 单层 GRU；
3. 一个 ST-Conv Block；
4. 两个完整 Block。

---

### 步骤 22：实现风向动态图模型

#### 张量契约

```text
输入特征：[B,T,N,F]
动态图：[B,T,N,N]
输出：[B,3,N]
```

#### 具体操作

1. 使用源站点风场计算污染传播方向。
2. 按第 17 步固定的 `A[i,j]` 语义转置或组织边权。
3. 对每个输入时刻执行动态 GCN：

   ```text
   [B,N,N] × [B,N,F] → [B,N,C]
   ```

4. 把 24 个时刻的空间表示送入共享节点 GRU。
5. 固定图 GCN-GRU 与动态图 Wind-GCN-GRU 使用相同：
   - 隐藏通道；
   - GRU 层数；
   - 输出头；
   - 优化器；
   - 输入特征。
6. 仅邻接矩阵不同，以保证公平比较。
7. 动态图按批次即时生成或按时间索引缓存，不重复保存每个样本的完整副本。
8. 风向缺失时回退固定图。
9. 低风速时按预设规则提高固定图占比。

#### 验收

- 西风手算案例传播方向正确；
- 静态模式下动态图模型可退化为固定图模型；
- 两个模型参数量差异不超过 5%；
- 动态图无 NaN；
- 一个批次的显存占用在预算内。

#### 失败处理

显存不足时按顺序：

1. 图按时间切片计算；
2. 降低批量；
3. 使用混合精度；
4. 缓存稀疏边列表；
5. 不降低节点完整性。

---

### 步骤 23：训练、调参与故障恢复

#### 调参协议

1. 首先使用种子 42。
2. 固定主选择指标：

   ```text
   validation_score =
   (MAE_1h + MAE_3h + MAE_6h) / 3
   ```

3. 分阶段搜索：
   - 第一阶段：学习率和隐藏通道；
   - 第二阶段：输入窗口和 Dropout；
   - 第三阶段：图参数；
   - 第四阶段：动态图融合系数。
4. 每阶段最多 20 个 trial。
5. 选择前 2 个候选，用 3 个随机种子复验。
6. 若验证分数差异小于 0.5%，优先选择：
   - 参数更少；
   - 推理更快；
   - 结构更简单的模型。
7. 早停：
   - patience 20；
   - 最小改善 `min_delta=0.01 μg/m³` 的标准化等价值；
   - 最多 200 epoch。

#### 故障恢复

- NaN：记录批次索引、输入范围、梯度范数，trial 判失败；
- OOM：批量减半后仅重试一次；
- 中断：从 `last.ckpt` 恢复；
- 超时：trial 标记 `TIMEOUT`；
- 已完成运行目录只读，不覆盖；
- 同一配置重新运行必须生成新 run ID。

#### 每轮记录

```text
train_loss
val_loss
val_mae_each_horizon
learning_rate
gradient_norm
epoch_time
memory_usage
```

#### 验收

- 搜索未超预算；
- 测试集从未参与调参；
- 最佳配置有明确选择理由；
- 三种子复验完成；
- 收敛曲线无持续发散。

---

### 步骤 24：冻结实验矩阵并执行正式测试

#### 具体操作

1. 将最终模型、特征和图配置写入 `configs/final/`。
2. 创建 `EXPERIMENT_FREEZE.md`，记录冻结时间和哈希。
3. 执行必做模型：
   - Historical Mean；
   - Persistence；
   - Seasonal Persistence；
   - LightGBM；
   - GRU；
   - STGCN-distance；
   - GCN-GRU-distance；
   - Wind-GCN-GRU。
4. 执行特征消融 F0 至 F4。
5. 执行图消融 G0 至 G4。
6. 执行缺失处理消融。
7. 每个主模型使用三个种子。
8. 只在全部配置冻结后运行测试集。
9. 生成逐样本预测，不只保存汇总指标。
10. 记录失败运行，不能从汇总中静默删除。

#### 输出

```text
configs/final/
EXPERIMENT_FREEZE.md
outputs/final_experiments/
```

#### 验收

- 实验矩阵无缺项；
- 各模型比较使用相同标签交集；
- 三种子齐全；
- 每个数字可追溯到运行 ID；
- 1、3、6 小时结果均报告；
- 测试结果只生成一次正式版本。

#### 作废条件

仅当发现实现错误或数据泄漏时才能作废正式测试。必须：

1. 记录错误；
2. 标记受影响运行；
3. 提升数据集或实验协议版本；
4. 重跑所有受影响对照；
5. 不只重跑表现不佳的模型。

---

### 步骤 25：分层评估和统计检验

#### 具体操作

1. 输出微平均和宏平均指标。
2. 按以下维度分层：
   - 时距；
   - 站点；
   - 月份和季节；
   - 工作日与周末；
   - PM2.5 浓度；
   - 风速；
   - 输入缺失率。
3. 计算相对 Persistence Skill。
4. 对主模型与最强基线执行逐日块自助法。
5. 同时报告：
   - 绝对 MAE 下降；
   - 相对下降百分比；
   - 95% 置信区间；
   - 样本数。
6. 检查合理性：
   - 一般情况下 1 小时不应系统性差于 6 小时；
   - RMSE 应不小于 MAE；
   - 极少样本分层不下强结论；
   - 宏平均与微平均差异过大时检查困难站点。

#### 输出

```text
outputs/metrics/overall.*
outputs/metrics/by_horizon.*
outputs/metrics/by_station.*
outputs/metrics/by_scene.*
outputs/metrics/significance.*
```

#### 验收

- 所有指标包含单位、样本数和聚合方式；
- 低支持分层已标记；
- 显著性结论与效应大小同时报告；
- 负结果没有被隐藏。

---

### 步骤 26：解释性实验和失败归因

#### 具体操作

1. 遮挡基线值统一使用训练均值，并同时把对应缺失掩码置 0。
2. 遮挡对象：
   - 最近 3 小时；
   - 4 至 12 小时；
   - 13 至 24 小时；
   - 本站历史；
   - 邻站历史；
   - 气象；
   - 其他污染物。
3. 在固定的 1000 个测试样本或全部测试样本上执行，样本索引保存。
4. 对照随机噪声遮挡，防止解释只反映输入尺度。
5. 计算遮挡后的 MAE 增量及置信区间。
6. 失败归因分类：
   - 输入大面积缺失；
   - 突发污染跃升；
   - 风向快速变化；
   - 高污染分布外样本；
   - 单站异常；
   - 固定图关系错误；
   - 模型系统性偏低或偏高。
7. 每个失败类别给出数量和占比。

#### 输出

```text
outputs/explanations/
outputs/cases/failure_taxonomy.csv
```

#### 验收

- 遮挡规则固定且可复现；
- 解释样本不是人工挑选；
- 至少有一次 sanity check；
- 不把相关性解释为因果性。

---

### 步骤 27：按规则选择展示案例

#### 具体操作

1. 从预测明细自动筛选：
   - STGCN 相比 Persistence 改善最大的高污染事件；
   - STGCN 明显失败事件；
   - 动态图相比固定图改善最大的强风事件；
   - 动态图无优势的弱风事件；
   - 高缺失但仍稳定的事件。
2. “明显”定义为逐事件 MAE 差值位于相应分布前 5% 或后 5%。
3. 每类最多保留 3 个，避免只展示个例。
4. 每个案例固定前后至少 24 小时上下文。
5. 导出观测、预测、气象、静态边、动态边和误差摘要。

#### 输出

```text
outputs/cases/index.json
outputs/cases/{case_id}/
```

#### 验收

- 案例全部由规则自动选择；
- 成功和失败案例同时存在；
- 案例可被论文脚本和平台数据导出器直接读取。

---

### 步骤 28：选择并冻结最终模型

#### 选择规则

1. 先满足无数据泄漏、可复现和稳定性要求。
2. 以三种子验证分数选择候选。
3. 测试结果只用于最终报告，不反向调参。
4. 若动态图没有稳定改善，最终部署模型可以选择 STGCN-distance，同时把动态图作为研究结果保留。
5. 若模型差异小于 0.5%，选择更简单、更快者。

#### 冻结操作

1. 决定是否使用训练集加验证集重训：
   - 若重训，epoch 固定为候选模型三种子最佳 epoch 的中位数；
   - 不再早停，不再查看测试集选 epoch。
2. 创建只读发布目录：

   ```text
   releases/model/v1.0.0/
   ```

3. 放入：
   - checkpoint；
   - 完整配置；
   - 插补器；
   - 标准化器；
   - 特征顺序；
   - 站点顺序；
   - 图；
   - 数据集 manifest；
   - 代码提交哈希；
   - 模型卡；
   - 全部 SHA-256。
4. 模型卡说明用途、数据范围、指标、局限、不可用场景和许可。
5. 发布目录不得覆盖；修复必须发布新版本。

#### 验收

- 独立 Python 进程可加载；
- golden sample 预测逐项一致；
- 所有依赖文件均在发布目录或有明确相对引用；
- 哈希校验通过。

---

### 步骤 29：实现并验收离线推理

#### 目的

确保最终模型能脱离训练脚本，对新数据或历史回放稳定生成预测。

#### 推理流程

1. 读取冻结模型版本。
2. 校验输入 schema。
3. 检查固定站点集合。
4. 检查至少 24 个连续小时历史。
5. 按训练规则执行因果插补。
6. 加载而非重新拟合预处理器。
7. 生成特征和图。
8. 执行模型。
9. 逆标准化。
10. 同时保存原始预测和非负裁剪预测。
11. 输出质量状态：
    - 输入缺失率；
    - 缺失站点；
    - 是否发生回退；
    - 模型版本；
    - 数据版本。

#### 两种模式

```text
single-origin：单个预测起点
backtest：给定历史时间范围批量回放
```

#### 建议命令

```powershell
python -m src.inference.predict `
  --model releases/model/v1.0.0 `
  --input data/new/observations.parquet `
  --origin "2026-01-01T08:00:00+08:00" `
  --output outputs/inference/
```

#### 验收

- golden sample 与训练阶段导出预测误差小于 \(10^{-6}\)；
- 推理过程未调用 `fit`；
- 缺少关键站点或不足 24 小时时明确失败；
- 失败返回非零退出码；
- 不生成半成品结果；
- 推理日志可追踪。

---

### 步骤 30：生成论文准备包

本步骤不撰写论文，只准备后续论文方案和写作必需的唯一可信材料。

#### 具体操作

1. 固定论文使用的运行 ID。
2. 自动导出：
   - 数据集统计；
   - 模型参数和资源；
   - 主结果；
   - 分时距结果；
   - 消融结果；
   - 分层结果；
   - 显著性结果。
3. 自动生成：
   - 站点地图；
   - 数据质量图；
   - 技术路线图；
   - 模型对比图；
   - 误差地图；
   - 消融图；
   - 成功和失败案例图。
4. 每张表和图生成 `metadata.json`：
   - 来源运行 ID；
   - 脚本；
   - 数据版本；
   - 生成时间；
   - 单位；
   - 过滤条件。
5. 建立参考文献库，但论文结构和具体写作后续另行规划。

#### 输出

```text
outputs/paper_package/
├─ tables/
├─ figures/
├─ metadata/
└─ run_index.json
```

#### 验收

- 不手工抄写任何实验数字；
- 删除包后可以一条命令重建；
- 所有图表均可追溯到运行 ID；
- 单位、样本数和模型名称一致。

---

### 步骤 31：生成展示平台准备包

本步骤不开发平台，只生成后续任意平台技术栈都可消费的稳定数据产品。

#### 具体操作

1. 为平台 schema 设置版本，例如 `1.0.0`。
2. 固定字段类型、单位、时区、主键和可空性。
3. 导出：
   - `stations.geojson`；
   - `observations.parquet`；
   - `forecasts.parquet`；
   - `station_metrics.json`；
   - `graph_static.geojson`；
   - 典型案例动态边；
   - `cases/index.json`；
   - `metadata.json`。
4. 动态图不导出全量三年每小时矩阵，只导出：
   - 选定案例；
   - 平台默认演示时间；
   - 必要时按月分片的稀疏边表。
5. Parquet 使用 `zstd` 压缩，并按年份或月份分区。
6. GeoJSON 坐标统一为 WGS84。
7. 每个文件记录 SHA-256。
8. 建立外键检查：
   - 所有预测站点都存在于站点表；
   - `target_time = origin_time + horizon`；
   - 指标模型名存在于模型清单；
   - 案例引用文件存在。
9. 添加数据来源和许可文本，供平台页脚使用。
10. 编写一个最小读取测试，不设计具体页面。

#### 输出

```text
outputs/platform_package/v1.0.0/
```

#### 验收

- schema 校验通过；
- 主键和外键通过；
- 时间统一为带时区 ISO 8601；
- 地图坐标可正确显示；
- 完整性哈希通过；
- 包内不依赖训练代码。

---

### 步骤 32：复现、归档和最终交付

#### 具体操作

1. 在全新虚拟环境安装项目。
2. 运行七天 smoke 流程。
3. 加载冻结模型并执行 golden sample。
4. 随机选择一个基线从数据构建到评估完整复跑。
5. 检查所有清单哈希。
6. 创建最终 Git 标签：

   ```text
   project-v1.0.0
   ```

7. 编写根目录 README：
   - 项目简介；
   - 环境安装；
   - 数据获取；
   - 小样本运行；
   - 正式训练；
   - 离线推理；
   - 结果生成；
   - 故障排查；
   - 数据与代码许可。
8. 创建交付清单和归档清单。
9. 原始数据若不能重新分发，只交付 manifest、下载脚本和校验信息。
10. 备份：
    - 代码仓库；
    - 最终模型；
    - 数据 manifest；
    - 论文准备包；
    - 平台准备包。

#### 输出

```text
README.md
RELEASE_NOTES.md
CHECKSUMS.sha256
docs/reproduction_log.md
releases/
```

#### 最终验收

第三方在全新环境中仅依据 README，能够：

1. 安装项目；
2. 下载或准备样例数据；
3. 运行 smoke 流程；
4. 生成一个基线结果；
5. 加载最终模型；
6. 完成一次离线预测；
7. 读取论文和平台准备包。

任何一步需要口头补充才能完成，都视为交付未通过。

---

## 26. 上游变化后的重跑规则

为避免只重跑部分实验导致结果不可比，按下列规则处理：

- 数据源或原始文件变化：步骤 8 以后全部重跑；
- 研究期或站点变化：步骤 10 以后全部重跑；
- 时间划分变化：步骤 14 以后全部重跑；
- 清洗或插补变化：步骤 15 以后全部重跑；
- 特征变化：步骤 17 以后全部重跑；
- 图构造变化：所有图模型重跑，非图基线可保留；
- 指标实现变化：全部模型重新评估，必要时无需重训；
- 模型代码变化：受影响模型及其消融重跑；
- 论文或平台导出代码变化：只重建相应准备包；
- 发现数据泄漏：所有下游运行立即标记无效，不得引用旧结果。

---

## 27. 推荐实际推进节奏

### 第一阶段：可靠数据

完成步骤 0 至 15。阶段终点不是“数据下载完”，而是：

- 来源通过准入；
- 研究期与站点冻结；
- 原始和清洗后审计通过；
- 无数据泄漏；
- 数据集 manifest 可追溯。

### 第二阶段：可靠基线

完成步骤 16 至 20。阶段终点是：

- 特征、图和样本契约固定；
- 统一评估器经过测试；
- 所有基线结果合理；
- 训练与推理接口打通。

### 第三阶段：研究模型

完成步骤 21 至 27。阶段终点是：

- 固定图和动态图实现正确；
- 正式实验矩阵完成；
- 分层、统计和解释结果完整；
- 成功与失败案例均已冻结。

### 第四阶段：成果冻结

完成步骤 28 至 32。阶段终点是：

- 最终模型可独立推理；
- 论文准备包可自动生成；
- 平台准备包通过 schema 和完整性检查；
- 全新环境复现通过。

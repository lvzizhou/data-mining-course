# 面向智慧城市的时空图数据挖掘

本项目是数据挖掘课程小组合作仓库，聚焦城市交通与空气质量的时空图建模，目标覆盖预测、异常检测和可解释分析。

## 研究问题

- 交通流量/速度的短期多步预测
- PM2.5、AQI 等空气质量指标的时空预测
- 交通与空气质量的联合异常检测
- 对模型结果进行节点、时间和特征级解释

## 仓库结构

`src/` 算法与数据处理代码；`notebooks/` 探索性分析；`data/` 仅存数据说明与小型样例；`models/` 模型配置或产物索引；`reports/` 实验报告；`docs/` 计划与会议记录；`tests/` 自动化测试。

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pytest
```

从 [MEMORY.md](MEMORY.md) 了解当前决策，从 [docs/team-workplan.md](docs/team-workplan.md) 领取任务。协作规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 数据原则

原始数据不提交到 Git；请在 `data/README.md` 记录来源、许可证、下载方式、时间范围、空间粒度和预处理步骤。所有实验必须固定随机种子并保存配置、指标和数据版本。

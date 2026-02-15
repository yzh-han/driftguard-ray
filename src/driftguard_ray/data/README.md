# Data Module README

本模块是轻量化的 Data Service 设计，实现最小化的数据读取与 RPC 服务。

## 目录结构
- `domain_dataset.py`：按域读取样本；只读取 `_meta.json`，不扫描目录。
- `service.py`：XML-RPC 服务入口；仅负责调度与传输。
- `drift.py`：可选漂移策略封装。
- `drift_events_generator.py`：生成漂移事件配置。
- `data_service_design.md`：设计说明与约束。

## 元数据
元数据文件位于 `datasets/<dataset_name>/_meta.json`，由预处理脚本生成。

## 基本流程
1) 预处理生成 `_meta.json`
2) `DomainDataset` 读取元数据
3) `DataService` 通过 RPC 提供按域采样

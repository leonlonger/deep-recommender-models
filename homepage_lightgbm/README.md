# Homepage LightGBM

首页推荐 CTR LightGBM 基线模型，复用 `homepage_recommendation` 目录生成的预处理 Parquet 数据。

## 文件说明

- `main.py`: 加载预处理数据、训练 LightGBM、评估并导出产物。
- `model.py`: 特征定义、类别编码、LightGBM Dataset 构造和二分类指标。
- `config.yaml`: 预处理数据路径、LightGBM 参数和训练参数。
- `requirements.txt`: 运行依赖。

## 安装依赖

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 输入数据

这个目录不直接 dump BigQuery，也不直接处理原始大 Parquet。先在 `homepage_recommendation` 里生成预处理数据：

```bash
cd ../homepage_recommendation
.venv/bin/python main.py \
  --preprocess-output /mnt/disk/datasets/homepage_training_preprocessed \
  --overwrite-preprocessed
```

LightGBM 默认读取：

- `/mnt/disk/datasets/homepage_training_preprocessed/train.parquet`
- `/mnt/disk/datasets/homepage_training_preprocessed/validation.parquet`
- `/mnt/disk/datasets/homepage_training_preprocessed/metadata.json`

`metadata.json` 里的 `feature_spec` 是默认特征和 label 来源。如果需要临时覆盖 label，可以使用 `--label-column`。

## 检查数据

```bash
.venv/bin/python main.py --inspect
```

## 训练

```bash
.venv/bin/python main.py
```

小样本 smoke test：

```bash
.venv/bin/python main.py --limit 5000 --num-boost-round 20 --skip-export
```

如果要改变模型输出目录：

```bash
.venv/bin/python main.py --output-dir /tmp/homepage_lightgbm
```

## 训练产物

默认写到 `models/homepage_lightgbm/runs/<run_id>/`：

- `model.txt`: LightGBM text model。
- `feature_encoder.json`: 训练集类别值到整数编码的映射。
- `feature_importance.csv`: split/gain 特征重要性。
- `training_metadata.json`: 特征、配置、best iteration 和评估指标。
- `reproducibility/`: source/effective config、CLI 参数、环境、pip freeze、git status/diff 和数据路径信息。

评估指标包括 AUC、binary logloss、PR AUC、PCOC、accuracy、precision 和 recall。

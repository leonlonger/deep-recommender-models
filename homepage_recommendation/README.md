# Homepage Recommendation

首页推荐 DNN 排序模型，基于 TensorFlow/Keras 和本地 Parquet 训练数据。

BigQuery dump 源：

- Project: `phia-prod-416420`
- Table: `phia-prod-416420.ml.recommendation_metadata_training_examples`

## 文件说明

- `main.py`: 本地 Parquet/CSV 数据读取、字段推断、训练、评估和模型导出入口。
- `model.py`: TensorFlow DNN 排序模型与 `tf.data` 数据集构造。
- `config.yaml`: 数据源、label、特征、模型和训练参数。
- `scripts/dump_bigquery_data.py`: 将 BigQuery 训练表导出为本地 Parquet。
- `requirements.txt`: 运行依赖。

## 安装依赖

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

如果要使用 GPU 训练，安装额外 CUDA/cuDNN wheel：

```bash
.venv/bin/python -m pip install -r requirements-gpu.txt
```

如果本机没有 `pip`，先在项目目录外创建一个支持 TensorFlow 的 Python 环境，再安装依赖。

## Dump 训练数据

默认训练数据路径是：

- `/mnt/disk/datasets/homepage_training_examples.parquet`

这个文件不会提交到 git。第一次训练前，先从 BigQuery dump 到本地：

```bash
.venv/bin/python scripts/dump_bigquery_data.py
```

脚本会读取 `config.yaml` 里的 `data.bigquery` 配置和 `selected_columns`，默认不加 `LIMIT`，因此会导出当前表里的完整训练数据。小样本调试可以显式限制行数：

```bash
.venv/bin/python scripts/dump_bigquery_data.py --limit 5000 --output /tmp/homepage_sample.parquet
```

脚本默认把 BigQuery Storage API 限制为单 stream、单队列，并先写 `.tmp` 文件，完整写完后才替换正式 Parquet。小内存机器上不要随意调大 `--max-stream-count`。

如果要先检查 BigQuery 查询预计扫描量：

```bash
.venv/bin/python scripts/dump_bigquery_data.py --dry-run
```

## GCP 认证

```bash
gcloud auth login
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/bigquery,https://www.googleapis.com/auth/userinfo.email,openid
gcloud config set project phia-prod-416420
```

运行账号需要有读取 BigQuery 表的权限。`gcloud auth login` 给 `bq` CLI 使用，`gcloud auth application-default login` 给 Python BigQuery client 和 dump 脚本使用。

## 先检查表结构

```bash
.venv/bin/python main.py --inspect
```

`--inspect` 会默认从本地 Parquet 读取 `config.yaml` 里的 `data.inspect_row_limit` 行样本，打印字段类型，并尝试推断：

- label: 优先使用 `label.column`，为空时尝试 `label`、`clicked`、`is_click`、`conversion` 等常见字段。
- 数值特征: 数值类型列。
- 类别特征: string/bool/object 列，以及 `*_id`、`*_uuid` 等 ID 风格列。

## 配置 label 和特征

至少需要确认 `config.yaml` 里的 `label.column`。如果目标字段不是 0/1，而是点击次数、评分、停留时长等连续值，需要设置：

```yaml
label:
  column: your_target_column
  positive_threshold: 0
```

如果自动推断不符合业务语义，可以显式指定：

```yaml
features:
  auto_infer: true
  numeric_columns: [price, score]
  categorical_columns: [user_id, item_id, category_id]
  exclude_columns: [event_time, request_id]
```

## 训练

```bash
.venv/bin/python main.py
```

训练默认只读取本地 `data.path`，不会访问 BigQuery。
训练时会默认写 TensorBoard event logs 到 `models/homepage_dnn/tensorboard/<UTC时间戳>/`，终端里也会打印实际的 `TensorBoard log dir`。

本地 Parquet 训练默认启用 streaming，会先读取 `data.inspect_row_limit` 行样本推断 schema，然后用 `data.streaming_batch_rows` 分块扫描 Parquet、统计 normalizer/class weight、再分块喂给 TensorFlow。这样不会把 20GB+ Parquet 一次性读进 pandas。可以在 `config.yaml` 调整每块行数：

```yaml
data:
  streaming: true
  streaming_batch_rows: 100000
```

## 先离线预处理再训练

如果不想每次训练都重复做 label 清洗、类型转换、train/validation 切分和统计，可以先把原始 Parquet 处理成一个新的训练数据集：

```bash
.venv/bin/python main.py \
  --preprocess-output /mnt/disk/datasets/homepage_training_preprocessed \
  --overwrite-preprocessed
```

这个目录会包含：

- `train.parquet`
- `validation.parquet`
- `metadata.json`
- `training_config.yaml`

之后直接用生成的配置训练：

```bash
.venv/bin/python main.py --config /mnt/disk/datasets/homepage_training_preprocessed/training_config.yaml
```

小样本验证可以加 `--limit`：

```bash
.venv/bin/python main.py \
  --limit 5000 \
  --preprocess-output /tmp/homepage_training_preprocessed_sample \
  --overwrite-preprocessed
.venv/bin/python main.py \
  --config /tmp/homepage_training_preprocessed_sample/training_config.yaml \
  --epochs 1 \
  --skip-export
```

如果要临时回到旧的 pandas 全量读取路径：

```bash
.venv/bin/python main.py --no-streaming
```

GPU 训练使用封装脚本，它会自动把 `.venv` 中的 CUDA/cuDNN 动态库加入 `LD_LIBRARY_PATH`：

```bash
scripts/train_gpu.sh
```

小样本 smoke test 可以跳过模型导出：

```bash
.venv/bin/python main.py --limit 5000 --epochs 1 --skip-export
scripts/train_gpu.sh --limit 5000 --epochs 1 --skip-export
```

如果想把 TensorBoard 日志写到别的位置：

```bash
.venv/bin/python main.py --tensorboard-log-dir /tmp/homepage_tensorboard
```

临时关闭 TensorBoard：

```bash
.venv/bin/python main.py --disable-tensorboard
```

训练完成后会输出：

- `models/homepage_dnn/best.keras`
- `models/homepage_dnn/final.keras`
- `models/homepage_dnn/saved_model`
- `models/homepage_dnn/training_metadata.json`

`training_metadata.json` 的 `history` 和 `evaluation` 会保留 AUC、PCOC、accuracy、precision、recall 等指标；启用验证集时 `history` 里也会包含 `val_pcoc`。

如果当前工作区磁盘空间不足，可以把模型写到 `/tmp`：

```bash
.venv/bin/python main.py --output-dir /tmp/homepage_dnn
```

## 本地查看 TensorBoard

如果 `.venv` 是之前创建的，先更新依赖：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

先启动一次训练，等终端出现 `TensorBoard log dir: ...` 后，另开一个终端运行：

```bash
.venv/bin/tensorboard --logdir models/homepage_dnn/tensorboard --host 127.0.0.1 --port 6006
```

然后在浏览器打开：

```text
http://127.0.0.1:6006
```

如果训练时用了自定义日志目录，例如 `--tensorboard-log-dir /tmp/homepage_tensorboard`，启动 TensorBoard 时也把 `--logdir` 换成同一个目录。

## 本地 CSV 调试

也可以把 `data.source` 改成 `csv`：

```yaml
data:
  source: csv
  path: /path/to/sample.csv
```

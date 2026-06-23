# Homepage Recommendation

首页推荐 DNN 排序模型，基于 TensorFlow/Keras 和 BigQuery 训练数据。

默认数据源：

- Project: `phia-prod-416420`
- Table: `phia-prod-416420.ml.recommendation_metadata_training_examples`

## 文件说明

- `main.py`: BigQuery/CSV 数据读取、字段推断、训练、评估和模型导出入口。
- `model.py`: TensorFlow DNN 排序模型与 `tf.data` 数据集构造。
- `config.yaml`: 数据源、label、特征、模型和训练参数。
- `requirements.txt`: 运行依赖。

## 安装依赖

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

如果本机没有 `pip`，先在项目目录外创建一个支持 TensorFlow 的 Python 环境，再安装依赖。

## GCP 认证

```bash
gcloud auth login
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/bigquery,https://www.googleapis.com/auth/userinfo.email,openid
gcloud config set project phia-prod-416420
```

运行账号需要有读取 BigQuery 表的权限。`gcloud auth login` 给 `bq` CLI 使用，`gcloud auth application-default login` 给 Python BigQuery client 使用。

## 先检查表结构

```bash
.venv/bin/python main.py --inspect
```

`--inspect` 会默认读取 `config.yaml` 里的 `data.inspect_row_limit` 行样本，打印字段类型，并尝试推断：

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

小样本 smoke test 可以跳过模型导出：

```bash
.venv/bin/python main.py --limit 5000 --epochs 1 --skip-export
```

训练完成后会输出：

- `models/homepage_dnn/best.keras`
- `models/homepage_dnn/final.keras`
- `models/homepage_dnn/saved_model`
- `models/homepage_dnn/training_metadata.json`

如果当前工作区磁盘空间不足，可以把模型写到 `/tmp`：

```bash
.venv/bin/python main.py --output-dir /tmp/homepage_dnn
```

## 本地 CSV 调试

也可以把 `data.source` 改成 `csv`：

```yaml
data:
  source: csv
  path: /path/to/sample.csv
```

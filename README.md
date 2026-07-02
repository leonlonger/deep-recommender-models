# Deep Learning Recommendation Models

这个仓库用于存放深度学习推荐算法模型代码。

根目录只保留 README、依赖说明等基础文件；每个推荐场景或模型单独放在一个目录中，目录内部维护自己的 `main.py`、`model.py`、`config.yaml` 和说明文档。

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── homepage_recommendation/    # 首页推荐场景
├── homepage_lightgbm/           # 首页推荐 LightGBM CTR 基线
├── deepfm/                     # DeepFM 排序模型
├── two_tower_recall/           # 双塔召回模型
└── din/                        # DIN 排序模型
```

## 现有目录

- `homepage_recommendation`: 首页推荐模型入口，适合组合召回、排序、重排等逻辑。
- `homepage_lightgbm`: 复用首页推荐预处理数据的 LightGBM CTR 基线。
- `deepfm`: DeepFM 模型示例。
- `two_tower_recall`: 用户塔和物品塔召回模型示例。
- `din`: Deep Interest Network 模型示例。

## 新增模型目录约定

新增模型时，在根目录创建一个新的文件夹：

```text
new_model_name/
├── README.md
├── config.yaml
├── main.py
└── model.py
```

每个模型目录应尽量独立，方便单独运行、调试和替换实验配置。

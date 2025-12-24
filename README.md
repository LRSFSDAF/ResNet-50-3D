# ResNet-50 FreiHAND 3D 手部关键点回归（PyTorch）

本项目使用 `torchvision.models.resnet50` 作为骨干网络，回归 FreiHAND 数据集的 21 个手部关键点 3D 坐标（相对根关节/第 9 号关节归一化）。训练、验证与测试流程均在 [main.py](main.py) 中实现。

## 1. 目录结构（必须）

脚本默认从项目根目录下读取数据，并在根目录输出结果与权重。请确保至少具备以下目录与文件：

```
ResNet-50-3D/
├─ main.py
├─ data/
│  └─ FreiHAND_pub_v2/
│     ├─ training_xyz.json
│     ├─ training_K.json
│     └─ training/
│        └─ rgb/
│           ├─ 00000000.jpg
│           ├─ 00000001.jpg
│           └─ ...
├─ results/                 # 可为空；程序会自动创建
└─ (训练结束后生成) resnet_freihand_full.pth
```

说明：
- `DATASET_ROOT` 在 [main.py](main.py) 顶部配置，默认指向 `data/FreiHAND_pub_v2`。
- 训练图片读取路径：`data/FreiHAND_pub_v2/training/rgb/%08d.jpg`。
- 标注读取文件：
  - `training_xyz.json`：每张图对应 21x3 的 3D 关键点坐标
  - `training_K.json`：每张图对应 3x3 相机内参矩阵

> 你工作区里已有的 `data/annotations/`、`OpenDataLab___FreiHAND/` 等目录不影响本脚本运行；`main.py` 当前不会读取它们。

## 2. 环境依赖

建议：Python 3.9+，PyTorch 2.x（CPU 或 CUDA 均可）。

最小依赖（由 [main.py](main.py) 的 import 决定）：
- `torch`
- `torchvision`
- `Pillow`
- `matplotlib`
- `numpy`

安装示例（二选一）：

### 2.1 pip

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pillow matplotlib numpy
```

> 上面 `cu121` 仅为示例；请按你的 CUDA 版本选择对应命令。若仅 CPU，改用 PyTorch 官网给出的 CPU 安装命令。

### 2.2 conda

```
conda create -n freihand python=3.10 -y
conda activate freihand
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install pillow matplotlib numpy
```

## 3. 数据准备

1. 下载 FreiHAND（公开版本）数据并解压。
2. 将解压后的内容整理到本项目的 `data/FreiHAND_pub_v2/` 下，至少包含：
   - `training_xyz.json`
   - `training_K.json`
   - `training/rgb/*.jpg`

常见问题：
- 若启动时报错 `找不到标注文件: ...training_xyz.json`，说明路径不对或文件缺失。
- 若个别图片读取失败，[main.py](main.py) 会用一张空白图替代（不建议长期如此，最好修复图片文件）。

## 4. 训练与验证（运行方式）

在项目根目录执行：

```
python main.py
```

脚本会自动：
- 检测设备：`cuda`（有 GPU）或 `cpu`
- 将数据随机划分为：训练 60% / 验证 20% / 测试 20%（`random_split`）
- 每个 epoch 结束时，抽取 1 个测试样本进行可视化并保存到 `results/`

### 4.1 关键超参数修改

当前脚本没有命令行参数，需直接修改 [main.py](main.py) 顶部常量：
- `BATCH_SIZE`（默认 64）
- `EPOCHS`（默认 15）
- `LEARNING_RATE`（默认 1e-4）
- `DATASET_ROOT`（默认 `data/FreiHAND_pub_v2`）

### 4.2 Windows/性能相关建议

- Windows 下 `num_workers=8`、`persistent_workers=True` 在某些机器上可能导致卡死或报错；可在 [main.py](main.py) 中把 `num_workers` 降到 `0~4`。
- 显存不足：降低 `BATCH_SIZE`。

## 5. 输出结果说明

运行结束后会生成：
- `results/epoch_{n}_test_0.png`：每个 epoch 的测试样本可视化（绿色=GT，红色=Pred）
- `results/training_metrics.png`：训练/验证损失与验证准确率曲线
- `resnet_freihand_full.pth`：训练好的模型参数（`state_dict`）

## 6. 指标含义（与脚本一致）

- Loss：`MSELoss(pred, gt)`，其中 `pred/gt` 为**相对根关节（第 9 号关节）**的 3D 坐标。
- MPJPE：$\|pred-gt\|$ 的平均值（按关节欧氏距离）。
- Accuracy：在阈值 `threshold=0.05`（单位与标注一致）内的关节比例。

## 7. 模型结构简述

- Backbone：ResNet-50（ImageNet 预训练权重 `IMAGENET1K_V1`）去掉最后分类层。
- Head：`2048 -> 1024 -> 63`（21x3），输出 reshape 为 `(B, 21, 3)`。

---

如果你希望 README 里再补充“加载已有权重做推理/只跑验证”的示例代码，我也可以基于当前 [main.py](main.py) 再加一个最小推理脚本或在 `main.py` 里加 `--eval` 入口。
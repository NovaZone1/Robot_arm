# 视觉数据采集

`scripts/collect_vision_dataset.py` 用于在机械臂已手动移动到观察位后，实时查看 D435 画面并采集新的物品、盒标和干扰样本。脚本不会移动机械臂或底盘。

运行前先停止 Dashboard 真机栈（尤其是 `camera_server_node`），因为 D435 同一时刻只能由一个进程打开。采集脚本会自动启用 D435 的自动曝光和自动白平衡；这只影响本脚本进程，不会改动真机栈的固定识别参数。

```bash
cd /home/nvidia/auto/Robot_arm
.venv/bin/python source/scripts/collect_vision_dataset.py \
  --mode object --class-id orange_bottle
```

三种模式：

- `object`：桌面上的抓取物体。
- `box_label`：透明盒后侧的纸质标识。
- `negative`：阴影、瓶盖、反光、空桌面等非目标干扰。

窗口按键：`Space` 保存一张、`R` 开关每 `0.4s` 连续采集、`Q` 或 `Esc` 退出。输出默认保存在 `source/data/vision_dataset/`，并追加 `manifest.jsonl`，记录每张图的类别、模式、时间和相对路径。

建议每个实际物体/盒标先采 30--50 张，改变物体位置、轻微角度和光照；对红色瓶盖、深阴影、透明盒反光等易误检区域采 15--25 张 `negative`。连续采集时请缓慢改变画面，避免大量近乎重复的图像。

## 盒标 YOLO 标注

盒标检测模型需要标签纸的边界框。采集结束后运行：

```bash
.venv/bin/python source/scripts/annotate_box_labels.py
```

程序会先自动给出白色标签纸候选框。框正确时按 `Space` 保存并进入下一张；不正确时用鼠标拖出新框后再按 `Space`。`N` 表示该帧无有效标签，`A`/左方向键回看上一张已保存框，`D`/右方向键只前进查看、不写文件，`Q` 暂停；再次运行会自动跳过已保存图片。标注以 YOLO 格式保存到 `source/data/vision_dataset/box_labels/labels/`。

只查看、不修改已有框：

```bash
.venv/bin/python source/scripts/review_box_labels.py
```

在只读浏览器中使用 `A`/左方向键上一张，`D`/右方向键/空格下一张，`Q` 退出。

完成全部审核后，构建可复现的六类 YOLO 训练集：

```bash
.venv/bin/python source/scripts/prepare_box_label_yolo_dataset.py
```

脚本会忽略空标注帧，按类别固定随机划分 `80%` 训练、`20%` 验证，并写入 `source/data/box_label_yolo6/`。

训练六类盒标检测模型：

```bash
.venv/bin/python source/scripts/train_box_label_yolo.py
```

若训练意外中断，使用 `--resume` 从 `models/box_label_yolo6/train/weights/last.pt` 继续。

物体图使用同一个标注器，但必须显式选择物体模式：

```bash
.venv/bin/python source/scripts/annotate_box_labels.py \
  --mode object \
  --images-root source/data/vision_dataset/objects
```

目录名决定该图要标注的目标类别；即使画面中有其他物体，也只保留当前目录对应目标的一个框。

由于物体图可包含其他未标注的比赛物体，当前先从已标目标框裁剪出六类分类训练集，避免把同类物体错误当成检测背景：

```bash
.venv/bin/python source/scripts/prepare_object_crop_classifier_dataset.py
.venv/bin/python source/scripts/train_object_crop_classifier.py
```

若要训练全画面 YOLO 检测器，则每张图内所有可见的六类物体都必须分别标框，不能只标目录对应的目标。

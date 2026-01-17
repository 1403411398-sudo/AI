

# 图像检索系统（Vision Transformer）

该项目基于 NumPy 手动复现了 Vision Transformer（DINOv2）的前向传播过程，包括 **Multi-Head Attention、LayerNorm、GELU、Patch Embedding** 等核心组件，并基于此构建了一个简单的图像检索系统。

## 1. 项目简介

本项目通过仅使用 **NumPy** 库，实现了 Vision Transformer（DINOv2）的前向传播过程。通过这一复现，结合图像特征提取与相似度计算，构建了一个图像检索系统。

## 2. 环境依赖

运行本项目需要以下 Python 库：

```bash
pip install numpy scipy pillow requests streamlit tqdm
```

### 核心库说明：

* **numpy**: 矩阵运算核心，用于所有张量计算。
* **scipy**: 用于图像缩放插值（`zoom`）和精确的 GELU 计算（`erf`）。
* **pillow**: 用于图像读取与处理。
* **streamlit**: 用于构建 Web 可视化界面。

## 3. 目录结构

```
dinov2_numpy.py      # 手写ViT模型
preprocess_image.py  # 图像预处理
debug.py             # 模型精度验证脚本
data processing.py     # 数据集下载脚本
researching.py         # 检索系统
steamlit.py               # 可视化界面
vit-dinov2-base.npz  # 模型权重文件
data.csv             # 图片URL数据源
gallery_images/      # 图片库文件夹
gallery_features.npy # 特征索引文件
demo_data/           # 测试用例图片
```

## 4. 运行指南

### [Step 1] 下载图片数据

运行脚本，从 `data.csv` 下载图片并构建图库。图片 URL 存储在 `data.csv` 中。

```bash
python data processing.py
```

### [Step 2] 验证模型精度

验证模型的输出精度与预期是否一致。

```bash
python debug.py
```

### [Step 3] 构建索引并测试

运行此脚本，从图库中提取特征并生成索引。

```bash
python researching.py
```

### [Step 4] 启动可视化界面

通过 Streamlit 启动可视化界面，进行图像检索操作。

```bash
streamlit run streamlit.py
```

## 5. 示例

1. 在浏览器中打开 Streamlit 可视化界面，上传一张图片。
2. 点击“开始搜索”，系统将返回与输入图片相似的前十个结果。

## 6. 注意事项

* 确保在执行步骤之前，**已下载并存放图片**。
* 本项目使用 DINOv2 预训练权重 `vit-dinov2-base.npz`，确保文件路径正确。
* 需要有 **稳定的网络连接** 来下载图片数据。

---

## 7. 项目展示

此系统可用于 **图像检索任务**，通过 Vision Transformer 提取图像特征并进行比对，适用于大规模图像检索和相似图像查询。

---

这样，你就可以直接将这个内容粘贴到 GitHub 的 `README.md` 文件里，完全可以直接使用。希望对你有帮助！

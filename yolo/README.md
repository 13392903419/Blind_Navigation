# YOLOv8 盲道检测模型

## 📊 模型信息

这是一个经过完整训练的YOLOv8盲道检测模型，用于实时识别视频中的盲道及其方向变化。

### 文件说明

- **best.pt** - 训练好的模型权重文件（开箱即用）
- **results.png** - 训练过程的综合结果可视化
- **results.csv** - 训练指标的详细数据
- **confusion_matrix.png** - 混淆矩阵
- **confusion_matrix_normalized.png** - 归一化混淆矩阵
- **P_curve.png** - 精确率曲线
- **R_curve.png** - 召回率曲线
- **F1_curve.png** - F1分数曲线
- **PR_curve.png** - 精确率-召回率曲线
- **labels.jpg** - 数据集标签分布
- **labels_correlogram.jpg** - 标签相关性分析
- **args.yaml** - 训练参数配置

## 🎯 模型性能

该模型在自定义盲道数据集上进行训练，能够：
- 准确识别盲道的位置
- 检测盲道的方向变化（左转/右转）
- 实时处理视频流
- 在各种光照条件下保持稳定性能

## 💡 使用方法

在 `config.py` 中配置模型路径：

```python
MODEL_WEIGHTS = 'yolo/best.pt'
```

系统会自动加载该模型进行盲道检测。

## 📝 训练信息

本模型基于ultralytics的YOLOv8框架训练，使用了：
- 自定义收集的盲道数据集
- 团队成员人工标注的高质量标签
- 优化的训练参数（见 `args.yaml`）

## 🙏 致谢

感谢所有参与数据收集和标注的团队成员：
- Chen Xingyu
- Wang Youyi
- Liu Yiheng
- Cai Yuxin
- Zhang Chenshu


# Day 17: TensorBoard 可视化

**日期：** 2026-06-10

## 为什么需要 TensorBoard

之前训练 CNN，判断「学得怎么样」只能看每个 epoch 末尾打印的一个 loss 数字。

> 相当于开车只看后视镜里上一秒的速度——不知道一路上是加速还是减速，不知道什么时候该踩刹车。

TensorBoard 把训练过程中「看不见的东西」变成「能看到的图」。

类比：ICU 监护仪——不是只告诉你「现在体温 38.5°C」，而是画一整条体温变化曲线。

## 核心概念：Callback（回调/钩子函数）

`model.fit()` 在每个 batch 结束、每个 epoch 结束时都有一个「通知点」。在这些点上挂一个记录器，把数据写到文件里，这就是 Callback。

### 三个关键参数

| 参数 | 类比 | 你的值 |
|---|---|---|
| `log_dir` | 「病历放哪个抽屉」— 日志存哪 | `'logs'` |
| `update_freq` | 「多久测一次」— 记录频率 | `'epoch'` |
| `histogram_freq` | 「要不要拍 CT」— 权重分布快照 | `1`（每 epoch 都拍） |

另外两个参数：
- `write_graph`：文档标注 "Not supported at this time"，实际不生效
- `write_images`：权重可视化，对深网络意义不大，会让日志变大

### 什么是 Histogram（直方图）

把一层所有权重按数值大小分桶，看它们落在哪个范围。

| 直方图长这样 | 说明什么 |
|---|---|
| 集中在 0 附近，像一座山 | ✅ 正常 |
| 大部分是 0 | ⚠️ 可能 ReLU 死神经元 |
| 散得很开 | 正常，网络在学不同特征 |
| 全挤在很小范围 | ⚠️ 可能梯度消失 |

## 实战过程

### 坑：Windows 中文路径

TensorFlow C++ 底层遇到中文「学习」会编码成乱码，导致 `tf.io.gfile.isdir()` 返回 False，TensorBoard 读不到数据。

**修法**：用 `mklink /J` 创建目录连接点（Junction）：

```cmd
mklink /J F:\tflearn F:\学习tensorflow
```

然后用 `--logdir=F:/tflearn/logs` 启动，纯 ASCII 路径绕过了编码问题。

> 💡 `mklink` = make link（创建链接），`/J` = Junction（目录连接点）。`F:\tflearn` 和 `F:\学习tensorflow` 指向完全相同的文件。

### 启动 TensorBoard

```bash
tensorboard --logdir=F:/tflearn/logs --port=6008 --bind_all
```

浏览器打开 `http://localhost:6008`

## 三个标签页

### SCALARS（标量曲线）— 最重要

看 loss/accuracy 曲线：
- **向下走** = loss 在降，模型在学 ✅
- **两条线分岔**（train vs val）= 过拟合 ⚠️
- **Smoothing 滑块拖到 0** = 看真实波动，不被「磨皮」骗了

### HISTOGRAMS（权重直方图）— 看模型内部

实战验证：
- **Conv 层直方图不动** → `base.trainable=False` 冻结生效 ✅
- **Dense 层直方图在变** → 权重在更新，在学 ✅

不用猜「这层到底有没有在训练」，看一眼直方图就知道了。

### IMAGES（权重图像）

把卷积核画成小格子图。对初学者帮助不大，可暂时忽略。

## 两次 model.fit() 在 TensorBoard 中

代码里跑了冻结→微调两段训练，TensorBoard 显示 epoch 编号各自从 0 开始。TensorBoard 不知道这是「两个阶段」，它只看文件夹里的日志文件。

## 当前代码

```python
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    update_freq='epoch',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)
model.fit(Dataset1, epochs=5, validation_data=Dataset2,
          callbacks=[earlystopping, tensorboard])
```

## 反思

今天的训练太顺利了（5 epoch loss 一路降），所以没真正感受到 TensorBoard 的价值。就像体温计——量一次正常体温觉得「跟手摸也差不多」，但发烧的时候才知道它的好。

**明天 Day 18**：故意制造问题（学习率设错、过拟合、梯度消失），用 TensorBoard 定位问题。

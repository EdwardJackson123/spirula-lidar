# Spirula + LiDAR

独立代码库：[EdwardJackson123/spirula-lidar](https://github.com/EdwardJackson123/spirula-lidar)。当前版本为 **0.1.0，实验阶段**。已通过合成配准、模拟引擎流程和真实模型导出回归；现有两项真实数据自动对齐试验尚未通过验收，不能据此承诺任意数据都能自动得到可靠米制模型。详见[验证记录](reports/VALIDATION.json)。

把原始图像训练成 SH0 高斯模型，再对齐到激光地图的米制尺度。Python 负责流程、配准、验收和导出；训练调用独立的 Spirula 原生程序。

业务必需输入只有图像目录与激光点云。Spirula 程序、人物分割权重属于运行环境。点云支持 LAS、LAZ、普通 XYZ PLY，单位通过 `lidar_unit_to_m` 声明；默认按米读取。当前直接接收图像，不包含视频抽帧和压缩包解包。

## 安装

训练建议在 Linux NVIDIA GPU 服务器运行，Python ≥3.10。

```bash
git clone https://github.com/EdwardJackson123/spirula-lidar.git
cd spirula-lidar
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[worker,test]'
# 没有足够平面结构、需要 FPFH 备选搜索时额外安装：
pip install -e '.[features]'
spirula-lidar doctor --spirula /path/to/spirula
```

Spirula 不被复制或修改，使用[上游程序](https://github.com/harry7557558/spirula-studio/releases)。已检查的接口版本见 `engine.lock.json`。`doctor` 检查实际程序支持的参数，运行记录实际二进制 SHA256。该文件中的既有服务器版本带 `dirty` 标记，不能据此声称与干净源码编译完全一致。

## 运行

```bash
spirula-lidar run \
  --images /data/capture/images \
  --lidar /data/capture/scene.las \
  --output /data/your-space/run001 \
  --spirula /opt/spirula/spirula \
  --person-model /data/models/yolo11x-seg.pt \
  --gpu-index 0 --ordered-images
```

`--ordered-images` 只用于按文件名排序的连续采集；普通照片集省略。人物权重未指定时，Ultralytics 可向任务私有缓存下载 `yolo11x-seg.pt`，离线机器请提供已有权重。原图保留分辨率，不覆盖原始目录。EXIF 非正常朝向图像须先统一处理，当前实现会明确拒绝以免图像、遮罩与相机方向不一致。

批处理或服务调用使用 JSON：

```bash
spirula-lidar run --config /data/configs/job.json
# 示例见 examples/job.json；其中路径需替换。
```

流程为：原图清单和拷贝 → 人物遮罩 → 图像 SfM 及重投影检查 → 尺度对齐及验收 → MoGe2 深度/法线预测 → 显存预算 → Spirula 训练 → 米制导出。对齐失败时提前停止，保留证据，避免继续消耗训练资源。

默认配置为 SH0、原分辨率训练、预测深度/法线权重各 0.01、mild 漂浮物抑制、图像 SfM 点初始化。激光点不用于初始化和训练损失；代码不调用会替换种子和深度的 `spirula lidar`。

点数和步数由图像数量、SfM 点数、图像分辨率和实时空闲显存估算，并受 `training.max_gaussians`、`training.max_iterations` 限制。可用 `training.gaussians`、`training.iterations` 指定目标；显存上限仍生效。这是预算启发式，不是质量承诺。默认最多 18GiB GPU、64GiB RAM、16 线程，GPU 保留至少总显存的 20%；运行时资源不足会停止本任务并保留已有检查点。

## 对齐路径

**有参考位姿时优先使用它。** 参考必须是本次图像名称对应的相机位姿，并位于激光地图坐标系中：

```bash
spirula-lidar run --config job.json --reference-poses /data/reference/camera_poses.json
# 或 --reference-poses /data/reference/colmap/sparse/0
```

JSON 约定见 `examples/reference_poses.json`：OpenCV camera-to-world 4×4，至少 12 个真实匹配图像名。用拟合集估计旋转、尺度和平移，另留每四个中的一个相机验收；旋转约束也支持接近直线的相机轨迹。

此前跨批次 MegaLoc 检索 / SuperPoint-LightGlue / PnP 生成锚点的完整视觉定位栈**尚未内置**；它生成的本次图像米制位姿可接入此接口。只提供图像和 LAS 时，不会假装拥有参考图像或位姿。

**只有两项输入时使用独立几何配准。** 从图像 SfM 点与完整激光地图各自提取平面间距，生成多个尺度候选，遍历上下朝向、水平旋转和地图位置，分尺度保留候选，再用稳健 Sim3 ICP 精配准。没有足够平面时可使用可选 Open3D FPFH 备选。可重复结构、低重叠、局部畸变或弱几何下允许返回 `needs_review`，不保证每次成功。

`--landmarks landmarks.json` 可提供已有 3D 对应点作为初始化，字段为 `sfm_xyz`、`lidar_xyz` 和 `lidar_unit_to_m`；不跳过后续验收。

## 已有 SfM 或已训练模型

```bash
spirula-lidar align \
  --sfm /data/image_sfm/sparse/0 \
  --lidar /data/capture/scene.las \
  --output /data/your-space/alignment001

spirula-lidar align \
  --sfm /data/image_sfm/sparse/0 --lidar /data/capture/scene.las \
  --model /data/training/final/step-000070000.ckpt/splat.ply \
  --scene-transform /data/training/final/scene_transform.json \
  --reference-poses /data/reference/camera_poses.json \
  --output /data/your-space/export001
```

导出使用 `map_from_train = map_from_sfm ∘ inverse(train_from_world)`，同时更新高斯中心、对数尺度、旋转和相机外参；不要求训练归一化为单位变换。输出 PLY 为 SH0；非零高阶 SH 会被拒绝，不静默丢掉颜色信息。提供的外部 SfM 必须来自图像；程序不能仅靠文件格式判断它是否曾被激光点替换。

## 验收和输出

候选仅由拟合集择优；`alignment/sealed_solution.json` 固定之后才计算空间留出点指标。独立几何路径检查留出距离、分组表面距离和法线一致性、Sim3 可观测性及已测试替代解分差；参考路径检查参考相机留出位姿、尺度稳定性和点云留出距离。每条路径的实际检查分别记录，参考路径不会假称通过了独立几何的全部法线检查。

米制精度相对于声明的激光地图单位；没有独立实测长度时，绝对尺度误差为 `null`。有限搜索不能证明全局唯一。整体变换不会修复图像模型内部的局部拉伸或裂缝。

```text
run001/
  job.json                  输入/代码/引擎指纹
  status.json               当前阶段、进度、错误
  logs/                     子进程日志
  stages/                   完成标记和结果 SHA256
  dataset/                  私有图像、遮罩、SfM、预测几何
  alignment/alignment.json  路径、指标、检查结果
  alignment/preview.html    离线三视图叠加
  budget.json               点数/步数/缓存预算
  training/final/           原生训练结果和检查点
  output/scene.ply           仅验收通过后生成
  output/sparse/0/           激光坐标系中的相机与 SfM
  output/manifest.json       单位、坐标变换、模型 SHA256
```

退出码：0 完成，2 对齐需要人工复核，1 执行故障。`accepted` 表示通过本工具当前路径的检查，不代表测绘认证或可直接用于碰撞、导航。

## 续跑与进度

```bash
# 同一命令与同一输出目录可续跑，完成阶段会核验 SHA256 后复用。
spirula-lidar status /data/your-space/run001
spirula-lidar serve /data/your-space/run001 --port 8765
# 电脑侧：ssh -L 8765:127.0.0.1:8765 user@server，再打开 http://127.0.0.1:8765
```

监控默认只监听 localhost，5 秒读一次状态和真实日志，不调用模型服务。需要停止时在任务目录创建 `STOP` 文件；移除后执行原命令续跑。已有检查点通过 Spirula `--resume` 继续。输入、配置、引擎或代码改变时要求新输出目录，避免混用旧缓存。若进程恰在封存解之后、报告落盘之前中断，该次封存证据保留，当前版本要求新对齐目录重新执行。

`run --dry-run` 只检查输入并写计划，不启动推理；可在相同目录随后执行真实任务。

## 测试

```bash
pip install -e '.[test]'
pytest -q
```

测试包含已知 Sim3 与异常点、退化几何、任意地图原点、相机留出隔离、高斯协方差变换、非单位训练归一化、LAS 单位和缓存校验、拒绝导出、模拟训练引擎下的完整流程与续跑。模拟测试不等于实际 GPU 训练验收。实际服务器验证结果记录于 `reports/VALIDATION.json`，大模型与数据不进入 Git。

GitHub Actions 会在 Windows / Ubuntu、Python 3.10 / 3.12 上运行 CPU 测试并构建 wheel，不要求 GPU 或 Spirula。系统接入和状态约定见[集成说明](docs/INTEGRATION.md)，发布内容见[版本记录](CHANGELOG.md)。Spirula、分割模型及其依赖使用各自的上游许可，本仓库不随附这些程序、权重或采集数据。

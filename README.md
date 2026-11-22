# 3DGS 查看器部署说明

[下载部署包（v0.1.0）](https://github.com/AlinaLuuu/ascend-3dgs-viewer/releases/tag/v0.1.0) · [项目看板](https://github.com/users/AlinaLuuu/projects/1)

源码包含原生融合算子与可调 NPU 张量两种渲染路径。当前版本已完成 CPU 数学与浏览器验证，真实昇腾卡上的兼容性和性能仍需运行自检确认。

在昇腾 Linux 主机上运行此目录的 Python 服务，再从本机浏览器查看。鼠标改变相机后，服务用原生 `gauss_splat.Rasterizer` 在 NPU 上渲染 RGB，并返回 JPEG。浏览器只计算相机、接收和显示图像。部署包随附的 `lego.pt` 是公开 Lego PLY 转换成的 checkpoint，含 42,401 个三阶球谐高斯。[源模型](https://huggingface.co/datasets/pablovela5620/nerf-synthetic-mirror/blob/main/trained/lego.ply)

## 昇腾主机

目标库是 `gauss-splat` 提交 `25c6e337761ad2414a6d3377415681d229fe7a3b`。按目标设备的 CANN 配套安装驱动、CANN toolkit、对应 SoC 的算子包、Python 3.12 和匹配的 PyTorch／torch_npu。该提交的 README 列出 A2/A3 和 A5 构建方式。[gauss-splat 安装要求](https://github.com/Atlas-Ascend-Fork/gauss-splat)

上传并解压部署包，再在服务器的对应环境里构建算子。以下示例中，将设备序号、CANN 路径和项目位置改为服务器上的实际值。A5 用注释中的构建命令替换默认命令。

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/gauss-splat
bash build.sh --python=3.12
# Atlas A5: bash build.sh --python=3.12 --a5
python -m pip install dist/*.whl Pillow

cd ~/ascend-viewer
ASCEND_RT_VISIBLE_DEVICES=0 python server.py --check --output smoke.jpg
ASCEND_RT_VISIBLE_DEVICES=0 python server.py
```

查看器按原仓库调度逻辑使用逻辑 `npu:0`。若要使用另一张卡，将 `ASCEND_RT_VISIBLE_DEVICES` 设为那张卡的编号，让进程只看到一个设备，并先用 `--check` 验证。请从 `gauss-splat` 源码目录之外运行服务，以便加载安装好的 wheel。浏览器资源已经打包，昇腾服务器上的查看器无需 Node 或 npm。

`--check` 调用原生算子渲染 320 × 240 测试图，检查尺寸、有限值和非空画面，并输出设备与耗时。服务器默认只监听 `127.0.0.1:8766`。

## 用电脑连接

在你的电脑上打开 SSH 隧道，再在浏览器访问 [127.0.0.1:8766](http://127.0.0.1:8766/)。

```bash
ssh -N -L 8766:127.0.0.1:8766 用户名@昇腾服务器
```

左键旋转，右键平移，滚轮缩放，按 `R` 重置。面板可改渲染分辨率、球谐阶数、请求频率上限、JPEG 质量和视场角。浏览器一次只请求一帧，拖动过程中会合并尚未发出的相机更新。实际交互速度取决于 NPU、分辨率、模型大小和网络。

- **服务端推理**是原生光栅化调用至 NPU 同步完成的墙钟时间，包括调用中的 CPU 编排。
- **NPU 事件**是设备时间线上两个事件之间的差，随配套是否支持而显示。
- **回传准备**包括数据拷回 CPU 和 JPEG 编码。**取帧到解码**还包含请求往返和浏览器图像解码。

## 换模型和调整容量

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python server.py \
  --checkpoint /path/to/ckpt.pt --port 8766 --mask-budget-mib 1024
```

checkpoint 采用当前库的 N-first 格式，包含 `splats` 下的 `means[N,3]`、`quats[N,4]`、`scales[N,3]`、`opacities[N]` 和 `sh[N,K,3]`。尺度和不透明度沿用对数尺度和 logit。默认根据高斯中心估计初始取景。不同坐标约定可用 `--position X Y Z`、`--target X Y Z` 和 `--up X Y Z` 覆盖。此版本使用针孔相机和 32 × 32 Tile。

`--mask-budget-mib` 限制原库 Tile 相交稠密缓冲的估算大小，默认 1024 MiB。相交耗材按高斯数和 Tile 数增长，百万级场景建议从最低分辨率开始。这个预算只覆盖相交的主要缓冲，实际峰值还包含模型、缓存及其他算子工作区。

本机通过 CPU 参考渲染验证了界面、相机变换、HTTP 请求、图像回传和计时展示。原生 Ascend C 路径需要在目标卡上完成上文的 `--check`。[CANN 安装入口](https://www.hiascend.com/cann/download)

## 可调高斯与高级参数（2026-11-22）

在“渲染模式”选择“可调 NPU 张量渲染”后，高级控件可用，并同时作用于并排的两组模型。原生融合算子模式保持原路径，使用算子固定参数；切回可调模式保留设置。“恢复高级参数默认值”不改变相机。

| 参数 | 作用 |
| --- | --- |
| 背景色 | 用剩余透射率合成背景，非替换黑色像素 |
| 排序 | 相机深度或到相机的径向距离，从近到远混合 |
| 透明度阈值 | 每像素 alpha 下限，以 /255 为单位 |
| 不透明度倍率 | 乘激活后的模型 opacity，限制在 0–0.999 |
| 覆盖范围 | 椭圆马氏距离截断，单位 σ |
| 最小半径 | 剔除投影最大半径低于阈值的高斯 |
| 最大半径 | 等比例缩小投影协方差，限制最大半径 |
| 预滤波方差 | 屏幕空间协方差对角线上加方差，单位 px² |
| 抗混叠滤波方差 | 增大协方差，并按行列式比补偿 opacity |
| 投影雅可比裁剪余量 | 限制计算雅可比时的 x/z、y/z；并非直接裁剪图像边缘 |
| 近／远裁剪面 | 按高斯中心的相机空间深度剔除 |
| 焦距修正 | 同时缩放投影中心与协方差使用的焦距 |

可调模式在输入张量的 NPU 设备执行投影、排序和混合，沿用安装的原生球谐算子。它是独立的前向张量实现，不是原生融合算子增加了参数，也不保证与 Spark 逐像素一致。默认仍是原生模式。可调模式按 65,536 个高斯投影、16×16 像素 Tile、128 个高斯一批混合，避免原生稠密 Tile 相交缓冲，但百万高斯的交互可能明显变慢。建议先用 192×128 或 320×192；原生 mask 预算仅约束原生路径，不是可调模式的总内存上限。

每帧服务端推理与 NPU 事件计时包含当前选择的渲染路径，比较性能时必须记录模式。此实现已通过 CPU 数学验证，真实 NPU 支持与性能必须在目标环境验证；不做静默 CPU 回退。

```bash
# 原生路径自检
python server.py --check --output native.jpg
# 可调路径自检；Bicycle 包可把 python server.py 换成 bash launch.sh
python server.py --check --check-renderer tunable --output tunable.jpg
```

部署需要一并替换 `server.py`、`tunable.py` 和 `static/`，重启服务并刷新浏览器。可在安装 PyTorch 的环境运行 `python test_tunable.py` 做数学回归。

## 仓库与模型

代码仓库不包含模型权重。下载仓库 Releases 中的 `ascend-viewer.tar.gz` 可运行 Lego；`ascend-bicycle-viewer.tar.gz` 包含 Bicycle 原模型与稠密化开启／关闭两组权重及启动脚本。

若使用 Git 克隆源码，请从 Lego 部署包取出 `lego.pt` 放到 `server.py` 同级，或用 `--checkpoint /path/to/model.pt` 指定自己的权重。Bicycle 包请保留 `compare/metrics.json` 和两组 `model.pt`，使用包内 `launch.sh`。预训练模型来源与数据说明见上文及部署包 README。

前端依赖已经放在 `static/vendor`，Three.js 的 MIT 许可证见 `static/vendor/THREE-LICENSE.txt`。CANN、torch_npu 与 gauss-splat 需要按目标 NPU 的配套版本安装，不能用通用 requirements 自动选择。

## 学习资料

- [3DGS 入门指南](docs/3dgs_beginner_guide.md)
- [3DGS 算法与算子测试指南](docs/3dgs_testing_guide.md)
- [《3DGS 算法实现》PPTX · 22 页](docs/3DGS-algorithm-implementation.pptx)

两份指南保留原有内容与验证边界；公开副本将原来的本机绝对路径链接替换为固定版本的上游链接。PPT 保留原稿内容，演示入口指向本机服务，使用前需部署查看器。

## 开源许可

原创代码与原创文档内容采用 [MIT](LICENSE)，可下载、使用、修改和分发。第三方依赖、图片和模型保留各自许可，详见 [资源来源与许可范围](THIRD_PARTY_NOTICES.md)。

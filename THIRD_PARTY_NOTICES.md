# 许可证与资源来源

本仓库原创查看器代码、两份指南中的原创文字/示例，以及演示文稿中的原创文字/布局采用根目录 MIT 许可证。第三方代码、图片、数据集、模型权重、引用片段及商标不因本仓库公开而改用 MIT。

- `static/vendor/`：Three.js 与 OrbitControls，保留其 MIT 许可证 `static/vendor/THREE-LICENSE.txt`。
- 昇腾运行依赖 `gauss-splat`、CANN 和 torch_npu：分别遵守各自上游许可证；仓库没有重新分发这些运行库。gauss-splat 基线采用 CANN Open Software License Agreement Version 2.0，请参阅 https://github.com/Atlas-Ascend-Fork/gauss-splat/blob/25c6e337761ad2414a6d3377415681d229fe7a3b/LICENSE 。本仓库 MIT 不替代这些依赖条款。
- Lego 权重：由公开 PLY 转换为 checkpoint，来源 https://huggingface.co/datasets/pablovela5620/nerf-synthetic-mirror/blob/main/trained/lego.ply 。格式转换不改变上游权利。
- Bicycle 照片、场景及派生可视化：Mip-NeRF 360，https://jonbarron.info/mipnerf360/ 。原始照片与基于数据集的模型/渲染继续受原始资源许可约束。
- Bicycle 公开预训练权重：https://huggingface.co/datasets/alexmkwizu/gaussian_training_datasets/blob/main/tested_outputs/bicycle.ply 。稠密化对照权重由同一 Bicycle 数据集训练生成，详见部署包说明。
- PPT 中使馆与青铜器应用示意为生成插图；其余引用来源保留在讲者备注与相关指南中。

公开可下载不等于第三方资源自动获得不受限的商业授权。使用数据集、模型或 PPT 中的第三方图片时，请同时核对原始发布方的许可。演示文稿中的本机查看器链接要求先部署对应服务，不是公共在线演示站点。

## 👏 感谢RapidDoc (https://github.com/RapidAI/RapidDoc) 原作者

#### 🛠️ 安装英特尔GPU驱动，适用于各类集显和独显加速设备。
https://dgpu-docs.intel.com/driver/client/overview.html
#### 🛠️ 安装英特尔NPU驱动，适用于各类Core Ultra系列芯片。
https://github.com/intel/linux-npu-driver

---

# RapidDoc – 高速文档解析系统

## 😺 项目介绍

**RapidDoc 是一个轻量级、专注于文档解析的开源框架，支持 **OCR、版面分析、公式识别、表格识别和阅读顺序恢复** 等多种功能。**

**框架基于 [Mineru](https://github.com/opendatalab/MinerU) 二次开发，移除 VLM，专注于 Pipeline 产线下的高效文档解析，在 CPU 上也能保持不错的解析速度。**

**本项目所使用的核心模型主要来源于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的 [PP-StructureV3](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PP-StructureV3.html) 系列（OCR、版面分析、公式识别、阅读顺序恢复，以及部分表格识别模型），并已全部转换为 ONNX 格式，支持在 CPU/GPU 上高效推理。**

**KittyDoc 已经成为 RapidAI 开源家族成员**

---

> ✨如果该项目对您有帮助，您的star是我不断优化的动力！！！
>
> - [github点击前往](https://github.com/RapidAI/RapidDoc)
> - [gitee点击前往](https://gitee.com/hzkitty/KittyDoc)

## 👏 项目特点

- **OCR 识别**
  - 使用 [RapidOCR](https://github.com/RapidAI/RapidOCR) 支持多种推理引擎
  - CPU 下默认使用 OpenVINO，GPU 下默认使用 torch
  
- **版面识别**
  - 模型使用 `PP-DocLayout` 系列 ONNX 模型（plus-L、L、M、S）
    - **PP-DocLayout_plus-L**：效果最好，速度稍慢，默认使用 
    - **PP-DocLayout-L**：速度快，效果也不错
    - **PP-DocLayout-S**：速度极快，存在部分漏检

- **公式识别**
  - 使用 `PP-FormulaNet_plus` 系列 ONNX 模型（L、M、S）
    - **PP-FormulaNet_plus-L**：速度慢，支持onnx  
    - **PP-FormulaNet_plus-M**：默认使用，支持onnx和torch    
    - **PP-FormulaNet_plus-S**：速度最快，支持onnx，复杂公式精度不够
  - 支持配置只识别行间公式
  - cuda环境，默认使用torch推理，公式模型onnx gpu推理会报错，暂时无人解决 [PaddleOCR/issues/15125](https://github.com/PaddlePaddle/PaddleOCR/issues/15125), [PaddleX/issues/4238](https://github.com/PaddlePaddle/PaddleX/issues/4238), [Paddle2ONNX/issues/1593](https://github.com/PaddlePaddle/Paddle2ONNX/issues/1593)

- **表格识别**
  - 基于 [rapid_table_self](rapid_doc/model/table/rapid_table_self) 增强，在原有基础上增强为多模型串联方案：  
    - **表格分类**（区分有线/无线表格）
    - **SLANeXt** 系列 表结构识别 + 单元格检测
    - **[有线表格识别UNET](https://github.com/RapidAI/TableStructureRec)** + SLANET_plus/UNITABLE（作为无线表格识别）

- **阅读顺序恢复**
  - 使用 PP-StructureV3 阅读顺序恢复算法，基于xycut算法和版面的结果
  - 速度快效果好，支持多栏、竖排等复杂版面，和V3不开启版面子模块检测效果一致

- **推理方式**
  - 所有模型通过 ONNXRuntime 推理，OCR可配置其他推理引擎
  - 除了 OCR 和 PP-DocLayout-M/S 模型，OpenVINO推理会报错，暂时难以解决。[PaddleOCR/issues/16277](https://github.com/PaddlePaddle/PaddleOCR/issues/16277)
---

## 🛠️ 准备环境

#### 安装Python依赖包
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install loguru pypdfium2 requests boto3 pydantic pypdf reportlab magika opencv-python tqdm omegaconf colorlog tokenizers rapidocr fast_langdetect scikit-image bs4 pdfminer pdfminer.six pdftext matplotlib
pip install onnxruntime-openvino onnxruntime
pip install --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

#### 克隆到本地仓库
```bash
git clone https://github.com/chrisma-2025/RapidDoc-OpenVINO
```

#### 转换OCR模型成OpenVINO支持的IR文件
```bash
# 下载OCR模型
cd RapidDoc-OpenVINO/models
wget https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/det/ch_PP-OCRv5_server_det.onnx -P ocr
wget https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server_infer.onnx -P ocr
wget https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx -P ocr

# 转换模型
python ir_converter.py

# 确认models/ocr/文件夹下有以下文件
- ch_ppocr_mobile_v2.0_cls_infer.bin
- ch_ppocr_mobile_v2.0_cls_infer.onnx
- ch_ppocr_mobile_v2.0_cls_infer.xml
- ch_PP-OCRv5_rec_server_infer.bin
- ch_PP-OCRv5_rec_server_infer.onnx
- ch_PP-OCRv5_rec_server_infer.xml
- ch_PP-OCRv5_server_det.bin
- ch_PP-OCRv5_server_det.onnx
- ch_PP-OCRv5_server_det.xml
```

#### 参照修改openvino.py
https://github.com/chrisma-2025/RapidOCR-OpenVINO/blob/main/python/rapidocr/inference_engine/openvino.py   L46~59
* 已尝试编译安装修改后的RapidOCR，但执行时无法找到rapidocr包


#### 使用OpenVINO推理（默认使用GPU）
```bash
cd ..
python demo.py
```
* 通过注释和取消GPU配置文件来确定是否使用GPU，L82 为加载使用GPU，L81 为不加载则使用CPU
- https://github.com/chrisma-2025/RapidDoc-OpenVINO/blob/main/rapid_doc/model/ocr/rapid_ocr.py   L81,82

#### 目前OpenVINO支持的模型和设备
- Layout - ORT (OV EP CPU/GPU/NPU)
- Formula - ORT CPU
- OCR_det - ORT (OV EP CPU/GPU), OV (CPU/GPU)
- OCR_rec - ORT (OV EP CPU/GPU), OV (CPU/GPU)
- Table - ORT CPU

---

## 📋 使用示例

- [代码示例](./demo/demo.py)

- [参数介绍](./docs/analyze_param.md)

- [FastAPI 示例](./docker/README_API.md)
---

## 模型下载
不指定模型路径，初次运行时，会自动下载
- [RapidDoc 模型集（版面/公式/表格）](https://www.modelscope.cn/models/RapidAI/RapidDoc)  
- [RapidOCR 模型](https://www.modelscope.cn/models/RapidAI/RapidOCR)  
- [部分表格模型RapidTable](https://www.modelscope.cn/models/RapidAI/RapidTable) 

---

## 📌 TODO

- [x] 跨页表格合并
- [x] 复选框识别，使用opencv（默认关闭、opencv识别存在误检）
- [x] 提供 fastapi，支持cpu和gpu版本的docker镜像构建
- [x] 文本型pdf，表格非OCR文本提取
- [x] 文本型pdf，使用pypdfium2提取文本框bbox
- [x] 文本型pdf，支持0/90/270度三个方向的表格解析
- [x] 文本型pdf，使用pypdfium2提取原始图片（默认截图会导致清晰度降低和图片边界可能丢失部分）
- [x] 表格内公式提取
- [x] 表格内图片提取
- [x] 优化阅读顺序，支持多栏、竖排等复杂版面恢复
- [x] 公式支持torch推理，可用GPU加速
- [ ] 版面、表格支持openvino
- [ ] 支持 PP-DocLayoutV2 版面识别+阅读顺序
- [ ] 支持 PaddleOCR-VL（vlm-http-client模式）

- [ ] 公式支持openvino
- [ ] RapidDoc4j（Java版本）


## 🙏 致谢

- [MinerU](https://github.com/opendatalab/MinerU)
- [PaddleOCR & PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)

## ⚖️ 开源许可

基于 [MinerU](https://github.com/opendatalab/MinerU) 改造而来，已**移除原项目中的 YOLO 模型**，并替换为 **PP-StructureV3 系列 ONNX 模型**。  
由于已移除 AGPL 授权的 YOLO 模型部分，本项目整体不再受 AGPL 约束。

该项目采用 [Apache 2.0 license](LICENSE) 开源许可证。

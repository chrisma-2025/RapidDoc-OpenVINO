# 转换 ONNX 模型成 OpenVINO 支持的 IR 文件

import openvino as ov

# OCR模型
ocr_svr_det = "ocr/ch_PP-OCRv5_server_det.onnx"
ocr_svr_rec = "ocr/ch_PP-OCRv5_rec_server_infer.onnx"
ocr_mob_cls = "ocr/ch_ppocr_mobile_v2.0_cls_infer.onnx"
ocr_svr_det_ov = "ocr/ch_PP-OCRv5_server_det.xml"
ocr_svr_rec_ov = "ocr/ch_PP-OCRv5_rec_server_infer.xml"
ocr_mob_cls_ov = "ocr/ch_ppocr_mobile_v2.0_cls_infer.xml"

# Formula模型
formula = "formula/pp_formulanet_plus_s.onnx"
formula_ov = "formula/pp_formulanet_plus_s.xml"

# Layout模型
layout = "layout/pp_doclayout_plus_l.onnx"
layout_ov = "layout/pp_doclayout_plus_l.xml"

# Table模型
table = "table/slanet-plus.onnx"
table_ov = "table/slanet-plus.xml"

onnx_models = [ocr_svr_det,ocr_svr_rec,ocr_mob_cls]
ov_models = [ocr_svr_det_ov,ocr_svr_rec_ov,ocr_mob_cls_ov]

# 编译 ONNX 模型
# for onnx_model in onnx_models:
#     compiled_model = ov.compile_model(onnx_model)

# 将 ONNX 模型转换成 IR 格式
for model in onnx_models:
    ov_model = ov.convert_model(model)
    xml_path = model.split('.')[0]+".xml"
    ov.save_model(ov_model, xml_path)

# 编译 OV 模型
# for ov_model in ov_models:
#     compiled_model = ov.compile_model(ov_model)
 

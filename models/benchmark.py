# 性能对比：ORT(CPU EP/OV EP CPU/OV EP GPU) vs. OV (CPU/GPU)

import onnxruntime as ort
import numpy as np
from time import time

# Specify OpenVINO as the Execution Provider
providers=["CPUExecutionProvider","OpenVINOExecutionProvider"]
model = "ocr/ch_PP-OCRv5_server_det.onnx"
input_name = 'x' # session.get_inputs()[0].name
input_tensor = np.random.random([1,3,800,800]).astype(np.float32)
input_data = {input_name: input_tensor}
devices = ['CPU','GPU']

warmup = 1
iterations = 10

def _ort_infer(session, input_data, fw, device):
    # warmup
    st = time()
    for _ in range(warmup):
        output = session.run(None, input_data)
    ed = time()
    print(f"[{device}] [{fw}] infer time (warmup): {(ed-st)/warmup:.3f}s")

    # Run inference
    st = time()
    for _ in range(iterations):
        output = session.run(None, input_data)
    ed = time()
    print(f"[{device}] [{fw}] infer time (avg): {(ed-st)/iterations:.3f}s")
    # print(output)

for d in devices:
    for p in providers:
        if d =='GPU' and p == 'CPUExecutionProvider':
            continue
            
        session = ort.InferenceSession(
            model,
            providers=[p],
            provider_options=[{'device_type': d, 'cache_dir': './cache'}] # Options: CPU, GPU, NPU
        )

        _ort_infer(session, input_data, p, d)


import openvino as ov
from openvino import Core

ir = "./ch_PP-OCRv5_server_det.xml"
core = Core()

def _ov_infer(compile_model, input_data, device):
    # warmup
    st = time()
    for _ in range(warmup):
        out = compile_model(input_data)
    ed = time()
    print(f"[{device}] [OpenVINO] infer time (warmup): {(ed-st)/warmup:.3f}s")

    # # Run inference
    st = time()
    for _ in range(iterations):
        out = compile_model(input_data)
    ed = time()
    print(f"[{device}] [OpenVINO] infer time (avg): {(ed-st)/iterations:.3f}s")
    # print(out)

for d in devices:
    compile_model = core.compile_model(ir, device_name=d, config={'CACHE_DIR': './cache','INFERENCE_PRECISION_HINT':'f32','PERFORMANCE_HINT':'LATENCY'})
    _ov_infer(compile_model, input_data, d)

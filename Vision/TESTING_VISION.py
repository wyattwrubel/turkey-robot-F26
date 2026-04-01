# save as test_hailo.py and run from your venv
import numpy as np
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface,
    ConfigureParams, InputVStreamParams,
    OutputVStreamParams, FormatType, InferVStreams,
)

MODEL_PATH = "/home/turkeysquad/ME450/MainDesign/Vision/weights/turkey_640.hef"

hef    = HEF(MODEL_PATH)
target = VDevice()

configure_params = ConfigureParams.create_from_hef(
    hef=hef, interface=HailoStreamInterface.PCIe
)
network_groups       = target.configure(hef, configure_params)
network_group        = network_groups[0]
network_group_params = network_group.create_params()

input_vstream_params  = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

print("=== INPUT STREAMS ===")
for info in hef.get_input_vstream_infos():
    print(f"  name={info.name}  shape={info.shape}  dtype={info.format}")

print("=== OUTPUT STREAMS ===")
for info in hef.get_output_vstream_infos():
    print(f"  name={info.name}  shape={info.shape}  dtype={info.format}")

# Run one dummy inference with a blank frame
input_info = hef.get_input_vstream_infos()[0]
h, w = input_info.shape[0], input_info.shape[1]
dummy = np.zeros((1, h, w, 3), dtype=np.uint8)

with InferVStreams(network_group, input_vstream_params, output_vstream_params) as pipeline:
    with network_group.activate(network_group_params):
        raw = pipeline.infer({input_info.name: dummy})

print("=== RAW OUTPUT TENSORS ===")
for name, tensor in raw.items():
    arr = np.array(tensor)
    print(f"  name={name}  shape={arr.shape}  min={arr.min():.4f}  max={arr.max():.4f}")
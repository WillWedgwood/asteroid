import onnx
import onnxruntime as ort

# Load the ONNX model
onnx_model_path = "onnxruntime/conv_tasnet_JorisCos_16k.onnx"
onnx_model = onnx.load(onnx_model_path)

# Check that the model is well-formed
onnx.checker.check_model(onnx_model)

# Create an ONNX Runtime session
ort_session = ort.InferenceSession(onnx_model_path)

# Print the input and output names
input_names = [input.name for input in ort_session.get_inputs()]
output_names = [output.name for output in ort_session.get_outputs()]
print("Input names:", input_names)
print("Output names:", output_names)
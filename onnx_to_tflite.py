import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

# STEP 1: Load ONNX model
onnx_model = onnx.load("model.onnx")

# STEP 2: Convert ONNX → TensorFlow SavedModel
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")  # Creates a folder "model_tf/"

# STEP 3: Convert TensorFlow → TensorFlow Lite (.tflite)
converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # enable quantization
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved TensorFlow Lite model as model.tflite")

# STEP 4: Quick test with dummy input
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite Input shape:", input_details[0]['shape'])
print("TFLite Output shape:", output_details[0]['shape'])

dummy_input = np.random.rand(1, 1, 28, 28).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Dummy output:", output_data)

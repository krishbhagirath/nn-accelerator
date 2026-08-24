from pathlib import Path
import onnx

project_root = Path(__file__).resolve().parent.parent
onnx_path = project_root / "model_data" / "tiny_nn.onnx"

model = onnx.load(onnx_path)

print("Model inputs:")
for inp in model.graph.input:
    print(" ", inp.name)

print("\nModel outputs:")
for out in model.graph.output:
    print(" ", out.name)

print("\nGraph operations:")
for node in model.graph.node:
    print(
        f"{node.op_type}: "
        f"inputs={list(node.input)} "
        f"outputs={list(node.output)}"
    )

print("\nStored parameters:")
for tensor in model.graph.initializer:
    print(tensor.name, tensor.dims)
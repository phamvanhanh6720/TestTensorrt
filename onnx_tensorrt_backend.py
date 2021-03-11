import onnx
import onnx_tensorrt.backend as backend
import numpy as np
from memory_profiler import profile
import time

@profile()
def main():
    model = onnx.load('./mobilenet0_25.onnx')
    engine = backend.prepare(model, device='CUDA:0')

    num_test = 100
    count = 0
    input_data = np.random.random(size=(1, 3, 480, 640)).astype(np.float32)
    for i in range(num_test):
        start = time.time()
        output_data = engine.run(input_data)[0]
        count += (time.time() - start)

    print("Avg Time: {:.4f}".format(count/num_test))

    print(output_data)
    print(output_data.shape)


if __name__ == '__main__':
    main()
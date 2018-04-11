import sys
import cntk


def convert(onnx_file, cntk_file):
    """
    Converts the ONNX model file into a native CNTK one.

    :param onnx_file: the ONNX model file to load
    :type onnx_file: str
    :param cntk_file: the CNTK model file to save to
    :type cntk_file: str
    """
    model = cntk.Function.load(onnx_file, format=C.ModelFormat.ONNX)
    model.save(cntk_file)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Requires ONNX input model and CNTK output model")
        print("E.g. in.onnx")
        exit(1)

    convert(sys.argv[1], sys.argv[2])

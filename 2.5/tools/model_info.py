import sys
import cntk


def output_info(model_file):
    """
    Loads the model and prints the outputs of the model to stdout.

    :param model_file: the model file to load
    :type model_file: str
    """
    print("\n" + m)
    model = cntk.load_model(model_file)
    print("\nOutputs:")
    for output in cntk.logging.get_node_outputs(model):
        print(output, "- UID:", output.uid)

    print("\nVariables:")
    for block in cntk.logging.graph.depth_first_search(model, (lambda x: type(x) == cntk.Variable)):
        print(block, "- UID:", block.uid)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("No model file supplied!")
        print("E.g. /some/where/PretrainedModels/ResNet_18.model")
        exit(1)

    for m in sys.argv[1:]:
        output_info(m)

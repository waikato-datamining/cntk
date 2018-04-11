import sys
import cntk


def output_info(model_file):
    """
    Loads the model and prints the outputs of the model to stdout.

    :param model_file: the model file to load
    :type model_file: str
    """
    model = cntk.load_model(model_file)
    for output in cntk.logging.get_node_outputs(model):
        print(output)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("No model file supplied!")
        print("E.g. /some/where/PretrainedModels/ResNet_18.model")
        exit(1)

    output_info(sys.argv[1])

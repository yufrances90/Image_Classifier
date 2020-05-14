from utils.io import IO
from utils.classifer import Classifier

def main():

    in_arg = IO.get_input_args(train=False)

    device = IO.get_device(in_arg.gpu)

    category_names = IO.get_label_mapping(in_arg.category_names)

    checkpoint = IO.load_checkpoint(in_arg.checkpoint)

    classifier = Classifier.generate_classifier_by_checkpoint(checkpoint, in_arg.arch, device)

    print(classifier.model)

if __name__ == "__main__":
    main()
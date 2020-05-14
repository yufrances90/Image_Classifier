from utils.io import IO
from utils.classifer import Classifier
from utils.image import ImageUtils

def main():

    in_arg = IO.get_input_args(train=False)

    device = IO.get_device(in_arg.gpu)

    category_names = IO.get_label_mapping(in_arg.category_names)

    checkpoint = IO.load_checkpoint(in_arg.checkpoint)

    classifier = Classifier.generate_classifier_by_checkpoint(checkpoint, in_arg.arch, device)

    image = IO.load_image(in_arg.image_path)

    image_tensor = ImageUtils.get_image_tensor(image)

    probs, classes = classifier.predict(image_tensor, in_arg.top_k, checkpoint['class_to_idx'])

    print(probs)
    print(classes)

if __name__ == "__main__":
    main()
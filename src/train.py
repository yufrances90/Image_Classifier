from utils.io import IO
from utils.classifer import Classifier

def main():

    in_arg = IO.get_input_args(train=True)

    device = IO.get_device(in_arg.gpu)

    dataloaders, class_to_idx = IO.get_image_data(in_arg.data_directory)

    classifier = Classifier(
        arch=in_arg.arch, 
        hidden_units=in_arg.hidden_units, 
        output_units=102,
        learning_rate=in_arg.learning_rate,
        epochs=in_arg.epochs,
        device=device
    )

    classifier.train_model(dataloaders['trainloader'], dataloaders['validloader'])

    trained_classifier = classifier.get_trained_classifier()

    checkpoint = {
        'classifier': trained_classifier['classifier'],
        'state_dict': trained_classifier['state_dict'],
        'learning_rate': trained_classifier['learning_rate'],
        'epochs': trained_classifier['epochs'],
        'class_to_idx': class_to_idx
    }

    IO.save_checkpoint(checkpoint, in_arg.save_dir)

if __name__ == "__main__":
    main()
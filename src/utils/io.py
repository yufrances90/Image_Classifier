import argparse
import torch
from torchvision import transforms, datasets
import json

class IO:

    @staticmethod
    def get_input_args(train=True):

        parser = argparse.ArgumentParser()

        ##########
        #
        # train.py
        #
        ##########

        # Basic usage: python train.py data_directory

        if train:
            parser.add_argument("data_directory", type=str,
                        help="Folder for Data")

        parser.add_argument("--save_dir", type=str, default = 'output/',
                        help="Folder for Checkpoints")

        parser.add_argument("--arch", type=str, default = 'vgg16',
                        help="CNN Model Architecture")

        parser.add_argument("--learning_rate", type=float, default = 0.003,
                        help="Learning Rate")

        parser.add_argument("--hidden_units", type=int, default = 256,
                        help="Hidden Units")

        parser.add_argument("--epochs", type=int, default = 200,
                        help="Number of Epochs")

        parser.add_argument("--gpu", type=bool, default = True,
                        help="If Use GPU for Training")

        ##########
        #
        # predict.py
        #
        ##########

        # Basic usage: python predict.py /path/to/image checkpoint

        parser.add_argument("--top_k", type=int, default = 5,
                        help="top K most likely classes")

        parser.add_argument("--category_names", type=str, default = 'cat_to_name.json',
                        help="A mapping of categories to real names")

        if not train:

            parser.add_argument("checkpoint", type=str,
                        help="Checkpoint")

            parser.add_argument("image_path", type=str,
                        help="/path/to/image")

        # Use GPU for inference

        return parser.parse_args()


    @staticmethod
    def get_device(gpu):
        return torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    
    @staticmethod
    def get_label_mapping(category_names_file):

        with open(category_names_file, 'r') as f:
            category_names = json.load(f)

        return category_names


    @staticmethod
    def get_image_data(data_directory):

        data_dirs = IO.__get_image_dirs(IO, data_directory)

        data_transform = IO.__get_data_transform(IO)

        datasets = IO.__get_datasets(IO, data_dirs, data_transform)

        return IO.__get_dataloaders(IO, datasets)

    
    @staticmethod
    def save_checkpoint(checkpoint, save_dir):

        file_path = f'{save_dir}checkpoint.pth'

        torch.save(checkpoint, file_path)



    def __get_dataloaders(self, datasets):

        dataloader_arr = [torch.utils.data.DataLoader(dataset, batch_size=32) for dataset in datasets]

        class_to_idx = datasets[0].class_to_idx

        return  {
            'trainloader': dataloader_arr[0],
            'validloader': dataloader_arr[1],
            'testloader': dataloader_arr[2]
        }, class_to_idx

    def __get_datasets(self, data_dirs, data_transform):

        return [datasets.ImageFolder(data_dir, transform=data_transform) for data_dir in data_dirs]

    def __get_data_transform(self):
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        return transforms.Compose([transforms.Resize(256), 
                                       transforms.CenterCrop(224),              
                                       transforms.ToTensor(), 
                                       normalize]) 

    def __get_image_dirs(self, data_dir):

        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        return [train_dir, valid_dir, test_dir]


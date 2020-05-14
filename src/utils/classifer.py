import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

model_dict = {
    'resnet18': resnet18, 
    'alexnet': alexnet, 
    'vgg16': vgg16
}

class Classifier:

    def __init__(
        self, 
        arch, 
        hidden_units, 
        output_units,
        learning_rate,
        epochs,
        device
    ):

        self.__initialize_model(arch, hidden_units, output_units)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.classifier.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate

        self.epochs = epochs

        self.device = device

    def get_trained_classifier(self):

        self.model.to(torch.device("cpu"))

        return {
            'classifier': self.model.classifier,
            'state_dict': self.model.state_dict(),
            'learning_rate': self.learning_rate,
            'epochs': self.epochs
        }

    def train_model(self, trainloader, validloader):

        self.model.to(self.device)

        for epoch in range(1, self.epochs+1):

            train_loss = 0.0
    
            for batch_i, (images, labels) in enumerate(trainloader):

                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
        
                output = self.model(images)
                
                loss = self.criterion(output, labels)
                
                loss.backward()
                
                self.optimizer.step()
                
                train_loss += loss.item()

            else:

                valid_loss = 0
                accuracy = 0
        
                with torch.no_grad():

                    self.model.eval()
            
                    for images, labels in validloader:
                
                        images, labels = images.to(self.device), labels.to(self.device)
                
                        ps = self.model(images)
                
                        valid_loss += self.criterion(ps, labels)
                
                        top_p, top_class = ps.topk(1, dim=1)

                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                
                train_p = train_loss/len(trainloader)
                valid_p = valid_loss/len(validloader)

                self.model.train()

                print(
                    "Epoch: {}".format(epoch),
                    "Train Loss: {:.3f}.. ".format(train_p),
                    "Valid Loss: {:.3f}.. ".format(valid_p),
                    "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))

    def __initialize_model(self, arch, hidden_units, output_units):

        self.model = model_dict[arch]

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[len(self.model.classifier)-1] = nn.Sequential(
            nn.Linear(
                self.model.classifier[len(self.model.classifier)-1].in_features, 
                hidden_units
            ), 
            nn.ReLU(), 
            nn.Dropout(0.4),
            nn.Linear(hidden_units, output_units))



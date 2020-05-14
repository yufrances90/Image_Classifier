import numpy as np
import matplotlib.pyplot as plt
import torch

class ImageUtils:

    @staticmethod
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
    
        ######
        #
        # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
        #
        ######
        
        default_shortest_size = 256
        
        w = image.width
        h = image.height
        
        width = 0
        height = 0

        if w < h:
        
            width = default_shortest_size
            height = int((h / w) * width)
            
        else:
            
            height = default_shortest_size
            width = int((w / h) * height)
            
        im_resize = image.resize((width, height))

        ######
        #
        # crop out the center 224x224 portion of the image
        #
        ######
        
        # The crop method from the Image module takes four coordinates as input.
        # The right can also be represented as (left+width)
        # and lower can be represented as (upper+height).
        (left, upper, right, lower) = (20, 20, 244, 244)

        im_crop = im_resize.crop((left, upper, right, lower))

        ######
        #
        # normalize image 
        #
        ######
        
        # subtract the means from each color channel, then divide by the standard deviation
        
        np_im  = np.array(im_crop)
        
        np_im_updated = np_im.reshape(np_im.shape[0] * np_im.shape[1], np_im.shape[2])
        
        mean = np_im_updated.mean(axis=0)
        std = np_im_updated.std(axis=0)
        
        np_image = (np_im - mean) / std

        ######
        #
        # PyTorch expects the color channel to be the first dimension but it's the third dimension 
        # in the PIL image and Numpy array
        #
        ######
        
        return np_image.transpose((2, 0, 1))

    @staticmethod
    def imshow(image, ax=None, title=None):

        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
    
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
    
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        ax.set_title(title)
        
        ax.axis('off')
    
        return ax

    
    @staticmethod
    def get_image_tensor(image):

        im_np = ImageUtils.process_image(image).astype(np.float32)
    
        return torch.from_numpy(np.expand_dims(im_np, axis=0))
    
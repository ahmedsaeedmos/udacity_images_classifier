import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse



def get_args():
    parser = argparse.ArgumentParser(description= "code for trainig set")

    parser.add_argument('--topk'  , type = int ,  help= "select your top classs need to show "  , default = 5 )
    parser.add_argument('--json_file'   ,dest='json_file' ,  type = str , help= "selct your json file " ,default =  'cat_to_name.json' )
    parser.add_argument('--save_dir'   , dest='save_dir', type = str , help= "selct your directory to Extracts checkpoint " , default = 'checkpoint.pth')
    parser.add_argument('--device'   , type = str , help= "selct your device cpu or gpu "  , default = 'gpu')
    parser.add_argument('--image_dir'   , type = str , help= "set image directry" ,default = 'flowers/test/10/image_07090.jpg')
    args = parser.parse_args()
    return args



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model_name']
    model.classifier = checkpoint['classifier']
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['model.class_to_idx']
    
    return model 



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    
    transform =transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    np_image = transform ( im)
    return np_image
    
    
    
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, topk ):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    
    model.cpu()
    
    image = process_image(image_path)
    
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model.forward(image)
        probs, labels = torch.topk(output, topk)        
        probs = probs.exp()
        
        class_to_idx_rev = {model.class_to_idx[k]: k for k in model.class_to_idx}
        
        classes = []
        
        for label in labels.numpy()[0]:
            classes.append(class_to_idx_rev[label])

        return probs.numpy()[0], classes
    
    

def main ():
    args = get_args()
    
    model = load_checkpoint(args.save_dir)
    import json

    with open(args.json_file, 'r') as f:
        cat_to_name = json.load(f)
    
        
    probs, classes = predict(args.image_dir, model , args.topk)
    


    for i in range(args.topk):
        print("Probability - {} - Class - {}".format(probs[i], classes[i]))
    

if __name__ == '__main__' :
    main()
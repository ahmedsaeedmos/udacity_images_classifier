
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

    parser.add_argument('--archtecture'  , metavar = '' , type = str ,  help= "select your model name " , default = 'vgg16' )
    parser.add_argument('--learningrate'   ,dest='learningrate' ,  type = str , help= "selct your leaning rate " ,default = 0.001)
    parser.add_argument('--epochs'   ,dest='epochs' , type = int , help= "selct your num of epochs "  ,default = 3)
    parser.add_argument('--save_dir'   , dest='save_dir', type = str , help= "selct your directory to save the model "  , default = 'checkpoint.pth')
    parser.add_argument('--device'   , type = str , help= "selct your device CPU or GPU "  , default = 'gpu')
    parser.add_argument('--hidden_layer'   , type = int , help= "selct your hidden layers " ,default = 2048)
    args = parser.parse_args()
    return args


def tranform (data_dir):
   
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_transforms =transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    #  Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir , transform= train_transforms)
    test_datasets = datasets.ImageFolder(test_dir , transform= test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir , transform=valid_transforms)



    # Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64 ,shuffle= True )
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64 , shuffle = True )
    return trainloaders , testloaders , validloaders , train_datasets


def cat_to_name ():
    import json

    with open('cat_to_name.json', 'r') as f:
       cat_to_name = json.load(f)
    return  cat_to_name




def chcek(dev):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    return device 



def model_name (model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    if model_name == "vgg16" : 
        model = models.vgg16(pretrained=True)
    elif model_name == "densenet121" :
        model = models.densenet121(pretrained=True)
    elif model_name == "vgg11" :
        model = models.vgg11(pretrained=True)
    else :
        print ('unsupported models')
    
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def classifier (model ,hidden_layers):
    
    input_features = model.classifier[0].in_features

    classifier = nn.Sequential (nn.Linear(input_features ,hidden_layers) , 
                                 nn.ReLU() , 
                                 nn.Dropout(0.2) , 
                                 nn.Linear(hidden_layers , 102) , 
                                
                                  
                                 nn.LogSoftmax (dim = 1))

    return classifier



def grad_des (trainloaders ,validloaders , testloaders ,epochs , optimizer , print_evry , model , device , criterion):
    
    
    Running_Loss = 0
    steps = 0
    for epoch in range(epochs) :
        
        for inputs , labels in trainloaders : 

            
            steps += 1 
            
            inputs , labels = inputs.to(device) , labels.to(device)
            optimizer.zero_grad()
            log = model.forward(inputs)
            loss=  criterion(log , labels)
            loss.backward()
            Running_Loss += loss.item()
            optimizer.step()
            
            if steps % print_evry == 0:
                validation_Loss =0
                Accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs , labels in validloaders :
                        inputs , labels = inputs.to(device) , labels.to(device)
                        log = model.forward(inputs)
                        validation_Loss += criterion (log , labels)
                        
                        #calculate accuracy 
                        ps = torch.exp(log)
                        top_p , top_class = ps.topk(1 , dim = 1)
                        equls = top_class == labels.view(*top_class.shape)
                        Accuracy += torch.mean(equls.type(torch.FloatTensor)).item()
                        
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {Running_Loss/print_evry:.3f}.. "
                      f"validation_Loss: {validation_Loss/len(testloaders):.3f}.. "
                      f"validation Accuracy: {Accuracy/len(testloaders):.3f}.."  )
                
                Running_loss = 0
                model.train()
                
    return model



def testing_accuracy (model , testloaders , device ):
    total = 0
    correct = 0

    with torch.no_grad():
        model.eval()
        for inputs, labels in iter(testloaders):
            inputs , labels = inputs.to(device) , labels.to(device)
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy obtained on the testing data - {((100 * correct) / total):.3f}" )
            
        

def check_point ( classifier , optimizer , class_to_idx , state_dict , model  ):
    checkpoint = { 'model_name' : model ,
               'classifier' : classifier ,
               'model.class_to_idx' : class_to_idx ,
              'optimizer.state_dict' : optimizer (),
              'state_dict' : state_dict() }
    
    return checkpoint 

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    model = checkpoint['model_name']
    model.classifier = checkpoint['classifier']
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['model.class_to_idx']
    
    return model 





def main ():
    args = get_args()
    
    
    trainloaders , testloaders , validloaders , train_datasets = tranform('flowers')
    
    ahmed = cat_to_name()
    
    #check the device 

    
    
    #select the model 
    
    model = model_name(args.archtecture)
    
    #run classifier 
    
    model.classifier = classifier ( model , hidden_layers = args.hidden_layer)
    
    
    


    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    
    #check the device 

    device = chcek(args.device)
    
    model.to(device)
    
    
    
    print_evry = 32 
    
    
    the_model = grad_des (trainloaders ,validloaders , testloaders ,args.epochs , optimizer , print_evry , model  , device , criterion)
    

    testing_accuracy (the_model , testloaders , device  )
    
    
    model.class_to_idx = train_datasets.class_to_idx
    
    
    checkpoint =  check_point ( model.classifier , optimizer.state_dict , model.class_to_idx , model.state_dict , model )
    
    
    torch.save (checkpoint , args.save_dir)
    
    
    new_model = load_checkpoint(args.save_dir)
    
    print("save the the in my checkpoint")
    
    


if __name__ == '__main__' :
    
    main()
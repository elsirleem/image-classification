from time import time, sleep, localtime, strftime
from torchvision import datasets, transforms
import torchvision.models as models
import torchvision.models as models
from collections import OrderedDict
import torchvision.models as models
import torch.nn.functional as F
from torch import optim
from PIL import Image
from torch import nn
import numpy as np
import predict
import argparse
import glob, os
import torch
import json


# Define transforms for the training, validation, and testing sets
def Train(args):
    data_dir = 'flowers'
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),normalize])

    val_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                               normalize])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             normalize]) 


    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(valid_dir, transform=val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)
    valloader= torch.utils.data.DataLoader(val_dataset,batch_size=32)
    testloader= torch.utils.data.DataLoader(test_dataset,batch_size=32)
    
    
    print('Pretrained vgg model loading\n')
   
    model=models.vgg16(pretrained=True)
    Feature = model.classifier[0].in_features
    output_size = 102

    for parm in model.parameters():
        parm.requires_grad=False;
    network = nn.Sequential(OrderedDict([('dropout1', nn.Dropout(p=0.3)),
                                            ('fc1', nn.Linear(Feature, args.hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout2', nn.Dropout(p=0.3)),
                                            ('fc2', nn.Linear(args.hidden_units, output_size)),  
                                            ('relu2', nn.ReLU()),
                                            ('output', nn.LogSoftmax(dim = 1)),
                                        ]))
    model.classifier= network
   
    model.class_to_idx = train_dataset.class_to_idx
    model.optim_state_dict = optimizer.state_dict()
    criterion = nn.NLLLoss()
    optimizer= optim.Adam(model.classifier.parameters(),lr=args.lr)
   
        
    print("Start Traing\n")
    if(args.gpu):
        if torch.cuda.is_available():
            device = 'gpu'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
      
    start_time = time()
    model.to(device)
    accuracy = 0
    for e in range(epochs):

        running_loss=0;
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad();
            output=model(images)
            loss=criterion(output,labels)
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                val_loss=0;
                save_accuracy= -10;

                for images , labels in valloader:
                    images, labels = images.to(device), labels.to(device)
                    output=model(images)
                    loss=criterion(output,labels)
                    val_loss+=loss.item()
                    output_Exp=torch.exp(output)
                    top_p,top_c = output_Exp.topk(1,dim=1)
                    equals= top_c ==labels.view(*top_c.shape)
                    accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                model.optim_state_dict=optimizer.state_dict()
                if accuracy>save_accuracy:
                    accuracy=save_accuracy
                    model.save_accuracy=accuracy
                    model.epoch=e
                    
                    saveCheckPoint(model,'checkpoint.ph')
               
                print(f"Epoch {e+1}/{epochs}.. ",
                  "Train loss: {}","Valloss: {}","valac: {}".format(running_loss/len(trainloader),
                  val_loss/len(valloader),
                  save_accuracy/len(valloader)))
                 
        model.train()
    return model
    
    print("Train successful\n")
    end_time = time()
    tot_time = end_time - start_time
    tot_time = strftime('%H:%M:%S', localtime(tot_time))
    print("\n** Total Elapsed Training Runtime: ", tot_time)
    
    model.epoch = args.epochs
    model.class_to_idx = train_dataset.class_to_idx
    


    
    print('Saving  the trained model')
    # predict.saveCheckPoint(model,args)
def saveCheckPoint(model,args):
    model=models.vgg16(pretrained=True)
    Feature = model.classifier[0].in_features
    checkpoint = {
               'state_dict': model.state_dict(),
               'epoch': model.epoch,
               'optimizer_state':model.optim_state_dict,
               'class_to_idx': model.class_to_idx,
               'output_size': 102,
               'fearure':Feature,
               'hidden_units':args.hidden_units,
                'accuracy':model.accuracy
             }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(checkpoint,save_dir +'checkpoint.pth')
    print('Model saved')



    
    print('Loading the saved model')
    # predict.loadCheckpoint(checkpointPath)
def loadCheckpoint(checkpointPath):
    checkpoint =torch.load(checkpointPath)
    model.classifier= nn.Sequential(OrderedDict([
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc1', nn.Linear(checkpoint['feature'],checkpoint['hidden_units'] )),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'])),
                              ('relu2', nn.ReLU()),
                              ('dropout3', nn.Dropout(p=0.5)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
                               
    model.class_to_idx=checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.epoch=checkpoint['epoch']
    model.optimizer_state=checkpoint['optimizer_state']
    print("Loading checkpoint done with sucess\n")
    return model    






if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description='Flowers Classifcation Trainer')
    parser.add_argument('--gpu', action= 'store_true' , help='Utilize gpu to train')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture [available: vgg16]')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--input', type=str, help='path for image to predict')
    parser.add_argument('--save_dir' , type=str, default='./', help='checkpoint directory path')
    args = parser.parse_args()

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    Train(args)
    print(" ***Finished***\n")


    
        




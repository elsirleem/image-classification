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
import argparse
import glob, os
import torch
import json

def evaluate(model,criterion,testloader,device):
    print('Evaluation in process...')
    with torch.no_grad():
            model.eval()
            val_loss=0;
            accuracy=0;
            for images , labels in testloader:
                images, labels = images.to(device), labels.to(device)
                output=model(images)
                loss=def_criterion(output,labels)
                val_loss+=loss.item()
                output_Exp=torch.exp(output)
                top_p,top_c = output_Exp.topk(1,dim=1)
                equals= top_c ==labels.view(*top_c.shape)
                accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"test  loss: {val_loss/len(testloader):.3f}.. "
                  f"test  accuracy: {accuracy/len(testloader):.3f}")
            print('Done eval.....')
    model.train()




def PredictFunctoin(args):
    if(in_arg.gpu):
        if torch.cuda.is_available():
            device = 'gpu'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
    model=loadCheckpoint(args.checkpoint_path)
    model.to(device)

    indexClass={}

    for i,value in model.class_to_idx.items():
        indexClass[value]=i
    probability, classes = utils.ImagePrediction(args.input, model,args.top_k,device,indexClass)
      
    if(args.category_names):
        cat_to_name=[]
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            names = [cat_to_name[c] for c in classes]

    print('Classes: {}','probabilities(%): {}','Top Classes: {}'.format(names,[float(round(p * 100.0, 2)) for p in probs],names[0]))



def ProcessImage(image):
    print('Crop the image for prediction')
    pro_image = Image.open(image)

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                                ])
    np_image = preprocess(pro_image)
    print('crop finished')
    return np_image.numpy()




def PredictImage(image_path, model, topks, device,indexClass):
        print('start the prediction process')
        img_pre=ProcessImage(image_path)
        img_pre = Image.open(image_path)
        img_pre=torch.FloatTensor([img_pre])
        model.eval()
        output=model(img_pre.to(device))
        probability=torch.exp(output.cpu())
        top_p,top_c = probability.topk(topks,dim=1)
        top_class = [indexClass.get(x) for x in top_c.numpy()[0]]
        return top_p,top_class



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flowers Classifcation Trainer')
    parser.add_argument('--gpu', type=bool, default=False, help='Enable/Disable GPU')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture [available: densenet121, vgg16]')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='path to en existance  checkpoint',required=True)
    parser.add_argument('--top_k', type=int, default=5, help='top k classes for the input')
    parser.add_argument('--category_names', type=str, help='json path file of categories names of flowers')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units for  fc layer')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--save_dir' , type=str, default='./', help='checkpoint directory path')
    parser.add_argument('--input', type=str, help='path for image to predict')
    args = parser.parse_args()

    PredictFunctoin(args)
    print("... completed...Prediction\n")

 
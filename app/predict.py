import argparse
import logging
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json


def load(path):
    checkpoint = torch.load(path) #loading checkpoint from a file

    arch = checkpoint['arch']

    model = None

    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        raise Exception(f"Unsupported architecture in checkpoint. Supported alexnet and vgg13. Got: {arch}.")

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint ['state'])
    model.class_to_idx = checkpoint['mapping']
    
    for param in model.parameters(): 
        param.requires_grad = False 
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    mean_ = [0.485, 0.456, 0.406]
    std_dev_ = [0.229, 0.224, 0.225]

    #size = 256, 256
    im = Image.open(image)
    width, height = im.size
    
    if width > height: 
        height = 256
        im.thumbnail((10000, height), Image.ANTIALIAS)
    else: 
        width = 256
        im.thumbnail((width, 10000), Image.ANTIALIAS)
        
    
    width, height = im.size 
    target = 224
    left = (width - target)/2 
    top = (height - target)/2
    right = left + 224 
    bottom = top + 224
    im = im.crop((left, top, right, bottom))
    
    np_image = np.array(im)/255
    np_image -= np.array(mean_) 
    np_image /= np.array(std_dev_)
    
    np_image= np_image.transpose ((2,0,1))
    return np_image

def predict(image_path, model, topk, device):
    image = process_image(image_path)    
    im = torch.from_numpy(image).type(torch.FloatTensor)
    im = im.unsqueeze(dim = 0)
    im = im.to(device)

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model.forward(im)
        
    output_prob = torch.exp(output).to('cpu')
    
    probs, indices = output_prob.topk(topk)
    probs = probs.numpy()
    indices = indices.numpy() 
    
    probs = probs.tolist()[0]
    indices = indices.tolist()[0]
    
    mapping = {val: key for key, val in model.class_to_idx.items()}
    
    classes = [mapping[item] for item in indices]
    classes = np.array(classes)
    
    return probs, classes

def get_device(gpu):
    if gpu and torch.cuda.is_available():
        return "cuda"

    return "cpu"

def main(args):
    model = load(args.checkpoint)
    device = get_device(args.gpu)
    probs, classes = predict(args.image, model, args.top_k, device)

    with open(args.category_names) as f:
        mapping = json.load(f)

        results = {mapping[c]: probs[i] for i, c in enumerate(classes)}
        
        print("Prediction:")
        for k in results:
            print(f"{k} - {results[k]*100.0:0.5f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='prediction app')
    parser.add_argument('image', action='store', help='Path to image')
    parser.add_argument('checkpoint', action='store', help='path to checkpoint file')
    parser.add_argument('--top_k', action='store', dest='top_k', required=False, 
                        default=3, type=int, help='likely classes to return')
    parser.add_argument('--category_names', action='store', dest='category_names', required=False,
                        default="cat_to_name.json", help='mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', dest='gpu', required=False,
                        default=False, help='specify for gpu acceleration')

    main(parser.parse_args())
#!/usr/bin/env python
# coding: utf-8

# In[1]:


MATRIC_NUM = 'A0076918N'
import sys
import os.path as osp
from google.colab import drive
drive.mount('/content/drive')
ROOT = osp.join('/content', 'drive', 'My Drive', 'CS5260')
sys.path.append(osp.join(ROOT, MATRIC_NUM))


# In[2]:


import torch
if torch.cuda.is_available():
  print("GPU is available.")
  device = torch.device('cuda')
else:
  print("Change runtime type to GPU for better performance.")
  device = torch.device('cpu')


# In[3]:


# import libraries here, modify as you like.
import numpy as np
import matplotlib.pyplot as plt
from random import randint, uniform
from torch.autograd.gradcheck import zero_gradients
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import requests, io
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd 
import torch.autograd as autograd #grad
import torch.utils.data as utils #Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.

import matplotlib.pyplot as plt

from torch.autograd import Variable #Extending torch.autograd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils import data
from os import makedirs
import torchvision
from PIL import Image
import sys
import copy

import glob
import re
import imageio
#from scipy.misc import imread
from matplotlib.pyplot import imread
from matplotlib.colors import Normalize
import os
submission_path = os.path.join(ROOT, "A0076918N")
os.chdir(submission_path)
#%cd A0076918N
get_ipython().system('pwd')
get_ipython().system('ls')


# In[4]:


from load_model_36 import load_model  # XX is the digit for python number, e.g. 37
print(help(load_model))  # show the docstring of load_model function
model_path = os.path.join(ROOT, "model") 
model_file=model_path + '/model.pt'
print(model_file)
model = load_model(model_file, 'cuda') # change cuda to cpu to load model to CPU
model.eval()
model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[5]:


batch_size = 1
class Net(nn.Module):
    def __init__(self,num_classes=4):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=64, num_layers=3, dropout=0.1)
        #self.lstm = nn.LSTM(input_size=4, hidden_size=64, num_layers=3, dropout=0.1,bidirectional=True)
        self.fc1 = nn.Linear(64, 256)
        #self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc = nn.Linear(in_features=128, out_features=num_classes)
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, n_samples=batch_size):
        self.hidden = self.init_hidden(n_samples)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #lstm_out, _ = pad_packed_sequence(lstm_out)
        x = lstm_out[-1]
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
       
        x = self.fc(x)
        #print(x)
        #x = self.softmax(x)
                         
        return x
    
    def init_hidden(self, n_samples):
            return(autograd.Variable(torch.randn(3, n_samples, 64)).to(device), autograd.Variable(torch.randn(3, n_samples, 64)).to(device))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_lstm = Net() 
model_lstm.cuda()
model_lstm =torch.load('lstm_mix_100_all_add1.pth')
model_lstm.eval()


print(model_lstm.eval())


# In[6]:


images_dir = os.path.join(ROOT, "images")
results_dir = os.path.join(ROOT, "results")

#lean_data_dir = 'test/original'
#adv_data_dir = 'test/adversarial'
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
#clean_data_dir = 'data/clean_images/clean_images'
#adv_data_dir = 'data/adv_images/adv_images'


#output the prediction results
#to be used later

def pixel_deflection_without_map(img, deflections, window):
    img = np.copy(img)
    H, W, C = img.shape
    while deflections > 0:
        #for consistency, when we deflect the given pixel from all the three channels.
        for c in range(C):
            x,y = randint(0,H-1), randint(0,W-1)
            while True: #this is to ensure that PD pixel lies inside the image
                a,b = randint(-1*window,window), randint(-1*window,window)
                if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
            # calling pixel deflection as pixel swap would be a misnomer,
            # as we can see below, it is one way copy
            img[x,y,c] = img[x+a,y+b,c] 
        deflections -= 1
    return img

def predict_image(image):    
    
    img= Image.open(image)
    preprocess = transforms.Compose ([
                    transforms.Resize(128),
                    transforms.CenterCrop(128),
                    transforms.ToTensor(),
                    normalize,])
    image_tensor = preprocess(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    img_variable1 = Variable(image_tensor, requires_grad=True)
    img_variable = img_variable1.to(device)
    output = model(img_variable)
    index1 = output.data.cpu().numpy().argmax() 
    return index1, output


label_dict = {0:'artifacts', 1:'cancer_regions' , 2:'normal_regions',3:'other'}
get_ipython().system('ls')
get_ipython().system('pwd')


# In[ ]:



def lstm(image, N, j):
    preprocess = transforms.Compose([transforms.Resize(128),
                                      transforms.CenterCrop(128),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
                                     ])
    label_dict = {0:'artifacts', 1:'cancer_regions' , 2:'normal_regions',3:'other'}
    x_prob = []
    X = np.zeros((1, 100, 4))
    for i in range(N):
         img_deflected = pixel_deflection_without_map(img, deflections=5000, window=10)
         filename = "test"
         i= str(i)
         imageio.imwrite(results_dir + '/' + filename + i +  '.png', img_deflected)
         img_pd =  Image.open(results_dir + '/' + filename + i  + '.png')
         image_tensor = preprocess(img_pd).float()
         os.remove(results_dir + '/' + filename + i  + '.png')
         image_tensor = image_tensor.unsqueeze_(0)
         img_variable1 = Variable(image_tensor, requires_grad=True)
         img_variable = img_variable1.to(device)
         output = model(img_variable)
         index = output.data.cpu().numpy().argmax() 
         x_pred = label_dict[index]
         #prep for input data to lstm model
         output_probs = F.softmax(output, dim=1)
         x=(Variable(output_probs).data).cpu().numpy()
         x=x.reshape(1, 4)
         x_prob=np.append(x_prob,x)
    x_prob=np.round(x_prob.reshape(100,4),4)
    np.save(filename, x_prob)
    x = np.load(filename +'.npy')
    X[0,:, :] = x[:, :]
    print("making prediction...")
    tensor_X = torch.Tensor(X)
    data = tensor_X.to(device)
    data = data.permute(1,0,2)
    output_lstm = model_lstm(data,n_samples=batch_size)
    output_index = output_lstm.data.cpu().numpy().argmax()
    output_pred = label_dict[output_index]
    os.remove(filename +'.npy')
    return output_index, output_pred


def voting(image, N, j):
    count0 =0
    count1 =0
    count2 =0
    count3 =0
    preprocess = transforms.Compose([transforms.Resize(128),
                                      transforms.CenterCrop(128),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
                                     ])
   
    for i in range(N):
 
            img_deflected = pixel_deflection_without_map(img, deflections=5000, window=10)
            #img_deflected = pixel_deflection_with_map(img, rcam_prob, deflections=1000, window=10)
            #img_deflected_denoised = denoiser(img_deflected)
            #plt.imshow(img_deflected)

            #print(img_deflected)
            filename = "test"
            i= str(i)
            imageio.imwrite(results_dir + '/' + filename + i +  '.png', img_deflected)

           #imageio.imwrite('temp'+ '/' + filename + i +  '.png', img_deflected_denoised)
            img_pd =  Image.open(results_dir + '/' + filename + i  + '.png') 
            #print(img_pd)

            image_tensor = preprocess(img_pd).float()
            os.remove(results_dir + '/' + filename + i  + '.png')
            #image_tensor = preprocess(img1)
            image_tensor = image_tensor.unsqueeze_(0)
            img_variable1 = Variable(image_tensor, requires_grad=True)
            img_variable = img_variable1.to(device)
            output = model(img_variable)
            index = output.data.cpu().numpy().argmax() 
            x_pred = label_dict[index]
          #  X_PRED[j] = np.append(x_pred, )

            j=j+1

            if x_pred == "artifacts":
                count0 = count0 + 1 
            elif x_pred == "cancer_regions":
                count1 = count1 + 1 
            elif x_pred == "normal_regions":
                count2 = count2 + 1 
            elif x_pred == "other":
                count3 = count3 + 1 


            #get probability dist over classes
            output_probs = F.softmax(output, dim=1)
            x_pred_prob =  round((torch.max(output_probs.data, 1)[0][0]).item() * 100,4)
            #print(x_pred, x_pred_prob)

    count = [count0,count1,count2,count3]
    output_index = count.index(max(count))
    output_pred = label_dict[output_index]
    return output_index, output_pred


# In[10]:



os.chdir(results_dir)
get_ipython().system('pwd')
import time
i = 0
correct = 0


folders = ('artifacts', 'cancer_regions', 'normal_regions', 'other')
pattern = '{}/*/*.png'.format(images_dir, ''.join(folders))

for filename in glob.glob(pattern):
   start = time.time()
   #print(filename)
   
   img= Image.open(filename) 
   #output_index, output_pred = voting(img, N=100, j=0)
   output_index, output_pred = lstm(img, N=100, j=0)
   #print(output_index, output_pred)
  # labels = filename.split('/')[6]
  #  i=i+1
  # if output_pred == labels:
  #     print(output_index, output_pred)
  #     correct = correct + 1 
   #else:
    #   print(filename, x_pred, x_pred_prob)
  # acc = correct/i
   
   #infer result to file
   output_index = str(output_index)
   filename = filename.split('/')[7].replace('.png','')
   filename = str(filename)
   
   file1 = open("A0076918N.txt","a") 
   i=i+1
   if i == 1:
       file1.write(filename + "#" +output_index)
   
   else:
       file1.write('\n' + filename + "#" +output_index)
   file1.close()  
   stop = time.time()
   duration = stop-start
   print("Image"+ str(i)+ " takes", str(round(duration, 4))+"s.", "Result is "+ filename + "#" +output_index) 


#print(acc)




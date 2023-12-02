import os
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms, models
import torch.nn.functional as F
class NonLinearVAE(nn.Module):
    def __init__(self,input_channel=1,Leaky=1.02,fc_list=[128*25*25,1024,521,128,32,16],ptype='Skew',component=3):
        """
        0424 Deeper channel VAE
        """
        super(NonLinearVAE,self).__init__()
        self.input_channel=input_channel
        self.Leaky=Leaky
        self.fc_layer_num_list=fc_list
        self.ptype=ptype
        self.component=component
        self.Encoder_generating()
        self.Decoder_generating()

    def Encoder_generating(self):
        """
        Manually assign the structure of encoder for each layer
        """
        Encoder_CNN=nn.Sequential(
                        nn.Conv2d(self.input_channel, 16, 3, stride=1, padding=1),\
                        nn.LeakyReLU(self.Leaky),\
                        nn.Conv2d(16, 32, 3, stride=1, padding=1),\
                        nn.LeakyReLU(self.Leaky),\
                        nn.Conv2d(32, 64, 3, stride=1, padding=1),\
                        nn.LeakyReLU(self.Leaky),\
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),\
                        nn.Conv2d(64, 128, 3, stride=1, padding=1),\
                        #nn.BatchNorm2d(256),
                        nn.Tanhshrink(),\
                        nn.Conv2d(128, 128, 3, stride=1, padding=1),\
                        nn.Tanhshrink(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                        )
        #Encoder FC
        #Heaver option
        encoder_4_VAE_list=self.fc_layer_num_list.copy()
        if self.ptype=='GMM':
            print('You are using GM-VAE')
            encoder_4_VAE_list[-1]=(3*self.component)*encoder_4_VAE_list[-1]
        else:
            print('You are using SkewNormal-VAE')
            encoder_4_VAE_list[-1]=3*encoder_4_VAE_list[-1]
        Encoder_fc=nn.Sequential()
        for idx,i in enumerate(encoder_4_VAE_list):
            if idx+1!= len(encoder_4_VAE_list):
                Encoder_fc.add_module(str(idx),nn.Linear(i,encoder_4_VAE_list[idx+1]))
                Encoder_fc.add_module('Activate'+str(idx),nn.LeakyReLU(self.Leaky))
        self.Encoder_CNN=Encoder_CNN
        self.Encoder_fc=Encoder_fc

    def Decoder_generating(self):
        Decoder_CNN=nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Tanhshrink(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 3, stride=1, padding=1,dilation=1),
                nn.Tanhshrink(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128,64, 3, stride=1, padding=1,dilation=1),
                nn.LeakyReLU(self.Leaky),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 32, 3, stride=1, padding=1,dilation=1),
                nn.LeakyReLU(self.Leaky),
                nn.Conv2d(32, 16, 3, stride=1, padding=1,dilation=1),
                nn.LeakyReLU(self.Leaky),
                nn.Conv2d(16, self.input_channel, 3, stride=1, padding=1,dilation=1),
                )
        self.Decoder_CNN=Decoder_CNN
        #Decoder FC
        Decoder_fc=nn.Sequential()
        j=0
        for i in range(len(self.fc_layer_num_list)-1,-1,-1):
            if (i==1):#|(i==2):
                Decoder_fc.add_module(str(j),nn.Linear(self.fc_layer_num_list[i],self.fc_layer_num_list[i-1]))
                Decoder_fc.add_module('Activate'+str(j),nn.Tanhshrink())                
            elif i != 0:
                Decoder_fc.add_module(str(j),nn.Linear(self.fc_layer_num_list[i],self.fc_layer_num_list[i-1]))
                Decoder_fc.add_module('Activate'+str(j),nn.LeakyReLU(self.Leaky))
                Decoder_fc.add_module('BatchNorm'+str(j),nn.BatchNorm1d(self.fc_layer_num_list[i-1]))
            j+=1
        self.Decoder_fc=Decoder_fc
    def forward_encoder(self,X):
        '''
        X: the 4D image tensor
        '''
        #Forward of encoder
        out=self.Encoder_CNN(X)
        out=self.Encoder_fc(out.view(out.size(0),-1))
        if self.ptype=='GMM': #Incase of GMM
            # Assuming batch size of 8, GMM with 3 components, and 32-dimensional latent vector
            encoder_output = out.reshape(len(X), self.component, 3, self.fc_layer_num_list[-1])  
            # Chunk the encoder output into mixture weights, means, and standard deviations
            weights, means, log_var = torch.chunk(encoder_output, chunks=3, dim=2)
            # Apply softmax to the weights
            weights = F.softmax(weights, dim=1)
            # Apply softplus to the standard deviations
            stds = torch.exp(0.5 * log_var)
            stds = F.softplus(stds)
            # Reshape the chunks if needed
            weights = weights.squeeze(2)  # Shape: (8, 3, 32)
            means = means.squeeze(2)  # Shape: (8, 3, 32)
            stds = stds.squeeze(2)  # Shape: (8, 3, 32)
            # Apply the reparametrization trick with GMM
            epsilon = torch.randn_like(stds).to(stds.device)  # Sample from a standard normal distribution
            z = torch.sum(weights * (means + stds * epsilon), dim=1)
            #print('GMM process')
            return means,stds,weights,z
        else:#In case of the skew-normal
            #Split to mean and standard deviation
            mean,log_var,alp=torch.chunk(out,3,1)
            stds = torch.exp(0.5 * log_var)
            stds = F.softplus(stds)#The softplus should be added here!
            epsilon=torch.randn_like(log_var).to(stds.device)
            z=mean+stds*epsilon+alp*torch.abs(epsilon)
            #print('Skew-normal process')
            return mean,stds,alp,z
    def forward(self,X):
        if self.ptype=='GMM':
            #print('GM forward')
            means,stds,weights,z=self.forward_encoder(X)
            X_hat=self.Decoder_CNN(self.Decoder_fc(z).reshape(-1,128,25,25))
            return [means,stds,weights],z,X_hat
        else:
            #print('Skew forward')
            mean,stds,alp,z=self.forward_encoder(X)
            X_hat=self.Decoder_CNN(self.Decoder_fc(z).reshape(-1,128,25,25))
            return [mean,stds,alp],z,X_hat
    
    def forward_decoder(self,z):
        return self.Decoder_CNN(self.Decoder_fc(z).reshape(-1,128,25,25))
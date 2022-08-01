#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:03:06 2019

@author: raiv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

class VisionLstm_SeqtoOne_Module(nn.Module):

    def __init__(self, model_params):
#         __init__(self, num_features, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2,seq_len=1):
        super(VisionLstm_SeqtoOne_Module, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers)

        # Define the output layer
        lstm_out_size = 32 # for Sequence to One
        
        self.linear = nn.Linear(self.hidden_dim, lstm_out_size)
        
        self.vision_model= models.resnet18(pretrained=True)
        num_ftrs = self.vision_model.fc.in_features
        vision_out_size=32
       # print ("vision num feature",num_ftrs) 512 for resnet
        self.vision_model.fc = nn.Linear(num_ftrs, vision_out_size)
        
        self.output_dim=1
        self.last_linear=nn.Linear(lstm_out_size+vision_out_size,self.output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, kinematics,userStats,frames):
        # Forward pass through LSTM layer
        # shape of i/p: (seq_len, batch, input_size)
        # shape of lstm_out: [seq_len, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        
        kinematics=kinematics.permute(1,0,2)

        print ("input shape",kinematics.shape,frames.shape)
        lstm_out_seq, self.hidden = self.lstm(kinematics)
        
        
        vision_out=self.vision_model(frames)
        
        print ("lstm out shape",lstm_out_seq.shape,vision_out.shape ) #[seq_len,batch_size,hid_dim]
        
        # Only take the output from the final timetep
        
        print ("lstm o/p last timestep", lstm_out_seq[-1].shape) #[batch_size,hid_dim]
        lstm_out = self.linear(lstm_out_seq[-1])
        
        print ('Lstm fc shape', lstm_out.shape) #
        
        lstm_vision_out=torch.cat((lstm_out,vision_out),1)
        
        print ("lstm_vision_out",lstm_vision_out.shape)
        
        y_pred=self.last_linear(lstm_vision_out)
        
        print ("y pred",y_pred.shape)
        
        return  y_pred.view(-1,self.output_dim)
    
    
class VisionLstm_SeqtoTwo_Module(nn.Module):

    def __init__(self, model_params):
#         __init__(self, num_features, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2,seq_len=1):
        super(Lstm_SeqtoTwo_Module, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.output_dim = 2 # for Sequence to bi variate
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.vision_model= models.resnet18(pretrained=True)
        num_ftrs = vision_model.fc.in_features
        vision_model.fc = nn.Linear(num_ftrs, 1)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, kinematics,userStats,frames):
        # Forward pass through LSTM layer
        # shape of i/p: (seq_len, batch, input_size)
        # shape of lstm_out: [seq_len, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        
        kinematics=kinematics.permute(1,0,2)

        print ("input shape",kinematics.shape,frames.shape)
        lstm_out, self.hidden = self.lstm(kinematics)
        
        
        vision_out=self.vision_model(frames)
        
        # print ("lstm out shape",lstm_out.shape ) [seq_len,batch_size,hid_dim]
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        
        # print ("lstm o/p last timestep", lstm_out[-1].shape) [batch_size,hid_dim]
        y_pred = self.linear(lstm_out[-1])
        
        #print ('y_pred.shape', y_pred.shape) #
        return  y_pred.view(-1,self.output_dim)

    
    def loss_reg_fn(preds,targets):
        
        mse_loss=torch.nn.MSELoss(reduction='mean');
        
        return mse_loss(preds,targets)    
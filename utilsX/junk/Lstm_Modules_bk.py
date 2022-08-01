#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:03:06 2019

@author: raiv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import device_utils
device = device_utils.getCudaDevice()

class Lstm_SeqtoOne_Module(nn.Module):

    def __init__(self, model_params):
#         __init__(self, num_features, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2,seq_len=1):
        super(Lstm_SeqtoOne_Module, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        self.target_len= model_params['target_len']

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers)
       


        # Define the output layer
        self.output_dim = 1 # for Sequence to One
        self.linear = nn.Linear(self.hidden_dim,self. output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input_seq,y_history,batchY, phase):
        # Forward pass through LSTM layer
        # shape of i/p: (seq_len, batch, input_size)
        # shape of lstm_out: [seq_len, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        
        input_seq=input_seq.permute(1,0,2) # seq length first, batch, input_features
        #print ("input shape",input.shape)
        lstm_out, self.hidden = self.lstm(input_seq)
        
        # print ("lstm out shape",lstm_out.shape ) [seq_len,batch_size,hid_dim]
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        
        # print ("lstm o/p last timestep", lstm_out[-1].shape) [batch_size,hid_dim]
        y_pred = self.linear(lstm_out[-1])
        
        #print ('y_pred.shape', y_pred.view(-1).shape) #
        return  y_pred.unsqueeze(1).view(-1,1,self.output_dim)
    
    def setOptimizer(self,optimizer):
        self.adamOpt= optimizer
        
    def zeroGrad(self):
        self.adamOpt.zero_grad()
        
    def stepOptimizer(self):
         self.adamOpt.step()
         
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
       

class LstmHistory_SeqtoOne_Module(nn.Module):

    def __init__(self, model_params):
#         __init__(self, num_features, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2,seq_len=1):
        super(LstmHistory_SeqtoOne_Module, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        self.target_len= model_params['target_len']

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.num_features+1, self.hidden_dim, self.num_layers) # +1 for y history 
       


        # Define the output layer
        self.output_dim = 1 # for Sequence to One
        self.linear = nn.Linear(self.hidden_dim,self. output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input_seq,y_history,batchY, phase):
   
        last_y_history=y_history[:,self.seq_len-2,:].unsqueeze(2)
        y_history=torch.cat((y_history,last_y_history),axis=1)
       
        input_seq=torch.cat((input_seq,y_history),dim=2)
        input_seq=input_seq.permute(1,0,2) # seq length first, batch, input_features
        #print ("input shape",input.shape)
        lstm_out,hidden = self.lstm(input_seq)
        
        # print ("lstm out shape",lstm_out.shape ) [seq_len,batch_size,hid_dim]
                
        # print ("lstm o/p last timestep", lstm_out[-1].shape) [batch_size,hid_dim]
        y_pred = self.linear(lstm_out[-1])
        
        #print ('y_pred.shape', y_pred.view(-1).shape) #
        return  y_pred.unsqueeze(1).view(-1,1,self.output_dim)
    
    def setOptimizer(self,optimizer):
        self.adamOpt= optimizer
        
    def zeroGrad(self):
        self.adamOpt.zero_grad()
        
    def stepOptimizer(self):
         self.adamOpt.step()
    
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
        elif phase=='train':
            self.train()
         
class Lstm_SeqtoTwo_Module(nn.Module):

    def __init__(self, model_params):
#         __init__(self, num_features, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2,seq_len=1):
        super(Lstm_SeqtoTwo_Module, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        self.target_len= model_params['target_len']

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.output_dim = 2 # for Sequence to bi variate
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input_seq,y_history,batchY, phase):
  
        
        input_seq=input_seq.permute(1,0,2) # seq length first, batch, input_features
        #print ("input shape",input.shape)
        lstm_out, self.hidden = self.lstm(input_seq)
        

        # print ("lstm o/p last timestep", lstm_out[-1].shape) [batch_size,hid_dim]
        y_pred = self.linear(lstm_out[-1])
        
        #print ('y_pred.shape', y_pred.shape) #
        return  y_pred.view(-1,self.output_dim)
   
    def setOptimizer(self,optimizer):
        self.adamOpt= optimizer
        
    def zeroGrad(self):
        self.adamOpt.zero_grad()
        
    def stepOptimizer(self):
         self.adamOpt.step()
    
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
        elif phase=='train':
            self.train()
class LstmHistory_SeqtoTwo_Module(nn.Module):

    def __init__(self, model_params):
#         __init__(self, num_features, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2,seq_len=1):
        super(LstmHistory_SeqtoTwo_Module, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.num_features+2, self.hidden_dim, self.num_layers) # +1 for y history 
       


        # Define the output layer
        self.output_dim = 2 # for Sequence to 2
        self.linear = nn.Linear(self.hidden_dim,self. output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input_seq,y_history,batchY, phase):
   
        
        input_seq=torch.cat((input_seq,y_history),dim=2)
        input_seq=input_seq.permute(1,0,2) # seq length first, batch, input_features
        #print ("input shape",input.shape)
        lstm_out, self.hidden = self.lstm(input_seq)
        
        # print ("lstm out shape",lstm_out.shape ) [seq_len,batch_size,hid_dim]
                
        # print ("lstm o/p last timestep", lstm_out[-1].shape) [batch_size,hid_dim]
        y_pred = self.linear(lstm_out[-1])
        
        #print ('y_pred.shape', y_pred.view(-1).shape) #
        return  y_pred.unsqueeze(1).view(-1,1,self.output_dim)
    
    def setOptimizer(self,optimizer):
        self.adamOpt= optimizer
        
    def zeroGrad(self):
        self.adamOpt.zero_grad()
        
    def stepOptimizer(self):
         self.adamOpt.step()
    
    def set_phase(self,phase):
        if phase =='val' or phase =='test':
            self.eval()  
        elif phase=='train':
            self.train()
            
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss    
    
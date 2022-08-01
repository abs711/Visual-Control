# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 23:33:40 2020

@author: vijet
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:03:06 2019

@author: raiv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
debugPrint=1;



class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    
    


class AttnEncoderDecoder_Ankle(nn.Module):
    
    def __init__(self, encoder, decoder,model_params):
        super(AttnEncoderDecoder_Ankle, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.batch_size = model_params['batch_size']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_seq):
       
       encoder_output, encoder_hidden = self.encoder(input_seq)
       
       if  debugPrint: print ("encoder output shape",encoder_output[0].shape,encoder_hidden[0].shape)

       #decoder_input = torch.tensor([[0.0 for _ in range(self.batch_size)]],dtype=torch.long, device=self.device)
       decoder_input = torch.tensor([[0.0]] * self.batch_size, dtype=torch.float,device=self.device)

    
       # Set initial decoder hidden state to the encoder's final hidden state
       decoder_hidden = encoder_hidden#[:self.decoder.num_layers]
       decoded_targets_list=[];
    
       for di in range(self.decoder.target_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden,encoder_output) 
            
            decoded_targets_list.append(decoder_output)
            decoder_input=decoder_output #hidden state becomes input ? or predicted out 

       decoded_target_seq=torch.stack(decoded_targets_list,dim=1)
       
       if debugPrint: print ("final decoded target shape",decoded_target_seq.shape )
       return decoded_target_seq

    def setOptimizer(self,enc_opt, dec_opt):
        self.encoder_optimizer=enc_opt
        self.decoder_optimizer=dec_opt
        
    def zeroGrad(self):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            
    def stepOptimizer(self):
             self.encoder_optimizer.step()
             self.decoder_optimizer.step()  
             
class Encoder_Ankle(nn.Module):

    def __init__(self, model_params):
#         __init__(self, num_features, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2,seq_len=1):
        super(Encoder_Ankle, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.seq_len= model_params['seq_length']
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers)


    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input_seqs):
        if debugPrint: print ("encoder input", input_seqs.shape)
        input_seqs=input_seqs.permute(1,0,2) # seq length first, batch, input_features
        encoder_out, enc_hidden = self.lstm(input_seqs)
        
        
        return encoder_out, enc_hidden


class LuongAttnDecoder_Ankle(nn.Module):
    def __init__(self, attn_model, model_params):
        super(LuongAttnDecoder_Ankle, self).__init__()
        self.num_features =model_params['num_features']
        self.hidden_dim = model_params['num_hid_units']
        self.batch_size = model_params['batch_size']
        self.num_layers = model_params['num_layers']
        self.target_len = model_params['target_len']
        
        
        self.attn_model = attn_model
        self.attn = Attn(attn_model, self.hidden_dim)

        self.input_FC=nn.Linear(1,self.hidden_dim)
        self.LSTM = nn.LSTM(self.hidden_dim, self.hidden_dim,self.num_layers)
        self.concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim, 1) # single step output
        

    def forward(self, input_batch, last_hidden,encoder_outputs):
        
        if debugPrint: print ("Dec: batch input ",input_batch.shape)

        #input_batch=input_batch.transpose(0,1).long() #shape needed [batch, input_shape=1]
        
        #if debugPrint: print ("Dec: transpose batch",input_batch.shape)

        input_step=self.input_FC(input_batch) # shape needed [batch, hidden_dim]
        input_step=input_step.unsqueeze(0) # shape needed [seq_len,batch,hidden_dim]

        if debugPrint: print ("Dec: LSTM in hidden shape",input_step.shape)
        

        dec_outs, dec_hidden = self.LSTM(input_step, last_hidden)
       
        if debugPrint: print ("attn inputs",dec_outs.shape,encoder_outputs.shape  )
        attn_weights = self.attn(dec_outs, encoder_outputs)
        
        if debugPrint: print ( "attn output",attn_weights.shape)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        
        if debugPrint: print ("context shape", context.shape)
         
        
        dec_outs= dec_outs.squeeze(0)
        
        context = context.squeeze(1)
        concat_input = torch.cat((dec_outs, context), 1)
        
        if debugPrint: print ("Dec: context concat inputs ",dec_outs.shape,context.shape,concat_input.shape)

        concat_output = torch.tanh(self.concat(concat_input))        
        if debugPrint: print ("concat out shape", concat_output.shape)
          
        decoder_target = self.out(concat_output)   
        if debugPrint: print ("Dec: target  shape",decoder_target.shape)
        
        
        
        return decoder_target, dec_hidden
    
         
        

#    def initHidden(self):
#        #return torch.zeros(1, 1, self.hidden_size, device=device)   
#        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
#                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
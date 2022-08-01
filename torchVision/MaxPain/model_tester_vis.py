
import sys
sys.path.append("../../utilsX")
import torch
import torch.nn as nn
import datamanager_v7 as dm

from datetime import datetime
import os

from torch_utils import *
from log_utils import *
from train_eval_functions import vis_evaluate, evaluate
from loss_modules import RMSELoss
from VisionKinematicsDataset import *
from KinematicsDataset import *
import runner_v8 as runner

import numpy as np
from itertools import product
from collections import defaultdict
from collections import OrderedDict
from math import sqrt 

timestamp= datetime.now().strftime("%m_%d_%H_%M")


    

if __name__ == "__main__":

    data_dir = '/home/spc/pytorch_challenge-master/data/vision_kinematics/Unstructured_data/Unstructured_Data/'

    ZV = True
    trial = "000"
    subject = "006"
    tr = ''
    if trial == "000":
        trial_id = "comb"
    else:
        trial_id = "_"+str(trial)

    config_file_name = "modelTest_xUD"+str(subject)+trial_id+".csv"
    
    expt_path = "/home/spc/pytorch_challenge-master/DL_Code/torchVision/MaxPain/models1"
    #for par in range(1,4):
    #    for trl in range(5):
            
    if ZV == True:

        model_path = expt_path + '/checkpoints_zv/checkpoint_param_1_trial_3.pt'#'/BestOF_full2exclude_2_19_17_08_ZV.pt'#'/checkpoints_zv/checkpoint_param_1_trial_3.pt'#'/BestOF_full2exclude_2_19_17_08_ZV.pt'#'/BestOF_full2exclude_2_19_17_08_ZV.pt'#BestOF_full2exclude_2_18_6_28.pt'#'/BestKine_full2exclude_2_18_13_13.pt'#
    else:
        model_path = expt_path + '/checkpoints_of/checkpoint_param_1_trial_4.pt'#'/BestOF_full2exclude_2_18_6_28.pt'#'/checkpoints_of'+tr+'/checkpoint_param_1_trial_4.pt'#

    model_type = 'OpticalFlow4Prosthetics'
    #model_type = 'SeqToTwo'

    results_path= expt_path

    dm.set_results_dir_manual(results_path)


    param_vals = { 'num_epochs': (1000), 'batch_size':(1000), 'minibatch_GD': True, 'num_trials':(3), 'shuffle': True,
                  
                   'seq_length':(10),'target_len':(1), 'pred_horizon':(30), # THESE ARE FIXED IN THE DATAMANAGER. NO LIST.
                   
                   'num_layers':(2),'num_hid_units':(32), 'num_features':(57), 'num_vis_features':(64),'num_vis_layers':(2),'numscenesample':1,'num_classes': (8), 'weighted_classification': (True),
                   'num_fc_layers': (4),'fc_taper':(2), 
                   'pretrained_LSTM': False, 'backprop_kinematics': False, 'backprop_vision': False, 'zero_like_vision':ZV,
                   'zero_like_kine':False,

                   'window_step':(1),'sub_sample_factor':(1), 'frame_subsample_factor':(1),

                   'noise_std':(0.02),'clip':(-1),'dropout':(0.0), 'pretrainC2A_w_Dropout':True, 'regularization':(0),
                   'learning_rate': (1e-5), 'custom_lr_decay_rate':(0.1), 'custom_lr_decay_step':(90),
                   'use_scheduler': (True), 'plateau_schedular_patience':5, 'decay_factor': 0.1,'EarlyStoppingPatience':30, 'Early_Stopping_delta':0,'valid_patience':1,
                   'num_workers':(0),'pin_memory':False,'non_blocking':True, 'device':(0)}

    param_vals=OrderedDict(param_vals)
     

    print(param_vals.keys())

    #model_type='OpticalFlow4Prosthetics'##'SeqToTwo'## #['Conv','Vision','Scene','Optical','Spatiotemporal']
    use_features = True#False #True
    runner.SetTrainingOpts(sanityflag=False,use_single_batch=False,overlap_datatransfers=param_vals['non_blocking'],model_name=model_type,use_feats=use_features,indie_checkpoints=True)#,clipGradients,indie_checkpoints,vision_substrings,numscenesamples=paramvals['numscenesample'])
    scheduler_type = 'ReduceOnPlateau'    
    

    train_dict, test_dict, val_dict , uid_manager = dm.get_and_reformat_all_data(data_dir, config_file_name,param_vals)


    print ('test label shape',test_dict['y'].shape)
    print ('test frame shape',test_dict['x'].shape)


    model=getModule(model_type,param_vals) 
    criterion = torch.nn.MSELoss(reduction='mean')#RMSELoss()

    dataloaders= getDataloader(train_dict,test_dict,val_dict,model_type,param_vals, uid_manager)


    log_predictions=getLogger() 

    thisTrial_model_dict = torch.load(model_path,map_location='cuda:0')
    model.load_state_dict(thisTrial_model_dict)#['model_state_dict'])
       #optimiser.load_state_dict(bestModel_dict['optimiser_state_dict'])

    yTests,yPreds=vis_evaluate(model,'test',model_type,dataloaders['test'])
    #yTests,yPreds=evaluate(model,'test',model_type,dataloaders['test'])
    this_trial_rmse=criterion(yPreds.view(yTests.shape), yTests).item()

    print ("RMSE",this_trial_rmse)
       
    log_predictions['y_preds'].append(yPreds.cpu().detach().numpy());
    log_predictions['y_tests'].append(yTests.cpu().detach().numpy())
    log_predictions['trial_info'].append(param_vals)
    log_predictions['test_Loss'].append(this_trial_rmse)
       #log_predictions['Epoch vs Loss'].append(hist)
         
#    dm.save_as_mat([model_type + '/preds','predictions'],log_predictions) 
    if ZV == True:
        dm.save_as_mat([model_type + '/pred_xUD'+str(subject)+trial_id+'ZV'+tr,'predictions'],log_predictions) 
    else:
        dm.save_as_mat([model_type + '/pred_xUD'+str(subject)+trial_id+tr,'predictions'],log_predictions) 
    #dm.save_as_mat(['hyperparams','predictions_1testtrial'],log_predictions) 
    

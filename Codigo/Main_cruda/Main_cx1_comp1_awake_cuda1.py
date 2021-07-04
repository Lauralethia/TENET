# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:04:20 2021

@author: Amelie
"""
#%%#############################################################################
# Dependencias
#!pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib auto
import time
import torch
# Dependencias
import torch.nn as nn
from torch.utils.data import TensorDataset
import GPUtil as GPU
import sys
import os
functionspath = 'E:/tiempo/Codigo/Funciones/'
sys.path.insert(0, functionspath)
from Functions import (printm,see_data,data2spectrum,prepareData,FullTrain,compnames,loadCP)

#%%#############################################################################
# Cosas para editar
## Donde 
datapath = 'E:/tiempo/Datos/'
resultpath = 'E:/tiempo/Resultados/'
# en basepath deben estar los archivos, el script de funciones, 
## Que condiciones analizar
filename =  'awake'
#filename =  'keta'
#filename =  'sleep'
filename_reversal = filename + '_reversal'
# Complejidad del modelo, puede ser 3 (5 bloques), 2 (3 bloques) o 1 (1 Bloque)
# Cada bloque tiene 2 CConv
model_complexity = 1
## Que componentes usar
NC = [1]
CPcondition = 'MCx_' +str(model_complexity) + '_comp'+'_'.join([str(int) for int in NC] )
## Usar TF o CP sin transformar? 
UsarTF = False # Si es True CPcondition agrega TF al principio y Batch Size es 1800. Si no BS de 
## En que placa laburo?
PLACA = "cuda:1" # o "cuda"/"cuda:0" y ver lo de paralelizar 
#%%############################################################################
# Data transformation to TF chart
#sample_rate = 256
#time = see_data(datapath + filename + '_comp1',sample_rate)
if UsarTF:
    s_Forward = torch.cat([data2spectrum(sp) for sp in compnames(datapath + filename,NC)],dim=1) # NC tiene las componentes usadas
    s_Backward = torch.cat([data2spectrum(sp) for sp in compnames(datapath + filename_reversal,NC)],dim=1)
    
    #fig, (ax1,ax2) = plt.subplots(1,2)
    #ax1.imshow(s_Forward[15,:,:].numpy(), cmap='rainbow', aspect='auto',extent = [0 , time[1023],  sample_rate // 2,0])
    #ax2.imshow(s_Backward[15,:,:].numpy(), cmap='rainbow', aspect='auto',extent = [0 , time[1023],  sample_rate // 2,0])
    #fig.colorbar(img,orientation="horizontal")
    CPcondition = "TF_" + CPcondition 
    batch_size = 1800

else:
    s_Forward = torch.stack([loadCP(sp, Npuntos = True) for sp in compnames(datapath + filename,NC)]).permute(1,0,2) # NC tiene las componentes usadas
    s_Backward = torch.stack([loadCP(sp, Npuntos = True) for sp in compnames(datapath + filename_reversal,NC)]).permute(1,0,2)
    batch_size = 1800


data = torch.cat((s_Forward, s_Backward), 0)
data = data.type('torch.FloatTensor')
labels = torch.tensor([1]*s_Forward.shape[0] + [0]*s_Backward.shape[0]) # 1 es s_Forward, 0 es s_Backward
shufled_labels = labels[torch.randperm(labels.size()[0])]

data_set = TensorDataset(data, labels)
sh_data_set = TensorDataset(data, shufled_labels)

del s_Forward,s_Backward,data
#%% ##############################################################################
# Parametros de red, entrenamiento y salvado de datos

device = torch.device(PLACA if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)

if device == PLACA:
    num_workers = 3
    pin_memory = True
else:
    num_workers = 10
    pin_memory = False

wd = 1e-3
base_lr, top_lr, cycle_epochs = 5e-5, 5e-4, 100 
lr_increase = (top_lr/base_lr)**(1/cycle_epochs)
B_ciclos = 1
criterion = nn.BCELoss().to(device)

# Salvado de datos
ModelosFolder =  resultpath + CPcondition +'_Modelos/'
ImagenesFolder = resultpath + CPcondition +'_figures/'
if not os.path.exists(ModelosFolder):
    os.makedirs(ModelosFolder)

if not os.path.exists(ImagenesFolder):
    os.makedirs(ImagenesFolder)
    
savePath = ModelosFolder + filename 
sh_savePath = ModelosFolder + 'sh_' + filename 
savePath_ima = ImagenesFolder + filename 

column_names = ["Iteration", 'TL', 'VL', 'TA', 'VA', 'Tauc', 'Vauc', 'epoca'] # T = Training, V = Validation
iter_df = pd.DataFrame(columns = column_names)
#iter_df =  pd.read_csv(savePath + 'iter_df.csv')
sh_iter_df = pd.DataFrame(columns = column_names)
#sh_iter_df =  pd.read_csv(savePath + 'sh_iter_df.csv')
# iteraciones
for i in range(0,100):
#i = 1
    model, optimizer, train_loader, test_loader = prepareData(data_set,batch_size,base_lr,wd,device,model_complexity)
    #model = nn.DataParallel(model)
    #torch.cuda.set_device(0)
    #model.cuda(0)
    # training loop bien  
    df,fig = FullTrain(optimizer,B_ciclos,lr_increase,cycle_epochs,
                  criterion,model,train_loader,test_loader,device,
                  savePath +'It'+str(i),iter_df)

    #fig.show()
    #plt.pause(0.05) 
    fig.suptitle('ADAM:' ' base_lr=' + str(base_lr)  + ' wd=' + str(wd))
    fig.savefig(savePath_ima + '_It'+str(i)+'.png')   
    plt.close(fig)   
    df['epoca'] = df.index
    df.insert(0, "Iteration", [i] * len(df), True) 
    iter_df = iter_df.append(df,ignore_index=True)
    iter_df.to_csv(savePath + 'iter_df.csv' )
    
    #printm(GPU.getGPUs()[0])
    #torch.cuda.empty_cache()
    #printm(GPU.getGPUs()[0])

    # training loop shufleado (corre en cpu xq no me daba la GPU) 
    sh_model, sh_optimizer, sh_train_loader, sh_test_loader = prepareData(sh_data_set,batch_size,base_lr,wd,device)

    sh_df,sh_fig = FullTrain(sh_optimizer,B_ciclos,lr_increase,cycle_epochs,
                  criterion,sh_model,sh_train_loader,sh_test_loader,device,#device,
                  sh_savePath + 'It'+str(i),sh_iter_df)                      # save model 
    sh_fig.suptitle('Shuffled ADAM:' ' base_lr=' + str(base_lr)  + ' wd=' + str(wd))
    #sh_fig.show()
    #plt.pause(0.05) 
    sh_fig.savefig(savePath_ima + '_It'+str(i)+'_Shuffled.png')   # save figure 
    plt.close(sh_fig)   
    
    sh_df['epoca'] = sh_df.index
    sh_df.insert(0, "Iteration", [i] * len(sh_df), True) 
    sh_iter_df = sh_iter_df.append(sh_df,ignore_index=True)
    sh_iter_df.to_csv(savePath + 'sh_iter_df.csv' )              # save data
    plt.close()
    
# iter_df.loc[ iter_df[['VL']].idxmin()]
# iter_df.loc[np.where( [iter_df['VA'] == max(iter_df['VA'])][0] == True)]['VA']).values
#         g_i_auc =  (iter_df.loc[ iter_df[['Vauc']].idxmax()]['Vauc']).values

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:09:29 2021
Functions TENET
import Functions  
dir(Functions)
Models:
 'SConv',
 'SConv1D',
 'SConv1D_1',
 'SConv1D_2',
 
Re/Training:
 'asfunction_corridaSimple',exacutes  script 
 'FullTrain',
 'cycleTrain',
 'train',
 'trainfig',
 'trainfig_online',
 'valida' 
 'REtrainfig',
 'ReTrain_FullTrain',
 'ReTrain_cycleTrain',
 'retrain_best',

Data/models managment and preparation:
 'compnames',
 'data2spectrum',
 'loadCP',
 'loadDataDfame',
 'prepareData',
 'saveModel',
 'LoadModel', 
 
Feature Importance:
    
@author: Laura Alethia de la Fuente
"""
# Dependencias
import sys
import os, fnmatch
from os.path import isfile
import time
import glob
import psutil
import humanize

import pandas as pd
import numpy as np
import GPUtil as GPU

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset

from scipy import signal
from sklearn.metrics import roc_auc_score
from statistics import mean

import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients

#%%
def printm(gpu):
  process = psutil.Process(os.getpid())
  print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

def see_data(name,sample_rate):
   awake =  pd.read_csv(name + '.csv', header=None)
   waveform = torch.tensor(awake[:100].values).squeeze(dim= 0)
   time = np.arange(waveform.shape[1])/sample_rate

   plt.figure()
   plt.plot(time,waveform.t().numpy())
   print(waveform.shape)
   return time

def compnames(name,NC):
    NCnames = []
    for cp in NC:
           NCnames.append(name + '_comp' + str(cp))       
    return NCnames
# base = datapath + filename
# cp = 1
# name = base + '_comp' + str(cp)
def loadDataDfame(name,maxPropo = False):
    data = pd.read_csv(name + '.csv', header=None)
    if maxPropo:
        data = data.sample(n = 290, replace = False)
    return data

def data2spectrum(name,UsarTF = False, UsarFase = False,maxPropo = False):
  data = loadDataDfame(name,maxPropo)    

  waveform = torch.tensor(data.values).squeeze(dim= 0) # ACA solo toma 1000[:50]
  
  if UsarFase and UsarTF:
      specgram = torchaudio.transforms.Spectrogram(win_length = 32,hop_length =4)(waveform)#win_length = 10,hop_length =10,n_fft = 50
      spectogram_db_transform = torchaudio.transforms.AmplitudeToDB()
      spectogram_db = spectogram_db_transform(specgram)
      
      spect_ph = torchaudio.transforms.Spectrogram(win_length = 32,hop_length =4,power = None)(waveform)
      spect_pha = torch.angle(torch.complex(spect_ph[:,:,:,0],spect_ph[:,:,:,1]))
      spectogram_db = torch.cat([spect_pha,spectogram_db],dim=1)
  
  if UsarTF and (not UsarFase):
       specgram = torchaudio.transforms.Spectrogram(win_length = 32,hop_length =4)(waveform)#win_length = 10,hop_length =10,n_fft = 50
       spectogram_db_transform = torchaudio.transforms.AmplitudeToDB()
       spectogram_db = spectogram_db_transform(specgram)

  if UsarFase and (not UsarTF):
      spect_ph = torchaudio.transforms.Spectrogram(win_length = 32,hop_length =4,power = None)(waveform)
      spect_pha = torch.angle(torch.complex(spect_ph[:,:,:,0],spect_ph[:,:,:,1]))
      spectogram_db = spect_pha
  

  print("Shape of DB spectrogram or phase: {}".format(spectogram_db.size()))
  return spectogram_db

def loadCP(name,Npuntos = False,maxPropo = False):
    data = loadDataDfame(name,maxPropo)    
    if Npuntos==True :
        data = [signal.resample(row, 514,axis=0) for index, row in data.iterrows()]# va a 514 = 2*257 porque es en lo que estan los TF
        data = torch.tensor(data)
    else:
        data = torch.tensor(data.values)
    return data

def train(model, train_loader,optimizer,device,totalsamplesize,criterion,model_flag=False):
    model.train()
    #totalsamplesize =len(train_loader.batch_sampler.sampler)
    batch_loss,train_loss,train_accuracy = [],[],[]
    correct = 0
    y_test,y_pred = [],[]
    
    for batch_idx, (image, target) in enumerate(train_loader):
            if model_flag:
              image = image.unsqueeze(1).to(device)
            else:
              image = image.to(device)

            target = target.to(device)
            output = model(image)

            y_test.extend(target.detach().cpu().numpy())
            y_pred.extend(output.detach().cpu().numpy())

            loss = criterion(output.squeeze(),target.to(device).float())

            output = (output.squeeze()>0.5).float()
            correct += (output == target).float().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            batch_loss.append(loss.item())
    train_loss = mean(batch_loss)
    train_accuracy = 100 * (correct / totalsamplesize) 
    train_auc = roc_auc_score(y_test, y_pred) 
    return train_loss,train_accuracy.detach().cpu().numpy(),train_auc

def valida(model, train_loader,device,totalsamplesize,criterion,model_flag=False):
    model.eval()
    batch_loss,train_loss,train_accuracy = [],[],[]
    correct = 0
    y_test,y_pred = [],[]

    for batch_idx, (image, target) in enumerate(train_loader):
            if model_flag:
              image = image.unsqueeze(1).to(device)
            else:
               image = image.to(device)

            target = target.to(device)
            output = model(image)

            y_test.extend(target.detach().cpu().numpy())
            y_pred.extend(output.detach().cpu().numpy())
            
            loss = criterion(output.squeeze(),target.to(device).float())

            output = (output.squeeze()>0.5).float()
            correct += (output == target).float().sum()

                        # record loss
            batch_loss.append(loss.item())
    train_loss = mean(batch_loss)
    train_accuracy = 100 * (correct / totalsamplesize)  
    train_auc = roc_auc_score(y_test, y_pred) 

    return train_loss,train_accuracy.detach().cpu().numpy(),train_auc

def trainfig(df):
    # multiple line plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

            ax1.plot( 'TL', data=df, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
            ax1.plot( 'VL', data=df, marker='o', markerfacecolor='red', markersize=8, color='pink', linewidth=4)
            ax1.set_title('Loss')
            ax2.plot( 'TA', data=df, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
            ax2.plot( 'VA', data=df, marker='o', markerfacecolor='red', markersize=8, color='pink', linewidth=4)
            ax2.set_title('Accuracy')
            ax3.plot( 'Tauc', data=df, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
            ax3.plot( 'Vauc', data=df, marker='o', markerfacecolor='red', markersize=8, color='pink', linewidth=4)
            ax3.set_title('AUC')
            return fig
def trainfig_online(fig, ax1, ax2, ax3,df):
    # multiple line plot

            ax1.plot( 'TL', data=df, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
            ax1.plot( 'VL', data=df, marker='o', markerfacecolor='red', markersize=8, color='pink', linewidth=4)
            ax1.set_title('Loss')
            ax2.plot( 'TA', data=df, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
            ax2.plot( 'VA', data=df, marker='o', markerfacecolor='red', markersize=8, color='pink', linewidth=4)
            ax2.set_title('Accuracy')
            ax3.plot( 'Tauc', data=df, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
            ax3.plot( 'Vauc', data=df, marker='o', markerfacecolor='red', markersize=8, color='pink', linewidth=4)
            ax3.set_title('AUC')
            return fig
def cycleTrain(optimizer,scheduler,cycle_epochs,criterion,
               model,train_loader,test_loader,device,
               savePath,start_point,i_loss,i_acc,i_auc):
    
    train_loss_a,train_accuracy_a,train_auc_a,test_loss_a,test_accuracy_a,test_auc_a = [],[],[],[],[],[]
    
    optimizer.zero_grad()
    #fig_online, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for epoch in range(cycle_epochs):        
            # Entrenamiento
            print(f'epoch: {epoch}')
            tic = time.time()
            train_loss,train_accuracy,train_auc = train(model, train_loader,optimizer,device,len(train_loader.batch_sampler.sampler),criterion)
            test_loss,test_accuracy,test_auc = valida(model, test_loader,device,len(test_loader.batch_sampler.sampler),criterion)
            toc = time.time()
            print(toc - tic)
            train_loss_a.append(train_loss)
            train_accuracy_a.append(train_accuracy)
            train_auc_a.append(train_auc)
            test_loss_a.append(test_loss)
            test_accuracy_a.append(test_accuracy)
            test_auc_a.append(test_auc)
            
            scheduler.step() 

            df=pd.DataFrame({ 'TL': train_loss_a, 'VL': test_loss_a, 'TA':train_accuracy_a, 'VA':test_accuracy_a, 'Tauc':train_auc_a, 'Vauc':test_auc_a })
            
            # fig_online= trainfig_online(fig_online, ax1, ax2, ax3,df)
            # fig_online.show()
            # plt.pause(0.05)
            # Guarda los mejores modelos segun los 3 criterios (Loss, accuracy y AUC)
            if test_loss<i_loss:
                old = glob.glob(savePath +'_Loss'+ '*')
                for filePath in old:
                     os.remove(filePath)
                
                saveModel(epoch+start_point,model,optimizer,savePath + '_Loss'+'_E'+ str(epoch+start_point)+'_'+time.strftime("%Y%m%d-%H%M%S"))
                i_loss = test_loss

            if test_accuracy > i_acc:
                old = glob.glob(savePath +'_ACC'+ '*')
                for filePath in old:
                     os.remove(filePath)
                
                saveModel(epoch+start_point,model,optimizer,savePath + '_ACC'+ '_E'+ str(epoch+start_point) + '_'+time.strftime("%Y%m%d-%H%M%S"))
                i_acc = test_accuracy     
            
            if test_auc > i_auc:
                old = glob.glob(savePath +'_AUC'+ '*')
                for filePath in old:
                     os.remove(filePath)
                
                saveModel(epoch+start_point,model,optimizer,savePath + '_AUC'+ '_E'+ str(epoch+start_point) + '_'+time.strftime("%Y%m%d-%H%M%S"))
                i_auc = test_auc                          
                
            model.to(device)
    # plt.close(fig_online)      
    return (optimizer,df,i_loss,i_acc,i_auc)

def saveModel(epoch,net,optimizer,PATH):
    torch.save({
            'epoch': epoch,
            'model_state_dict': net.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
    return

def  prepareData(data_set,batch_size,base_lr,wd,device,model_complexity = 3):
    idx = list(range(len(data_set)))
    train_batch_size = 4*len(data_set)//5
    test_batch_size = len(data_set)-train_batch_size
    
    #train_idx, test_idx = random_split(idx, (train_batch_size, test_batch_size),generator=torch.Generator().manual_seed(42))
    train_idx, test_idx = random_split(idx, (train_batch_size, test_batch_size))

    elem_freq = int(data_set[1][0].shape[0])
    # Modelos
    if model_complexity == 3:
        if  elem_freq >= 200: # TF o Fase
            n_puntos,FCval = 4, 9600 # cuando pasa por ventaneado (TF o fase) N puntos chico
        else: # CP lineal
            n_puntos,FCval = 8, 9600 # cuando cruda N puntos grande
        # torch.manual_seed(42)
        # #or for cuda
        # torch.cuda.manual_seed(42)
        # torch.cuda.manual_seed_all(42)
        
        #model = SConv1D(in_features = elem_freq ,n_features = 50,p_bias = False,n_puntos = 8,DO= 0.4 ,FCval = 17600).to(device)# cambiando n features completa
        #model = SConv1D(in_features = elem_freq ,n_features = 50,p_bias = False,n_puntos = 8,DO= 0.4 ,FCval = 4800).to(device)# cambiando n features completa
        model = SConv1D(in_features = elem_freq ,n_features = 100,p_bias = False,n_puntos = n_puntos,DO= 0.4 ,FCval = FCval).to(device)# senial a 1/2
        #model = SConv1D(in_features = elem_freq ,n_features = 100,p_bias = False,n_puntos = 8,DO= 0.4 ,FCval = 35200).to(device)# todo
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr,betas=(0.9, 0.999), eps=1e-6, weight_decay=wd)    
    
    if model_complexity == 2:
        if  elem_freq >= 200: # TF o Fase
            n_puntos,FCval = 4, 12000 # cuando pasa por ventaneado (TF o fase) N puntos chico
        else:  # CP lineal
            n_puntos,FCval = 8, 22000 # cuando es directo N puntos mayor para compensar tamaño
       # torch.manual_seed(42)
       # #or for cuda
       # torch.cuda.manual_seed(42)
       # torch.cuda.manual_seed_all(42)
        model = SConv1D_2(in_features = elem_freq ,n_features = 100,p_bias = False,n_puntos = n_puntos,DO= 0.4 ,FCval = FCval).to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr,betas=(0.9, 0.999), eps=1e-6, weight_decay=wd)

    
    if model_complexity == 1:
        if  elem_freq >= 200: # TF o Fase
            n_puntos,FCval = 4, 12700 # cuando pasa por ventaneado (TF o fase) N puntos chico
        else:  # CP lineal
            n_puntos,FCval = 8, 25200 # cuando es directo N puntos mayor para compensar tamaño
        # torch.manual_seed(42)
        # #or for cuda
        # torch.cuda.manual_seed(42)
        # torch.cuda.manual_seed_all(42)         
        model = SConv1D_1(in_features = elem_freq ,n_features = 100,p_bias = False,n_puntos = n_puntos,DO= 0.4 ,FCval = FCval).to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr,betas=(0.9, 0.999), eps=1e-6, weight_decay=wd)
   
    #full_loader = TensorDataset(data_set, batch_size = len(data_set), pin_memory=True) 
    # full_loader agregado para maxim uso de placas
    train_loader = DataLoader(data_set, batch_size = batch_size, sampler = SubsetRandomSampler(train_idx), pin_memory=True) 
    test_loader = DataLoader(data_set, batch_size = batch_size,sampler = SubsetRandomSampler(test_idx), pin_memory=True)
 
    
    return   ( model, optimizer, train_loader, test_loader)

def FullTrain(optimizer,B_ciclos,lr_increase,cycle_epochs,
                  criterion,model,train_loader,test_loader,device,
                  savePath,iter_df):
    if iter_df.empty:
        g_i_loss,g_i_acc,g_i_auc = 1000,0,0
    else:
        g_i_loss = (iter_df.loc[ iter_df[['VL']].idxmin()]['VL']).values[0]                  
        g_i_acc =  (iter_df.loc[np.where( [iter_df['VA'] == max(iter_df['VA'])][0] == True)]['VA']).values[0]  
        g_i_auc =  (iter_df.loc[ iter_df[['Vauc']].idxmax()]['Vauc']).values[0]  


    df = pd.DataFrame(columns=['TL', 'VL', 'TA', 'VA', 'Tauc', 'Vauc'])

    optimizer.zero_grad()
    for repeat in range(B_ciclos):
        print(f'Ciclo: {repeat}')    
        # UP
        start_point = repeat * (cycle_epochs*2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_increase) 
                                        
        optimizer,dfU,i_loss,i_acc,i_auc = cycleTrain(optimizer,scheduler,cycle_epochs,criterion,
                                      model,train_loader,test_loader,device,
                                      savePath,start_point,g_i_loss,g_i_acc,g_i_auc) 
        g_i_loss = min(i_loss,g_i_loss) # updatea a los mejores para guardar
        g_i_acc = max(i_acc,g_i_acc)
        g_i_auc = max(i_auc,g_i_auc)
        
        df = df.append(dfU,ignore_index=True)
        
        # DOWN
        start_point = (repeat*(cycle_epochs*2)) + cycle_epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1/lr_increase) 
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1) 

        optimizer,dfD,i_loss,i_acc,i_auc = cycleTrain(optimizer,scheduler,cycle_epochs,criterion,
                                  model,train_loader,test_loader,device,
                                  savePath ,start_point,g_i_loss,g_i_acc,g_i_auc)
        
        g_i_loss = min(i_loss,g_i_loss) # updatea a los mejores para guardar
        g_i_acc = max(i_acc,g_i_acc)
        g_i_auc = max(i_auc,g_i_auc)
        
        df = df.append(dfD,ignore_index=True)
        
    fig_tot = trainfig(df)
    
    return (df,fig_tot)


def ReTrain_FullTrain(batch_size,optimizer,B_ciclos,lr_increase,cycle_epochs,
                  criterion,model,data_set,device,
                  resultpath,CPcondition):

    # Salvado de datos
    ModelosFolder =  resultpath + '/Modelos/'
    
    df = pd.DataFrame(columns=['TL', 'TA', 'Tauc'])
    full_dataloader = DataLoader(data_set, batch_size = batch_size, pin_memory=True) 

    optimizer.zero_grad()
    for repeat in range(B_ciclos):
        print(f'Ciclo: {repeat}')    
        # UP
        start_point = repeat * (cycle_epochs*2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_increase) 
                                        
        optimizer,dfU,model = ReTrain_cycleTrain(optimizer,scheduler,cycle_epochs,criterion,
                                      model,full_dataloader,device,start_point) 
        df = df.append(dfU,ignore_index=True)              
    saveModel(cycle_epochs,model,optimizer,ModelosFolder +'Pesos_'+CPcondition+ '_E'+ str(cycle_epochs))

    fig_tot = REtrainfig(df)
    
    return (df,fig_tot)

def REtrainfig(df):
    # multiple line plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

            ax1.plot( 'TL', data=df, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
            ax1.set_title('Loss')
            ax2.plot( 'TA', data=df, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
            ax2.set_title('Accuracy')
            ax3.plot( 'Tauc', data=df, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
            ax3.set_title('AUC')
            return fig
        
def ReTrain_cycleTrain(optimizer,scheduler,cycle_epochs,criterion,
               model,full_dataloader,device,start_point):
    
    train_loss_a,train_accuracy_a,train_auc_a = [],[],[]
    test_loss,test_accuracy,test_auc = valida(model, full_dataloader,device,len(full_dataloader.batch_sampler.sampler),criterion)
    print('Base',test_loss,test_accuracy,test_auc)
    
    train_loss_a.append(test_loss)
    train_accuracy_a.append(test_accuracy)
    train_auc_a.append(test_auc)
    optimizer.zero_grad()
    for epoch in range(cycle_epochs):        
            # Entrenamiento
            print(f'epoch: {epoch}')
            tic = time.time()
            train_loss,train_accuracy,train_auc = train(model, full_dataloader,optimizer,device,len(full_dataloader.batch_sampler.sampler),criterion)
            print('Train',train_loss,train_accuracy,train_auc )
            toc = time.time()
            print(toc - tic)
            train_loss_a.append(train_loss)
            train_accuracy_a.append(train_accuracy)
            train_auc_a.append(train_auc)
            
            scheduler.step() 

            df=pd.DataFrame({ 'TL': train_loss_a, 'TA':train_accuracy_a, 'Tauc':train_auc_a })
                
            model.to(device)
    return (optimizer,df,model)

def cycleTrain(optimizer,scheduler,cycle_epochs,criterion,
               model,train_loader,test_loader,device,
               savePath,start_point,i_loss,i_acc,i_auc):
    
    train_loss_a,train_accuracy_a,train_auc_a,test_loss_a,test_accuracy_a,test_auc_a = [],[],[],[],[],[]
    
    optimizer.zero_grad()
    #fig_online, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for epoch in range(cycle_epochs):        
            # Entrenamiento
            print(f'epoch: {epoch}')
            tic = time.time()
            train_loss,train_accuracy,train_auc = train(model, train_loader,optimizer,device,len(train_loader.batch_sampler.sampler),criterion)
            test_loss,test_accuracy,test_auc = valida(model, test_loader,device,len(test_loader.batch_sampler.sampler),criterion)
            toc = time.time()
            print(toc - tic)
            train_loss_a.append(train_loss)
            train_accuracy_a.append(train_accuracy)
            train_auc_a.append(train_auc)
            test_loss_a.append(test_loss)
            test_accuracy_a.append(test_accuracy)
            test_auc_a.append(test_auc)
            
            scheduler.step() 

            df=pd.DataFrame({ 'TL': train_loss_a, 'VL': test_loss_a, 'TA':train_accuracy_a, 'VA':test_accuracy_a, 'Tauc':train_auc_a, 'Vauc':test_auc_a })
            
            # fig_online= trainfig_online(fig_online, ax1, ax2, ax3,df)
            # fig_online.show()
            # plt.pause(0.05)
            # Guarda los mejores modelos segun los 3 criterios (Loss, accuracy y AUC)
            if test_loss<i_loss:
                old = glob.glob(savePath +'_Loss'+ '*')
                for filePath in old:
                     os.remove(filePath)
                
                saveModel(epoch+start_point,model,optimizer,savePath + '_Loss'+'_E'+ str(epoch+start_point)+'_'+time.strftime("%Y%m%d-%H%M%S"))
                i_loss = test_loss

            if test_accuracy > i_acc:
                old = glob.glob(savePath +'_ACC'+ '*')
                for filePath in old:
                     os.remove(filePath)
                
                saveModel(epoch+start_point,model,optimizer,savePath + '_ACC'+ '_E'+ str(epoch+start_point) + '_'+time.strftime("%Y%m%d-%H%M%S"))
                i_acc = test_accuracy     
            
            if test_auc > i_auc:
                old = glob.glob(savePath +'_AUC'+ '*')
                for filePath in old:
                     os.remove(filePath)
                
                saveModel(epoch+start_point,model,optimizer,savePath + '_AUC'+ '_E'+ str(epoch+start_point) + '_'+time.strftime("%Y%m%d-%H%M%S"))
                i_auc = test_auc                          
                
            model.to(device)
    # plt.close(fig_online)      
    return (optimizer,df,i_loss,i_acc,i_auc)

def asfunction_corridaSimple(datapath,resultpath,filename,model_complexity,NC,UsarTF,UsarFase,device,iteraciones) :
    maxPropo = False
    filename_reversal = filename + '_reversal'
    CPcondition = 'MCx_' +str(model_complexity) + '_comp'+'_'.join([str(int) for int in NC] )
    
    if UsarTF and (not UsarFase):
        CPcondition = "TF_" + CPcondition 
        batch_size = 1800
    
    if UsarTF and UsarFase:
        CPcondition = "Ph_TF_" + CPcondition 
        batch_size = 1800
    
    if (not UsarTF) and UsarFase:
       CPcondition = "Ph_" + CPcondition 
       batch_size = 1800
       
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

    #%%############################################################################
    # Data transformation to TF chart
    #sample_rate = 256
    #time = see_data(datapath + filename + '_comp1',sample_rate)
    if UsarTF or UsarFase:
        s_Forward = torch.cat([data2spectrum(sp,UsarTF,UsarFase,maxPropo) for sp in compnames(datapath + filename,NC)],dim=1) # NC tiene las componentes usadas
        s_Backward = torch.cat([data2spectrum(sp,UsarTF,UsarFase,maxPropo) for sp in compnames(datapath + filename_reversal,NC)],dim=1)
        #fig, (ax1,ax2) = plt.subplots(1,2)
        #ax1.imshow(s_Forward[15,:,:].numpy(), cmap='rainbow', aspect='auto',extent = [0 , time[1023],  sample_rate // 2,0])
        #ax2.imshow(s_Backward[15,:,:].numpy(), cmap='rainbow', aspect='auto',extent = [0 , time[1023],  sample_rate // 2,0])
        #fig.colorbar(img,orientation="horizontal")      
    else:
        Npuntos = True
        s_Forward = torch.stack([loadCP(sp,Npuntos,maxPropo) for sp in compnames(datapath + filename,NC)]).permute(1,0,2) # NC tiene las componentes usadas
        s_Backward = torch.stack([loadCP(sp, Npuntos,maxPropo) for sp in compnames(datapath + filename_reversal,NC)]).permute(1,0,2)
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
    print(device)
    
    if device == PLACA:
        num_workers = 3
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    
    wd = 1e-3
    base_lr, top_lr, cycle_epochs = 5e-5, 5e-4, 100 
    lr_increase = (top_lr/base_lr)**(1/cycle_epochs)
    B_ciclos = 1
    criterion = nn.BCELoss().to(device)
    
# inicializa DFrames       
    column_names = ["Iteration", 'TL', 'VL', 'TA', 'VA', 'Tauc', 'Vauc', 'epoca'] # T = Training, V = Validation
    iter_df = pd.DataFrame(columns = column_names)
    sh_iter_df = pd.DataFrame(columns = column_names)
    
 #%%# iteraciones
    for i in range(iteraciones[0],iteraciones[1]):#100
        print('Iteracion'+ str(i))

        if  isfile(savePath_ima + '_It'+str(i)+'_Shuffled.png') :    # corre el modelo solo si no esta guardada la figura
            print('Levantando dfs hasta'+ str(i))
            iter_df =  pd.read_csv(savePath + 'iter_df.csv') # si esta guardada levanta los DFrame
            sh_iter_df =  pd.read_csv(savePath + 'sh_iter_df.csv')
            continue
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
                
    return

def LoadModel(elem_freq,model_complexity,device,prefilepathORI,filename,prefilename = None ):
    
    if prefilename == None:
        listOfFiles = os.listdir(prefilepathORI)
        pattern = filename + 'It*_*E*'
        for entry in listOfFiles:
            if fnmatch.fnmatch(entry, pattern):
               prefilename = entry   
        
    if model_complexity == 3:
            n_puntos,FCval = 4, 9600 # cuando pasa por ventaneado (TF o fase) N puntos chico
            model = SConv1D(in_features = elem_freq ,n_features = 100,p_bias = False,n_puntos = n_puntos,DO= 0.4 ,FCval = FCval).to(device)# senial a 1/2 
    if model_complexity == 2:
            n_puntos,FCval = 4, 12000 # cuando pasa por ventaneado (TF o fase) N puntos chico
            model = SConv1D_2(in_features = elem_freq ,n_features = 100,p_bias = False,n_puntos = n_puntos,DO= 0.4 ,FCval = FCval).to(device)        
    if model_complexity == 1:
            n_puntos,FCval = 4, 12700 # cuando pasa por ventaneado (TF o fase) N puntos chico
            model = SConv1D_1(in_features = elem_freq ,n_features = 100,p_bias = False,n_puntos = n_puntos,DO= 0.4 ,FCval = FCval).to(device)
        
    checkpoint = torch.load(prefilepathORI +'/'+prefilename)
    model.load_state_dict(checkpoint['model_state_dict'])
    params = [p for p in model.parameters() if p.requires_grad]
    # params = [p for p in net.parameters()]
    optimizer = torch.optim.Adam(params = params)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
               state[k] = v.to(device)

    return (model,optimizer)

def retrain_best(datapath,resultpath,filename,model_complexity,NC,UsarTF,UsarFase,device,prefix,folderdata,basepath,base_lr, top_lr, cycle_epochs):
    filename_reversal = filename + '_reversal'
    CPcondition = prefix + 'MCx_' +str(model_complexity) + '_comp'+'_'.join([str(int) for int in NC] )
    batch_size = 1800
    ImagenesFolder = resultpath +'/Figures/'
    
    iter_df = pd.DataFrame(columns = ['TL',  'TA', 'Tauc', 'epoca'])
    #%%############################################################################
    # Data transformation to TF chart
    #sample_rate = 256
    #time = see_data(datapath + filename + '_comp1',sample_rate)
    if UsarTF or UsarFase:
        maxPropo = False
        s_Forward = torch.cat([data2spectrum(sp,UsarTF,UsarFase,maxPropo) for sp in compnames(datapath + filename,NC)],dim=1) # NC tiene las componentes usadas
        s_Backward = torch.cat([data2spectrum(sp,UsarTF,UsarFase,maxPropo) for sp in compnames(datapath + filename_reversal,NC)],dim=1)  
    else:
        Npuntos = True
        s_Forward = torch.stack([loadCP(sp,Npuntos,maxPropo) for sp in compnames(datapath + filename,NC)]).permute(1,0,2) # NC tiene las componentes usadas
        s_Backward = torch.stack([loadCP(sp, Npuntos,maxPropo) for sp in compnames(datapath + filename_reversal,NC)]).permute(1,0,2)
    
            
    data = torch.cat((s_Forward, s_Backward), 0)
    data = data.type('torch.FloatTensor')
    labels = torch.tensor([1]*s_Forward.shape[0] + [0]*s_Backward.shape[0]) # 1 es s_Forward, 0 es s_Backward   
    data_set = TensorDataset(data, labels)

    del s_Forward,s_Backward,data
   
    #####Levantar modelo################################################
    prefilepathORI = basepath +folderdata + CPcondition + '_Modelos'
    elem_freq = int(data_set[1][0].shape[0])
    model,optimizer = LoadModel(elem_freq,model_complexity,device,prefilepathORI,filename)    
    #%% ##############################################################################
    # Parametros de red, entrenamiento y salvado de datos
    for g in optimizer.param_groups:
        g['lr'] = base_lr
    wd = 1e-3
    lr_increase = (top_lr/base_lr)**(1/cycle_epochs)
    B_ciclos = 1
    criterion = nn.BCELoss().to(device)    

    df,fig = ReTrain_FullTrain(batch_size,optimizer,B_ciclos,lr_increase,cycle_epochs,
                      criterion,model,data_set,device,resultpath,CPcondition+'_'+filename)
         
    fig.suptitle(CPcondition +'_'+filename +' ADAM:' ' base_lr=' + str(base_lr)  + ' wd=' + str(wd))
    fig.savefig(ImagenesFolder +'Figure_'+CPcondition +'_'+filename+'.png')   
    plt.close(fig) 
    df['epoca'] = df.index
    iter_df = iter_df.append(df,ignore_index=True)
    iter_df.to_csv( resultpath + '/Modelos/' + 'df_'+CPcondition +'_'+filename+ '.csv' )
        
    return

# Modelo imagen 2D 
class SConv(nn.Module):

    def __init__(self,n_features,p_bias,n_freqs):
        super(SConv, self).__init__()
       
        # Block 1
        self.bn11 = nn.BatchNorm2d(1)
        self.conv11 = nn.Conv2d(1, n_features, kernel_size = n_freqs , padding=1, bias=p_bias) 
        self.bn12 = nn.BatchNorm2d(n_features)
        self.conv12 = nn.Conv2d(n_features, n_features, 3, padding=1, bias=p_bias)
        self.mp1 = nn.MaxPool2d(kernel_size = (3,3) ,stride = (2,2),padding=1)
        self.do1 = nn.Dropout2d(p=0.25)
        
        self.bn21 = nn.BatchNorm2d(n_features)
        self.conv21 = nn.Conv2d(n_features, 2*n_features, 3, padding=1, bias=p_bias)
        self.bn22 = nn.BatchNorm2d(2*n_features)
        self.conv22 = nn.Conv2d(2*n_features, 2*n_features, 3, padding=1, bias=p_bias)
        self.mp2 = nn.MaxPool2d(kernel_size = (3,3) ,stride = (2,2),padding=1)
        self.do2 = nn.Dropout2d(p=0.25)

        self.bn31 = nn.BatchNorm2d(2*n_features)
        self.conv31 = nn.Conv2d(2*n_features, 4*n_features, 3, padding=1, bias=p_bias)
        self.bn32 = nn.BatchNorm2d(4*n_features)
        self.conv32 = nn.Conv2d(4*n_features, 4*n_features, 3, padding=1, bias=p_bias)
        self.mp3 = nn.MaxPool2d(kernel_size = (3,3) ,stride = (2,2),padding=1)
        self.do3 = nn.Dropout2d(p=0.25)
         
        self.fc1 = nn.Linear(7920 , 256, bias=p_bias)#bias = False 39087
        self.do6 = nn.Dropout(p = 0.25) #0.2
        self.fc2 = nn.Linear(256, 1, bias=p_bias)#bias = False 
   
    def forward(self, x):
        
        # Block 1
        x = self.conv11(F.elu(self.bn11(x)))
        x = self.conv12(F.elu(self.bn12(x)))
        x = self.mp1(x)
        x = self.do1(x)
       
        x = self.conv21(F.elu(self.bn21(x)))
        x = self.conv22(F.elu(self.bn22(x)))
        x = self.mp2(x)
        x = self.do2(x)

        x = self.conv31(F.elu(self.bn31(x)))
        x = self.conv32(F.elu(self.bn32(x)))
        x = self.mp3(x)
        x = self.do3(x)
        #print('pref:'+ str(x.size()))

        # Fully connected layers
        x= torch.flatten(x, start_dim=1)
        #print('post:'+ str(x.size()))
        
        x = F.elu(self.fc1(x))
        x = self.do6(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# Modelo conv 1D  complex 3
class SConv1D(nn.Module):

    def __init__(self,in_features,n_features,p_bias,n_puntos, DO,FCval):
        super(SConv1D, self).__init__()
       
        # Block 1
        self.bn11 = nn.BatchNorm1d(in_features)
        self.conv11 = nn.Conv1d(in_features, n_features, kernel_size = n_puntos , padding=1, bias=p_bias) 
        self.bn12 = nn.BatchNorm1d(n_features)
        self.conv12 = nn.Conv1d(n_features, n_features, n_puntos, padding=1, bias=p_bias)
        self.mp1 = nn.MaxPool1d(kernel_size = (2) ,stride = (2),padding=0)
        self.do1 = nn.Dropout(p=DO)
        
        self.bn21 = nn.BatchNorm1d(n_features)
        self.conv21 = nn.Conv1d(n_features, 2*n_features, n_puntos, padding=1, bias=p_bias)
        self.bn22 = nn.BatchNorm1d(2*n_features)
        self.conv22 = nn.Conv1d(2*n_features, 2*n_features, n_puntos, padding=1, bias=p_bias)
        self.mp2 = nn.MaxPool1d(kernel_size = (2) ,stride = (2),padding=0)
        self.do2 = nn.Dropout(p=DO)

        self.bn31 = nn.BatchNorm1d(2*n_features)
        self.conv31 = nn.Conv1d(2*n_features, 4*n_features, n_puntos, padding=1, bias=p_bias)
        self.bn32 = nn.BatchNorm1d(4*n_features)
        self.conv32 = nn.Conv1d(4*n_features, 4*n_features, n_puntos, padding=1, bias=p_bias)
        self.mp3 = nn.MaxPool1d(kernel_size = (2) ,stride = (2),padding=0)
        self.do3 = nn.Dropout(p=DO)

        self.bn41 = nn.BatchNorm1d(4*n_features)
        self.conv41 = nn.Conv1d(4*n_features, 8*n_features, n_puntos, padding=1, bias=p_bias)
        self.bn42 = nn.BatchNorm1d(8*n_features)
        self.conv42 = nn.Conv1d(8*n_features, 8*n_features, n_puntos, padding=1, bias=p_bias)
        self.mp4 = nn.MaxPool1d(kernel_size = (2) ,stride = (2),padding=0)
        self.do4 = nn.Dropout(p=DO)

        self.bn51 = nn.BatchNorm1d(8*n_features)
        self.conv51 = nn.Conv1d(8*n_features, 16*n_features, n_puntos, padding=1, bias=p_bias)
        self.bn52 = nn.BatchNorm1d(16*n_features)
        self.conv52 = nn.Conv1d(16*n_features, 16*n_features, n_puntos, padding=1, bias=p_bias)
        self.mp5 = nn.MaxPool1d(kernel_size = (2) ,stride = (2),padding=0)
        self.do5 = nn.Dropout(p=DO)

        #self.bn61 = nn.BatchNorm1d(16*n_features)
        #self.conv61 = nn.Conv1d(16*n_features, 32*n_features, n_puntos, padding=1, bias=p_bias)
        #self.bn62 = nn.BatchNorm1d(32*n_features)
        #self.conv62 = nn.Conv1d(32*n_features, 32*n_features, n_puntos, padding=1, bias=p_bias)
        #self.mp6 = nn.MaxPool1d(kernel_size = (2) ,stride = (2),padding=0)
        #self.do6 = nn.Dropout(p=DO)
         
        self.fc1 = nn.Linear(FCval , 1024, bias=p_bias)#bias = False 
        self.dofc = nn.Dropout(p = DO*0.25) #0.2
        self.fc2 = nn.Linear(1024, 1, bias=p_bias)#bias = False 
   
    def forward(self, x):
        
        # Block 1
        x = self.conv11(F.elu(self.bn11(x)))
        x = self.conv12(F.elu(self.bn12(x)))
        x = self.mp1(x)
        x = self.do1(x)
       
        x = self.conv21(F.elu(self.bn21(x)))
        x = self.conv22(F.elu(self.bn22(x)))#
        x = self.mp2(x)
        x = self.do2(x)

        x = self.conv31(F.elu(self.bn31(x)))
        x = self.conv32(F.elu(self.bn32(x)))
        x = self.mp3(x)
        x = self.do3(x)

        x = self.conv41(F.elu(self.bn41(x)))
        x = self.conv42(F.elu(self.bn42(x)))
        x = self.mp4(x)
        x = self.do4(x)

        x = self.conv51(F.elu(self.bn51(x)))
        x = self.conv52(F.elu(self.bn52(x)))
        x = self.mp5(x)
        x = self.do5(x)

        #x = self.conv61(F.elu(self.bn61(x)))
        #x = self.conv62(F.elu(self.bn62(x)))
        #x = self.mp6(x)
        #x = self.do6(x)
        #print('pref:'+ str(x.size()))

        # Fully connected layers
        x= torch.flatten(x, start_dim=1)
        #print('post:'+ str(x.size()))
        
        x = F.elu(self.fc1(x))
        x = self.dofc(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# Modelo conv 1D  complex 2
class SConv1D_2(nn.Module):

    def __init__(self,in_features,n_features,p_bias,n_puntos, DO,FCval):
        super(SConv1D_2, self).__init__()
       
        # Block 1
        self.bn11 = nn.BatchNorm1d(in_features)
        self.conv11 = nn.Conv1d(in_features, n_features, kernel_size = n_puntos , padding=1, bias=p_bias) 
        self.bn12 = nn.BatchNorm1d(n_features)
        self.conv12 = nn.Conv1d(n_features, n_features, n_puntos, padding=1, bias=p_bias)
        self.mp1 = nn.MaxPool1d(kernel_size = (2) ,stride = (2),padding=0)
        self.do1 = nn.Dropout(p=DO)
        
        self.bn21 = nn.BatchNorm1d(n_features)
        self.conv21 = nn.Conv1d(n_features, 2*n_features, n_puntos, padding=1, bias=p_bias)
        self.bn22 = nn.BatchNorm1d(2*n_features)
        self.conv22 = nn.Conv1d(2*n_features, 2*n_features, n_puntos, padding=1, bias=p_bias)
        self.mp2 = nn.MaxPool1d(kernel_size = (2) ,stride = (2),padding=0)
        self.do2 = nn.Dropout(p=DO)

        self.bn31 = nn.BatchNorm1d(2*n_features)
        self.conv31 = nn.Conv1d(2*n_features, 4*n_features, n_puntos, padding=1, bias=p_bias)
        self.bn32 = nn.BatchNorm1d(4*n_features)
        self.conv32 = nn.Conv1d(4*n_features, 4*n_features, n_puntos, padding=1, bias=p_bias)
        self.mp3 = nn.MaxPool1d(kernel_size = (2) ,stride = (2),padding=0)
        self.do3 = nn.Dropout(p=DO)
         
        self.fc1 = nn.Linear(FCval , 1024, bias=p_bias)#bias = False 
        self.dofc = nn.Dropout(p = DO*0.25) #0.2
        self.fc2 = nn.Linear(1024, 1, bias=p_bias)#bias = False 
   
    def forward(self, x):
        
        # Block 1
        x = self.conv11(F.elu(self.bn11(x)))
        x = self.conv12(F.elu(self.bn12(x)))
        x = self.mp1(x)
        x = self.do1(x)
       
        x = self.conv21(F.elu(self.bn21(x)))
        x = self.conv22(F.elu(self.bn22(x)))#
        x = self.mp2(x)
        x = self.do2(x)

        x = self.conv31(F.elu(self.bn31(x)))
        x = self.conv32(F.elu(self.bn32(x)))
        x = self.mp3(x)
        x = self.do3(x)

        #print('pref:'+ str(x.size()))

        # Fully connected layers
        x= torch.flatten(x, start_dim=1)
        #print('post:'+ str(x.size()))
        
        x = F.elu(self.fc1(x))
        x = self.dofc(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# Modelo conv 1D  complex 1
class SConv1D_1(nn.Module):

    def __init__(self,in_features,n_features,p_bias,n_puntos, DO,FCval):
        super(SConv1D_1, self).__init__()
       
        # Block 1
        self.bn11 = nn.BatchNorm1d(in_features)
        self.conv11 = nn.Conv1d(in_features, n_features, kernel_size = n_puntos , padding=1, bias=p_bias) 
        self.bn12 = nn.BatchNorm1d(n_features)
        self.conv12 = nn.Conv1d(n_features, n_features, n_puntos, padding=1, bias=p_bias)
        self.mp1 = nn.MaxPool1d(kernel_size = (2) ,stride = (2),padding=0)
        self.do1 = nn.Dropout(p=DO)
         
        self.fc1 = nn.Linear(FCval , 1024, bias=p_bias)#bias = False 
        self.dofc = nn.Dropout(p = DO*0.25) #0.2
        self.fc2 = nn.Linear(1024, 1, bias=p_bias)#bias = False 
   
    def forward(self, x):
        
        # Block 1
        x = self.conv11(F.elu(self.bn11(x)))
        x = self.conv12(F.elu(self.bn12(x)))
        x = self.mp1(x)
        x = self.do1(x)
       
        #print('pref:'+ str(x.size()))

        # Fully connected layers
        x= torch.flatten(x, start_dim=1)
        #print('post:'+ str(x.size()))
        
        x = F.elu(self.fc1(x))
        x = self.dofc(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

def LoadDataTensors(datapath,filename,NC,UsarTF,UsarFase):
    filename_reversal = filename + '_reversal'
    #sample_rate = 256
    #time = see_data(datapath + filename + '_comp1',sample_rate)
    if UsarTF or UsarFase:
        maxPropo = False
        s_Forward = torch.cat([data2spectrum(sp,UsarTF,UsarFase,maxPropo) for sp in compnames(datapath + filename,NC)],dim=1) # NC tiene las componentes usadas
        s_Backward = torch.cat([data2spectrum(sp,UsarTF,UsarFase,maxPropo) for sp in compnames(datapath + filename_reversal,NC)],dim=1)  
    else:
        Npuntos = True
        s_Forward = torch.stack([loadCP(sp,Npuntos,maxPropo) for sp in compnames(datapath + filename,NC)]).permute(1,0,2) # NC tiene las componentes usadas
        s_Backward = torch.stack([loadCP(sp, Npuntos,maxPropo) for sp in compnames(datapath + filename_reversal,NC)]).permute(1,0,2)
    
            
    data = torch.cat((s_Forward, s_Backward), 0)
    data = data.type('torch.FloatTensor')
    labels = torch.tensor([1]*s_Forward.shape[0] + [0]*s_Backward.shape[0]) # 1 es s_Forward, 0 es s_Backward   
    data_set = TensorDataset(data, labels)

    return data_set,labels

def evalModel(data_set, model,device,batch_size):  
    model.eval()
    model = model.to(device)
    full_dataloader = DataLoader(data_set, batch_size = batch_size) 
    targets,predictions = [],[]
    
    for batch_idx, (image, target) in enumerate(full_dataloader):
        print(round(len(targets)/len(data_set),1))
        image = image.to(device)     
        target = target.to(device)
        output = model(image)
        
        targets.extend(target.detach().cpu().numpy())
        predictions.extend(output.detach().cpu().numpy())
        # plt.plot(predictions,'o',alpha = 0.09)
        # plt.plot(targets) 
    predictions = [item for sublist in [list(x) for x in predictions] for item in sublist]
    torch.cuda.empty_cache()            

    return targets,predictions

def IG_CAPTUM(data_set, model,device):  
    model.zero_grad()

    model = model.to(device).half()
    IG = IntegratedGradients(model)    
    
    full_dataloader = DataLoader(data_set, batch_size = 800) 
    GradienImages = []# [[] for i in range(int(len(data_set)))]

    for batch_idx, (image, target) in enumerate(full_dataloader):
        print(round(batch_idx*len(target)/len(data_set),3))
        image = image.to(device).half()
        attributions_IG = IG.attribute(image,internal_batch_size=800)
        GradienImages.extend(attributions_IG.detach().cpu().numpy())
        
        del attributions_IG
        torch.cuda.empty_cache()
    # plt.plot(predictions,'o',alpha = 0.09)
    # plt.plot(targets)

    # #sample_rate = 256
    # #time = see_data(datapath + filename + '_comp1',sample_rate)
    # fig, (ax1,ax2) = plt.subplots(1,2)
    # im1 = ax1.imshow(img.squeeze(0).detach().cpu().numpy(), cmap='rainbow', aspect='auto',extent = [0 , time[1023],  sample_rate // 2,0])
    # im2 = ax2.imshow(attributions_IG.squeeze(0).detach().cpu().numpy(),vmin = -0.005, vmax = 0.005, cmap='RdBu', aspect='auto',extent = [0 , time[1023],  sample_rate // 2,0])
    # fig.colorbar(im1,ax=ax1,orientation="horizontal") 
    # fig.colorbar(im2,ax=ax2,orientation="horizontal") 
    del model
    torch.cuda.empty_cache()            
    return GradienImages

def IG_CAPTUM_single(data_set, model,device):  
    model.zero_grad()

    model = model.to(device).half()
    IG = IntegratedGradients(model)    
    
    #full_dataloader = DataLoader(data_set, batch_size = 1) 
    GradienImages =  torch.empty(0, int(data_set[1][0].shape[0]),int(data_set[1][0].shape[1])).to(device)
    
    #for batch_idx, (image, target) in enumerate(full_dataloader):
    for i in range(len(data_set)):
        sample = data_set[i]
        print(round(i/len(data_set),3))
        image = sample[0].unsqueeze(0).to(device).half()
        attributions_IG = IG.attribute(image,internal_batch_size=1)
        torch.cuda.empty_cache()
        
        GradienImages = torch.cat((GradienImages,attributions_IG),dim = 0)
    # plt.plot(predictions,'o',alpha = 0.09)
    # plt.plot(targets)

    # #sample_rate = 256
    # #time = see_data(datapath + filename + '_comp1',sample_rate)
    # fig, (ax1,ax2) = plt.subplots(1,2)
    # im1 = ax1.imshow(img.squeeze(0).detach().cpu().numpy(), cmap='rainbow', aspect='auto',extent = [0 , time[1023],  sample_rate // 2,0])
    # im2 = ax2.imshow(attributions_IG.squeeze(0).detach().cpu().numpy(),vmin = -0.005, vmax = 0.005, cmap='RdBu', aspect='auto',extent = [0 , time[1023],  sample_rate // 2,0])
    # fig.colorbar(im1,ax=ax1,orientation="horizontal") 
    # fig.colorbar(im2,ax=ax2,orientation="horizontal") 
    del model
    torch.cuda.empty_cache()            
    return GradienImages
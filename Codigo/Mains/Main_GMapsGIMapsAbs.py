# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:18:45 2021
Re-Entrenamiento
@author: xochipilli
"""

import pandas as pd
# Dependencias
import sys
import os
import torch
import matplotlib.pyplot as plt
%matplotlib qt
from os.path import isfile
import pickle
import numpy as np

functionspath = 'E:/tiempo/Codigo/Funciones/'
sys.path.insert(0, functionspath)
from Functions import (IG_CAPTUM,LoadDataTensors,LoadModel,evalModel,IG_CAPTUM_single)

#%%#############################################################################
# Cosas para editar
basepath = 'E:/tiempo/Resultados/zz_ModelosReEntrenados'
datapath = 'E:/tiempo/Datos/'
resultpath = 'E:/tiempo/Resultados/zz_ModelosReEntrenados'
device = "cuda:0" ## En que placa laburo? o "cuda:1"/"cuda:0" y ver lo de paralelizar

Ncomp = [[1],[1,2],[1,2,3]] ## Que componentes usar
TF_conditions = [[False,True,'Ph_'],[True,False,'TF_'],[True,True,'Ph_TF_']] # TF + Fase , solo Fase
conditions = ['keta','awake','sleep']
 # #sample_rate = 256
 # #time = see_data(datapath + filename + '_comp1',sample_rate)

for i,cond in enumerate(conditions):
    filename = cond  
    for i,NC in enumerate(Ncomp):
        for i,tf_cond in enumerate(TF_conditions):
            UsarTF,UsarFase,prefix = tf_cond
             # Load data
            for model_complexity in [1,2,3]: # Complejidad del modelo, puede ser 3 (5 bloques), 2 (3 bloques) o 1 (1 Bloque)              
            # Load model
                modelName = 'Pesos_'+ prefix + 'MCx_'+ str(model_complexity) + '_comp'+ '_'.join([str(int) for int in NC]) +'_'+ filename + '_E10'
                colname = prefix + 'MCx_'+ str(model_complexity) + '_comp'+ '_'.join([str(int) for int in NC]) +'_'+ filename 
                
                if  isfile(resultpath +'/GIMapsAbs/Favg_'+colname + '_df.csv') :    # corre el modelo solo si no esta guardada la figura
                    print('Ya corrio '+ colname)
                    continue

                GradienImages = pd.read_pickle(resultpath +'/GIMaps/IG_'+colname+".p")
                GradienImages = [abs(number) for number in GradienImages]

                timeaverage = pd.DataFrame([np.mean(val,axis = 0) for val in GradienImages])
                freqaverage = pd.DataFrame([np.mean(val,axis = 1) for val in GradienImages])
                
                timeaverage.to_csv(resultpath +'/GIMapsAbs/Tavg_'+colname+  '_df.csv' )              # save data
                freqaverage.to_csv(resultpath +'/GIMapsAbs/Favg_'+colname+  '_df.csv' )              # save data

                # npdata = freqaverage.detach().cpu().numpy()
                # import numpy as np
                # plt.plot(np.transpose(npdata),'o',alpha = 0.09)
                # plt.plot(targets)
 # plt.plot(predictions,'o',alpha = 0.09)
    # plt.plot(targets)

    # #sample_rate = 256
    # #time = see_data(datapath + filename + '_comp1',sample_rate)
    # fig, (ax1,ax2) = plt.subplots(1,2)
    # im1 = ax1.imshow(img.squeeze(0).detach().cpu().numpy(), cmap='rainbow', aspect='auto',extent = [0 , time[1023],  sample_rate // 2,0])
    # im2 = ax2.imshow(attributions_IG.squeeze(0).detach().cpu().numpy(),vmin = -0.005, vmax = 0.005, cmap='RdBu', aspect='auto',extent = [0 , time[1023],  sample_rate // 2,0])
    # fig.colorbar(im1,ax=ax1,orientation="horizontal") 
    # fig.colorbar(im2,ax=ax2,orientation="horizontal") 
            
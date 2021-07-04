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
functionspath = 'E:/tiempo/Codigo/Funciones/'
sys.path.insert(0, functionspath)
from Functions import (retrain_best)

#%%#############################################################################
# Cosas para editar
basepath = 'E:/tiempo/Resultados/'
datapath = 'E:/tiempo/Datos/'
resultpath = 'E:/tiempo/Resultados/zz_ModelosReEntrenados'
device = "cuda:0" ## En que placa laburo? o "cuda:1"/"cuda:0" y ver lo de paralelizar

Ncomp = [[1] ,[1,2],[1,2,3]] ## Que componentes usar
TF_conditions = [[True,True,'Ph_TF_','TFyFase/'],[False,True,'Ph_','Fase/'],[True,False,'TF_','TF/']] # TF + Fase , solo Fase

base_lr, top_lr, cycle_epochs = 5e-5, 9e-5, 10 # Aca defino epocas de re entrenamiento y el lr es siempre el mismo 

for model_complexity in [1,2,3]: # Complejidad del modelo, puede ser 3 (5 bloques), 2 (3 bloques) o 1 (1 Bloque)
    for i,NC in enumerate(Ncomp):
        for i,tf_cond in enumerate(TF_conditions):
            print(tf_cond)
            UsarTF,UsarFase,prefix,folderdata = tf_cond
     
            retrain_best(datapath,resultpath,'awake',model_complexity,NC,UsarTF,UsarFase,device,prefix,folderdata,basepath,base_lr, top_lr, cycle_epochs) 
            retrain_best(datapath,resultpath,'keta',model_complexity,NC,UsarTF,UsarFase,device,prefix,folderdata,basepath,base_lr, top_lr, cycle_epochs) 
            retrain_best(datapath,resultpath,'sleep',model_complexity,NC,UsarTF,UsarFase,device,prefix,folderdata,basepath,base_lr, top_lr, cycle_epochs) 
        
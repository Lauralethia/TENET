# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:04:20 2021

@author: Amelie
"""
#%%#############################################################################
# Dependencias
#!pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
import pandas as pd
# Dependencias
import sys
import os
functionspath = 'E:/tiempo/Codigo/Funciones/'
sys.path.insert(0, functionspath)
from Functions_figures import (all_cleanModelFolders)
from asfunction_corridaSimple import (retrain10_best)

#%%#############################################################################
# Cosas para editar
basepath = 'E:/tiempo/Resultados/'
Ncomp = [[1] ,[1,2],[1,2,3]] ## Que componentes usar
TF_conditions = [['Ph_TF_','TFyFase/'],['TF_','TF/'],['Ph_','Fase/']] # TF + Fase , solo Fase, solo TF

for model_complexity in [1,2,3]: # Complejidad del modelo, puede ser 3 (5 bloques), 2 (3 bloques) o 1 (1 Bloque)
    for i,NC in enumerate(Ncomp):
        for i,tf_cond in enumerate(TF_conditions):
            print(tf_cond)
            prefix,folderdata = tf_cond
            all_cleanModelFolders(model_complexity,NC,prefix,folderdata,basepath)
            

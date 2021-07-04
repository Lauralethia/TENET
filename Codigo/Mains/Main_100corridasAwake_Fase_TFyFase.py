# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 07:36:24 2021

@author: xochipilli
"""
import matplotlib.pyplot as plt
%matplotlib auto
import sys
import os
functionspath = 'E:/tiempo/Codigo/Funciones/'
sys.path.insert(0, functionspath)
from asfunction_corridaSimple import (asfunction_corridaSimple)

#%%#############################################################################
# Cosas para editar
PLACA = "cuda:0" ## En que placa laburo? o "cuda:1"/"cuda:0" y ver lo de paralelizar
## Que condiciones analizar
filename =  'awake'
#filename =  'keta'
#filename =  'sleep'
#filename =  'propo'
#%%#############################################################################
## Donde 
datapath = 'E:/tiempo/Datos/'
basepath = 'E:/tiempo/Resultados/'
iteraciones = [0,100]

Ncomp = [[1] ,[1,2],[1,2,3]] ## Que componentes usar
TF_conditions = [[True,True],[False,True]] # TF + Fase , solo Fase

for model_complexity in [1,2,3]: # Complejidad del modelo, puede ser 3 (5 bloques), 2 (3 bloques) o 1 (1 Bloque)
    for i,NC in enumerate(Ncomp):
        for i,tf_cond in enumerate(TF_conditions):
            UsarTF,UsarFase = tf_cond
            if  UsarFase and UsarTF: # TF y Fase
             resultpath = basepath + 'TFyFase/'
             asfunction_corridaSimple(datapath,resultpath,filename,model_complexity,NC,UsarTF,UsarFase,PLACA,iteraciones) 
            else:  # Fase
              resultpath = basepath + 'Fase/'
              asfunction_corridaSimple(datapath,resultpath,filename,model_complexity,NC,UsarTF,UsarFase,PLACA,iteraciones) 
               
            
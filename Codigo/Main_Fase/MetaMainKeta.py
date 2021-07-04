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
#filename =  'awake'
filename =  'keta'
#filename =  'sleep'
#filename =  'propo'
#%%#############################################################################
## Donde 
datapath = 'E:/tiempo/Datos/'
#resultpath = 'E:/tiempo/Resultados/SingleTest/'
resultpath = 'E:/tiempo/Resultados/SingleTestMaxPropo/'
maxPropo = True

Ncomp = [[1] ,[1,2],[1,2,3]] ## Que componentes usar
TF_conditions = [[True,True],[True,False],[False,True],[False,False]] # TF + Fase , solo Fase, solo TF, Cruda

for model_complexity in [3,2,1]: # Complejidad del modelo, puede ser 3 (5 bloques), 2 (3 bloques) o 1 (1 Bloque)
    for i,NC in enumerate(Ncomp):
        for i,tf_cond in enumerate(TF_conditions):
            UsarTF,UsarFase = tf_cond
            asfunction_corridaSimple(datapath,resultpath,filename,model_complexity,NC,UsarTF,UsarFase,PLACA,maxPropo) 
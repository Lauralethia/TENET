# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:52:30 2021

@author: xochipilli
"""

#%%#############################################################################
# Dependencias
#!pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
import pandas as pd
import matplotlib.pyplot as plt
import time
# Dependencias

import sys
import os
import seaborn as sns
#%%#############################################################################
# Cosas para editar
## Donde correr
functionspath = 'E:/tiempo/Codigo/Funciones/'

sys.path.insert(0, functionspath)
from Functions_figures import (get_data,figure_all)

basepath = 'E:/tiempo/Resultados/Fase/' 
prefix = 'Ph_' 
savefix = 'Fase/'

UsarTF = True
# en basepath deben estar los archivos, el script de funciones, 
## Que condiciones analizar
#filename =  'awake'
#filename =  'keta'
#filename =  'sleep'

# Load all scv result files
model_complexity = 1
#prefix = 'Ph_TF_' 
#savefix = 'TFyFase/'



savepath = 'E:/tiempo/Figuras/'+savefix+'MCx'+str(model_complexity)
      
if not os.path.exists(savepath):
    os.makedirs(savepath)
    
NC = [1]
keta_c1,sh_keta_c1 = get_data(basepath,'keta',NC,model_complexity,prefix )
awake_c1,sh_awake_c1 = get_data(basepath,'awake',NC ,model_complexity,prefix )
sleep_c1,sh_sleep_c1 = get_data(basepath,'sleep',NC ,model_complexity,prefix )

NC =[1,2]
keta_c12,sh_keta_c12 = get_data(basepath,'keta', NC ,model_complexity,prefix )
awake_c12,sh_awake_c12 = get_data(basepath,'awake', NC ,model_complexity,prefix )
sleep_c12,sh_sleep_c12 = get_data(basepath,'sleep', NC  ,model_complexity,prefix)

NC =[1,2,3]
keta_c123,sh_keta_c123 = get_data(basepath,'keta', NC ,model_complexity,prefix  )
awake_c123,sh_awake_c123 = get_data(basepath,'awake', NC  ,model_complexity,prefix)
sleep_c123,sh_sleep_c123 = get_data(basepath,'sleep', NC ,model_complexity ,prefix )

import pickle

# make an example object to pickle
some_obj = {'keta_c1':keta_c1, 'sh_keta_c1':sh_keta_c1, 
            'awake_c1':awake_c1, 'sh_awake_c1':sh_awake_c1,
            'sleep_c1':sleep_c1, 'sh_sleep_c1':sh_sleep_c1,

            'keta_c12':keta_c12, 'sh_keta_c12':sh_keta_c12,
            'awake_c12':awake_c12, 'sh_awake_c12':sh_awake_c12,
            'sleep_c12':sleep_c12, 'sh_sleep_c12':sh_sleep_c12,

            'keta_c123':keta_c123, 'sh_keta_c123':sh_keta_c123,
            'awake_c123':awake_c123, 'sh_awake_c123':sh_awake_c123,
            'sleep_c123':sleep_c123, 'sh_sleep_c123':sh_sleep_c123}

pickle.dump(some_obj, open(basepath+"/Pickle_"+prefix+'MCx'+str(model_complexity)+".p", "wb"))  # save it into a file named save.p

# Figuras
variable,tvar,chivar = 'A',100,50
figure_all(awake_c123,sh_awake_c123,variable,tvar,chivar,savepath,'ACC_awake_123')
figure_all(keta_c123,sh_keta_c123,variable,tvar,chivar,savepath,'ACC_keta_123')
figure_all(sleep_c123,sh_sleep_c123 ,variable,tvar,chivar,savepath,'ACC_sleep_123')

figure_all(awake_c12,sh_awake_c12,variable,tvar,chivar,savepath,'ACC_awake_12')
figure_all(keta_c12,sh_keta_c12,variable,tvar,chivar,savepath,'ACC_keta_12')
figure_all(sleep_c12,sh_sleep_c12 ,variable,tvar,chivar,savepath,'ACC_sleep_12')

figure_all(awake_c1,sh_awake_c1,variable,tvar,chivar,savepath,'ACC_awake_1')
figure_all(keta_c1,sh_keta_c1,variable,tvar,chivar,savepath,'ACC_keta_1')
figure_all(sleep_c1,sh_sleep_c1 ,variable,tvar,chivar,savepath,'ACC_sleep_1')

variable,tvar,chivar = 'auc',1,0.4
figure_all(awake_c123,sh_awake_c123,variable,tvar,chivar,savepath,'AUC_awake_123')
figure_all(keta_c123,sh_keta_c123,variable,tvar,chivar,savepath,'AUC_keta_123')
figure_all(sleep_c123,sh_sleep_c123 ,variable,tvar,chivar,savepath,'AUC_sleep_123')

figure_all(awake_c12,sh_awake_c12,variable,tvar,chivar,savepath,'AUC_awake_12')
figure_all(keta_c12,sh_keta_c12,variable,tvar,chivar,savepath,'AUC_keta_12')
figure_all(sleep_c12,sh_sleep_c12 ,variable,tvar,chivar,savepath,'AUC_sleep_12')

figure_all(awake_c1,sh_awake_c1,variable,tvar,chivar,savepath,'AUC_awake_1')
figure_all(keta_c1,sh_keta_c1,variable,tvar,chivar,savepath,'AUC_keta_1')
figure_all(sleep_c1,sh_sleep_c1 ,variable,tvar,chivar,savepath,'AUC_sleep_1')

variable,tvar,chivar = 'L',3,0
figure_all(awake_c123,sh_awake_c123,variable,tvar,chivar,savepath,'Loss_awake_123')
figure_all(keta_c123,sh_keta_c123,variable,tvar,chivar,savepath,'Loss_keta_123')
figure_all(sleep_c123,sh_sleep_c123 ,variable,tvar,chivar,savepath,'Loss_sleep_123')

figure_all(awake_c12,sh_awake_c12,variable,tvar,chivar,savepath,'Loss_awake_12')
figure_all(keta_c12,sh_keta_c12,variable,tvar,chivar,savepath,'Loss_keta_12')
figure_all(sleep_c12,sh_sleep_c12 ,variable,tvar,chivar,savepath,'Loss_sleep_12')

figure_all(awake_c1,sh_awake_c1,variable,tvar,chivar,savepath,'Loss_awake_1')
figure_all(keta_c1,sh_keta_c1,variable,tvar,chivar,savepath,'Loss_keta_1')
figure_all(sleep_c1,sh_sleep_c1 ,variable,tvar,chivar,savepath,'Loss_sleep_1')

# # fig, ax = plt.subplots(figsize=(8,6))
# # bp = dfline.groupby('Iteration').plot()

# plt.close('all')
# plt.figure()
# plt.xlabel('epoca', fontsize=20);
# plt.ylim([0, 3])

# plt.scatter(dfline['epoca'],dfline['TL'],c='blue',alpha=0.3)
# plt.scatter(dfline['epoca'],dfline['VL'],c='red',alpha=0.3)


# ax = df.plot(secondary_y=['TL', 'VL'])
# g =sns.scatterplot(x="epoca", y="value", hue="variable",
#               data=dfline,alpha=0.3)
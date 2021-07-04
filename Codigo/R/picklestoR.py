# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:49:23 2021

@author: xochipilli
"""
import pandas as pd

filename = 'TF_MCx3' # 2 3 Ph_TF_MCx1 / 2 / 3 TF_MCx1
prefix = 'Pickle_'

basepath = 'E:/tiempo/Resultados/A_ResultadosR/Permutaciones/'
Ph_MCx1 = pd.read_pickle(basepath+prefix+filename+'.p')

keta_c1,sh_keta_c1 = pd.DataFrame(Ph_MCx1['keta_c1']), pd.DataFrame(Ph_MCx1['sh_keta_c1'])
awake_c1,sh_awake_c1 = pd.DataFrame(Ph_MCx1['awake_c1']), pd.DataFrame(Ph_MCx1['sh_awake_c1'])
sleep_c1,sh_sleep_c1 =  pd.DataFrame(Ph_MCx1['sleep_c1']), pd.DataFrame(Ph_MCx1['sh_sleep_c1'])

keta_c12,sh_keta_c12 = pd.DataFrame(Ph_MCx1['keta_c12']), pd.DataFrame(Ph_MCx1['sh_keta_c12'])
awake_c12,sh_awake_c12 =  pd.DataFrame(Ph_MCx1['awake_c12']), pd.DataFrame(Ph_MCx1['sh_awake_c12'])
sleep_c12,sh_sleep_c12 =  pd.DataFrame(Ph_MCx1['sleep_c12']), pd.DataFrame(Ph_MCx1['sh_sleep_c12'])

keta_c123,sh_keta_c123 =  pd.DataFrame(Ph_MCx1['keta_c123']), pd.DataFrame(Ph_MCx1['sh_keta_c123'])
awake_c123,sh_awake_c123 =  pd.DataFrame(Ph_MCx1['awake_c123']), pd.DataFrame(Ph_MCx1['sh_awake_c123'])
sleep_c123,sh_sleep_c123 = pd.DataFrame(Ph_MCx1['sleep_c123']), pd.DataFrame(Ph_MCx1['sh_sleep_c123'])

keta_c1.to_csv(basepath +'df_'+ filename+'_'+'keta_c1.csv' )
sh_keta_c1.to_csv(basepath +'df_'+ filename+'_'+'sh_keta_c1.csv' )
awake_c1.to_csv(basepath +'df_'+ filename+'_'+'awake_c1.csv' )
sh_awake_c1.to_csv(basepath +'df_'+ filename+'_'+'sh_awake_c1.csv' )
sleep_c1.to_csv(basepath +'df_'+ filename+'_'+'sleep_c1.csv' )
sh_sleep_c1.to_csv(basepath +'df_'+ filename+'_'+'sh_sleep_c1.csv' )

keta_c12.to_csv(basepath +'df_'+ filename+'_'+'keta_c12.csv' )
sh_keta_c12.to_csv(basepath +'df_'+ filename+'_'+'sh_keta_c12.csv' )
awake_c12.to_csv(basepath +'df_'+ filename+'_'+'awake_c12.csv' )
sh_awake_c12.to_csv(basepath +'df_'+ filename+'_'+'sh_awake_c12.csv' )
sleep_c12.to_csv(basepath +'df_'+ filename+'_'+'sleep_c12.csv' )
sh_sleep_c12.to_csv(basepath +'df_'+ filename+'_'+'sh_sleep_c12.csv' ) 

keta_c123.to_csv(basepath +'df_'+ filename+'_'+'keta_c123.csv' )
sh_keta_c123.to_csv(basepath +'df_'+ filename+'_'+'sh_keta_c123.csv' )
awake_c123.to_csv(basepath +'df_'+ filename+'_'+'awake_c123.csv' )
sh_awake_c123.to_csv(basepath +'df_'+ filename+'_'+'sh_awake_c123.csv' )
sleep_c123.to_csv(basepath +'df_'+ filename+'_'+'sleep_c123.csv' )
sh_sleep_c123.to_csv(basepath +'df_'+ filename+'_'+'sh_sleep_c123.csv' ) 

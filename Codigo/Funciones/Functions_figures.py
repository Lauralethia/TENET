# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:53:11 2021

@author: xochipilli
"""
import pandas as pd
import numpy as np
from plotnine import *
import sys
import os

def get_data(basepath,filename,NC,model_complexity,prefix,UsarTF = True):
    
    ## Que componentes usar
   #CPcondition = 'comp'+'_'.join([str(int) for int in NC] )
    CPcondition = 'MCx_' +str(model_complexity) + '_comp'+'_'.join([str(int) for int in NC] )

    ## Usar TF o CP sin transformar? 
    if UsarTF == True: # Si e
      CPcondition = prefix + CPcondition 
    
    ModelosFolder =  basepath + CPcondition +'_Modelos/'
    
    savePath = ModelosFolder + filename 

    iter_df =  pd.read_csv(savePath + 'iter_df.csv')
    sh_iter_df =  pd.read_csv(savePath + 'sh_iter_df.csv')
    
    return iter_df,sh_iter_df

def figure_all(sleep_c1,sh_sleep_c1,variable,tvar,chivar,savepath,savename):
    dfline = sleep_c1[["Iteration","epoca",'T'+ variable,'V'+ variable]]
    dflinesh = sh_sleep_c1[["Iteration","epoca",'T'+ variable,'V'+ variable]]
    dfline.insert(0, "Cond", 'normal', True) 
    dflinesh.insert(0, "Cond", 'shuffled', True) 
    df3 =pd.concat( [dfline,dflinesh])
    fulldf = pd.melt(df3, id_vars=['epoca','Iteration','Cond'], value_vars=[ 'T'+ variable,'V'+ variable])
    fulldf['CF'] = fulldf['variable'].str.cat(fulldf['Cond'],sep="_")
  
    
    bp = (ggplot(fulldf)         # defining what data to use
     + aes(x='epoca',
           y='value',
          #group = 'variable',
           color= 'CF',
           shape = 'CF')    # defining what variable to use
     + geom_point(alpha = 0.09)
     #+ geom_smooth(method = "loess") # mavg loess defining the type of plot to use
     + theme_bw()
     + theme(legend_position=("top"))
     + ylim(chivar,tvar)
     + ggtitle(savename)
    )
    
    bp = bp + scale_color_manual(values=["#3843f5", "#a4a9f5","#fa0a0a", "#f5a4a4"])
    ggsave(plot = bp ,filename = savename , path = savepath)
    
    if not os.path.exists(savepath + '/svg_'):
        os.makedirs(savepath+ '/svg_')
    ggsave(plot = bp, device = "svg" ,filename = savename + '.svg', path = savepath + '/svg_')

    return bp
#savename = 'lala'

def cleanModelFolders (ModelosFolder,filename):  
    iter_df =  pd.read_csv(ModelosFolder +filename + 'iter_df.csv')
    import os, fnmatch
    listOfFiles = os.listdir(ModelosFolder)
    pattern = filename + 'It*_*E*'
    filename_vl = filename +'It'+ str(int(iter_df.loc[iter_df.idxmin()[ 'VL']]['Iteration'])) +'_Loss*'
    #filename_acc = filename +'It'+ str(int(iter_df.loc[iter_df.idxmax( )['VA']]['Iteration'])) +'*'
    #filename_auc = filename +'It'+ str(int(iter_df.loc[iter_df.idxmax( )['Vauc']]['Iteration'])) +'*'


    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            #print(entry)
            if not fnmatch.fnmatch(entry, filename_vl):
                #print(entry)
                #if not fnmatch.fnmatch(entry, filename_acc):
                 #   if not fnmatch.fnmatch(entry, filename_auc):
                os.remove(ModelosFolder + entry)
                        
    sh_iter_df =  pd.read_csv(ModelosFolder + filename +'sh_iter_df.csv')
    import os, fnmatch
    listOfFiles = os.listdir(ModelosFolder)
    pattern = 'sh_' + filename + 'It*_*E*'
    filename_vl ='sh_' +  filename +'It'+ str(int(sh_iter_df.loc[sh_iter_df.idxmin()[ 'VL']]['Iteration'])) +'_Loss*'
    #filename_acc = 'sh_' + filename +'It'+ str(int(sh_iter_df.loc[sh_iter_df.idxmax( )['VA']]['Iteration'])) +'*'
    #filename_auc ='sh_' +  filename +'It'+ str(int(sh_iter_df.loc[sh_iter_df.idxmax( )['Vauc']]['Iteration'])) +'*'

    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            if not fnmatch.fnmatch(entry, filename_vl):
                #if not fnmatch.fnmatch(entry, filename_acc):
                    #if not fnmatch.fnmatch(entry, filename_auc):
                        #print(entry)
                        os.remove(ModelosFolder + entry)

    return 

def all_cleanModelFolders(model_complexity,NC,prefix,folderdata,basepath):

    CPcondition = prefix + 'MCx_' +str(model_complexity) + '_comp'+'_'.join([str(int) for int in NC] )
    #%%##############################################################################
    
    ModelosFolder = basepath+folderdata + CPcondition +'_Modelos/'
    
    cleanModelFolders (ModelosFolder,'sleep')
    cleanModelFolders (ModelosFolder,'keta')
    cleanModelFolders (ModelosFolder,'awake')
    return



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv

import subprocess
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

"""
  Artigo Calibration models database of near infrared spectroscopy to predict agricultural soil fertility properties
    
    Recomenda executar préprocessamento em seu daaset aplicando:
      Step 1. centralização de média -> Centralizar em relação a todo o dataset ou pra cada amomstra
      Step 2. suavização
      Step 3. Normalização  |  Baseados em Médias ou em Picos MSC, ou SNV

"""
def check_csvFiles(filePath):
  if not os.path.exists(filePath):
    with open(filePath, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(['Header1', 'Header2'])  # Example headers
    file.close()
figure_counter = 0

command = ["git", "branch", "-v"]
result  = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
gitOut = result.stdout[:7]

if gitOut == "* Colab":
  path_2Use="Colab"
else:
  path_2Use=None


figure_counter = 0

if path_2Use=="Colab":
  generalPahth=(f"/home/gitFiles/Universal_pyEnv/Brasil_SpectralLib_MIR")
else:
  generalPahth=(f"./Universal_pyEnv/Brasil_SpectralLib_MIR")

SVRPahth=(f"{generalPahth}/RandomForest")
rbf=(f"{SVRPahth}/rf_env/src/")

fileName=(f"{rbf}/bestFit_RandomForest.csv")
check_csvFiles(fileName)
fileName=(f"{rbf}RandomForest_BackUp.csv")
check_csvFiles(fileName)
fileName=(f"{rbf}/NearBestR2.csv")
check_csvFiles(fileName)

def snv(dataSet):
  
  output_data = np.zeros_like(dataSet)
  for i in range(dataSet.shape[0]):
                 
    # Apply correction
    output_data[i,:] = (dataSet[i,:] - np.mean(dataSet[i,:])) / np.std(dataSet[i,:])
                                  
  return output_data

def PlotGeneric(ylabel, SamplesData, *dataSet):
  count = 0
  if len(dataSet)==1:
    dataSet1 = dataSet[0]
    count=1
  elif len(dataSet)==2:
    count=2
    dataSet1 = dataSet[0]
    dataSet2 = dataSet[1]
  elif len(dataSet)==3:
    count=3
    dataSet1 = dataSet[0]
    dataSet2 = dataSet[1]
    dataSet3 = dataSet[2]
  elif len(dataSet)==4:
    count=4
    dataSet1 = dataSet[0]
    dataSet2 = dataSet[1]
    dataSet3 = dataSet[2]
    dataSet4 = dataSet[3]
  

  wl = np.arange(1001.06, 2500, 0.964568855 )
  print(len(wl))
  
  if len(dataSet)==1:
    plt.plot(wl, dataSet1.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Absorbance")     
  
  else:
    fig, axes = plt.subplots(int(count/2), int(count/2), constrained_layout=True)
  
  if len(dataSet)>1:
    """
    if dataSet1 is not None:
      axes[0,0].plot(wl, dataSet1[0].T)
    if dataSet2 is not None:
      axes[0,1].plot(wl, dataSet2[0].T)
    if dataSet3 is not None:
      axes[1,0].plot(wl, dataSet3[0].T)
    if dataSet4 is not None:
      axes[1,1].plot(wl, dataSet4[0].T)
    """
    if dataSet1 is not None:
      axes[0,0].plot(wl, dataSet1.T)
    if dataSet2 is not None:
      axes[0,1].plot(wl, dataSet2.T)
    if dataSet3 is not None:
      axes[1,0].plot(wl, dataSet3.T)
    if dataSet4 is not None:
      axes[1,1].plot(wl, dataSet4.T)
  SampleID = np.arange(0,40, 1)
  if ylabel is not None:
    print("PlotGenerg - ", SampleID.shape, " |Sammples - ", SamplesData.shape)
    plt.figure()
    plt.bar(SampleID, SamplesData)
    plt.ylabel(ylabel)
    plt.xlabel("Samples")

    for i, v in enumerate(SamplesData):
      plt.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=10)
  
  #plt.tight_layout()
  #plt.show()

def msc(dataSet):
  ref_specValue = dataSet.mean(axis=0) 
  data_msc = np.zeros_like(dataSet)

  for i in range(dataSet.shape[0]):
    fit = np.polyfit(ref_specValue, dataSet[i,:], 1, full=True)
    # Apply correction
    data_msc[i,:] = (dataSet[i,:] - fit[0][1]) / fit[0][0] 
                                     
  return data_msc

def Centralization(dataSet):
  
  for i in np.arange(0, dataSet.shape[0]):   # linhas
    for j in np.arange(0, dataSet.shape[1]): # colunas
      dataSet[i][j] = dataSet[i].mean()
      # Ou...
      #dataSet[i][j]=dataSet.mean()

  return dataSet

def optimise_RandomForest_cv(X, y, n_comp):
    
  random_num=0
  control_print=0
  max_r2=0
  max_rpd=0
  best_randomR2=0
  best_randomRpd=0
  
  nearBest_crit = 0.50
  
  #n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes
   
  n_estimators             = [50, 100, 150, 200, 250, 500, 1000, 1500, 2000]     
  criterion                = ["squared_error", "absolute_error", "poisson" ]  
  max_depth                = [5, 25, 125, 625]
  min_samples_split        = [2, 4, 8, 16, 32, 64] 
  min_samples_leaf         = [1, 2, 4, 8, 16, 32, 64, 128]
  min_weight_fraction_leaf = [0, 0.01, 0.1, 0.2, 0.4, 0.5] 
  max_features             = [None, "auto", "log2", "sqrt"]
  max_leaf_nodes           = [None, 2, 4, 8, 16, 32, 64, 128]

  n_estimators_R2              = 0 
  criterion_R2                 = 0 
  max_depth_R2                 = 0 
  min_samples_split_R2         = 0 
  min_samples_leaf_R2          = 0 
  min_weight_fraction_leaf_R2  = 0 
  max_features_R2              = 0 
  max_leaf_nodes_R2            = 0 

  n_estimators_RPD             = 0 
  criterion_RPD                = 0 
  max_depth_RPD                = 0 
  min_samples_split_RPD        = 0 
  min_samples_leaf_RPD         = 0 
  min_weight_fraction_leaf_RPD = 0 
  max_features_RPD             = 0 
  max_leaf_nodes_RPD           = 0 

  n_estimators_i               = 0 
  criterion_i                  = 0 
  max_depth_i                  = 0 
  min_samples_split_i          = 0 
  min_samples_leaf_i           = 0 
  min_weight_fraction_leaf_i   = 0 
  max_features_i               = 0 
  max_leaf_nodes_i             = 0 

  bestFit_file=(f"{rbf}bestFit_RandomForest.csv")
  NearbestFit_R2_file=(f"{rbf}NearBestR2.csv")
  Bkp_file=(f"{rbf}RandomForest_BackUp.csv")
  
  #"""
  with open(bestFit_file, 'r') as file:
    lines=file.readlines()

  AllData = lines[-3].strip().split(',');
  
  print("\
  Random Forest MIR Backup  Best Fit Analyse\n\
  \t\tMaxRPD:          ",AllData[0],"\n\
  \t\tMaxR2:           ",AllData[1],"\n\
  \t\tbest_randomR2:   ",AllData[2],"\n\
  \t\tBest RPD random: ",AllData[3],"\n\
  \t\tn_estimators_R2                  ",AllData[4],"\n\
  \t\tcriterion_R2                     ",AllData[5],"\n\
  \t\tmax_depth_R2                     ",AllData[6],"\n\
  \t\tmin_samples_split_R2             ",AllData[7],"\n\
  \t\tmin_samples_leaf_R2              ",AllData[8],"\n\
  \t\tmin_weight_fraction_leaf_R2      ",AllData[9],"\n\
  \t\tmax_features_R2                  ",AllData[10],"\n\
  \t\tmax_leaf_nodes_R2                ",AllData[11],"\n\
  \t\tn_estimators_RPD                 ",AllData[12],"\n\
  \t\tcriterion_RPD                    ",AllData[13],"\n\
  \t\tmax_depth_RPD                    ",AllData[14],"\n\
  \t\tmin_samples_split_RPD            ",AllData[15],"\n\
  \t\tmin_samples_leaf_RPD             ",AllData[16],"\n\
  \t\tmin_weight_fraction_leaf_RPD     ",AllData[17],"\n\
  \t\tmax_features_RPD                 ",AllData[18],"\n\
  \t\tmax_leaf_nodes_RPD               ",AllData[19],"\n\
  \t\tRandom Numbers:                  ",AllData[20],"\n")
  file.close()
  
  if float(AllData[0]) > max_rpd:
    max_r2  = float(AllData[1])
    max_rpd = float(AllData[0])

  with open(Bkp_file, 'r') as file:
    lines=file.readlines()

  AllData = lines[-3].strip().split(',');
  print("\
  Random Forest MIR Backup Analyse\n\
  \t\tn_estimators_i                 ",AllData[0],"\n\
  \t\tcriterion_i                    ",AllData[1],"\n\
  \t\tmax_depth_i                    ",AllData[2],"\n\
  \t\tmin_samples_split_i            ",AllData[3],"\n\
  \t\tmin_samples_leaf_i             ",AllData[4],"\n\
  \t\tmin_weight_fraction_leaf_i     ",AllData[5],"\n\
  \t\tmax_features_i                 ",AllData[6],"\n\
  \t\tmax_leaf_nodes_i               ",AllData[7],"\n\
  \t\tRandomNumber ",AllData[8],"\n")
  file.close()
  
  if int(float(AllData[8])) >= random_num:
    n_estimators_i                 =int(float(AllData[0])) 
    criterion_i                    =int(float(AllData[1]))
    max_depth_i                    =int(float(AllData[2]))
    min_samples_split_i            =int(float(AllData[3]))
    min_samples_leaf_i             =int(float(AllData[4]))
    min_weight_fraction_leaf_i     =int(float(AllData[5]))
    max_features_i                 =int(float(AllData[6]))
    max_leaf_nodes_i               =int(float(AllData[7]))
    random_num     = int(float(AllData[8]))
  #"""
  print("Random Forest MIR Begin")
  while random_num < 2147483647:
    #print("Layer 1")
    while  n_estimators_i      < len( n_estimators):              
      #print("Layer 2")
      while  criterion_i         < len( criterion):                    
        #print("Layer 3")
        while  max_depth_i         < len( max_depth):                    
          #print("Layer 4")
          while  min_samples_split_i < len( min_samples_split):   
           #print("Layer 5")
            while  min_samples_leaf_i  < len( min_samples_leaf):
              #print("Layer 6")
              while  min_weight_fraction_leaf_i < len(min_weight_fraction_leaf):
                #print("Layer 7")
                while  max_features_i   < len( max_features):            
                  while  max_leaf_nodes_i < len( max_leaf_nodes):       
                    #print("Random Forest MIR ", random_num)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=random_num)

                    # Initialize and fit the PLS regression model
                    RF_model = RandomForestRegressor(n_estimators=n_estimators[n_estimators_i], criterion=criterion[criterion_i],\
                                                     random_state=0, max_depth=max_depth[max_depth_i], \
                                                     min_samples_split=min_samples_split[min_samples_split_i],\
                                                     min_samples_leaf=min_samples_leaf[min_samples_split_i],\
                                                     min_weight_fraction_leaf=min_weight_fraction_leaf[min_weight_fraction_leaf_i],\
                                                     max_features=max_features[max_features_i], max_leaf_nodes=max_leaf_nodes[max_leaf_nodes_i]\
                                                     )
                    RF_model.fit(X_train, y_train)
                    #print(epsilon)

                    # Predict the target variable on the test set
                    y_cv = RF_model.predict(X_test)

                    # Calculate scores
                    r2 = r2_score(y_test, y_cv)
                    mse = mean_squared_error(y_test, y_cv)
                    rpd = y_test.std()/np.sqrt(mse)
                    
                    
                    if r2>max_r2:
                      max_r2 = r2               
                      best_randomR2 = random_num
                      n_estimators_R2             =  n_estimators[n_estimators_i]            
                      criterion_R2                =  criterion[criterion_i] 
                      max_depth_R2                =  max_depth[max_depth_i]
                      min_samples_split_R2        =  min_samples_split[min_samples_split_i]
                      min_samples_leaf_R2         =  min_samples_leaf[min_samples_leaf_i]
                      min_weight_fraction_leaf_R2 =  min_weight_fraction_leaf[min_weight_fraction_leaf_i]
                      max_features_R2             =  max_features[max_features_i]
                      max_leaf_nodes_R2           =  max_leaf_nodes[max_leaf_nodes_i]


                    if rpd>max_rpd:
                      max_rpd = rpd
                      best_randomRpd= random_num
                      n_estimators_RPD             = n_estimators               [n_estimators_i]
                      criterion_RPD                = criterion                  [criterion_i]
                      max_depth_RPD                = max_depth                  [max_depth_i]
                      min_samples_split_RPD        = min_samples_split          [min_samples_split_i]
                      min_samples_leaf_RPD         = min_samples_leaf           [min_samples_leaf_i]
                      min_weight_fraction_leaf_RPD = min_weight_fraction_leaf   [min_weight_fraction_leaf_i]
                      max_features_RPD             = max_features               [max_features_i]
                      max_leaf_nodes_RPD           = max_leaf_nodes             [max_leaf_nodes_i]

                      with open(bestFit_file, 'a') as file:
                        data=(f"max_rpd,max_r2,R2_rdn,RPD_rdn, n_estimators_R2, criterion_R2, max_depth_R2, min_samples_split_R2, min_samples_leaf_R2, min_weight_fraction_leaf_R2, max_features_R2, max_leaf_nodes_R2, n_estimators_RPD, criterion_RPD, max_depth_RPD, min_samples_split_RPD, min_samples_leaf_RPD, min_weight_fraction_leaf_RPD, max_features_RPD, max_leaf_nodes_RPD ,random_numbers\n {max_rpd},{max_r2},{best_randomR2},{best_randomRpd}, {n_estimators_R2}, {criterion_R2}, {max_depth_R2}, {min_samples_split_R2}, {min_samples_leaf_R2}, {min_weight_fraction_leaf_R2}, {max_features_R2}, {max_leaf_nodes_R2}, {n_estimators_RPD}, {criterion_RPD}, {max_depth_RPD}, {min_samples_split_RPD}, {min_samples_leaf_RPD}, {min_weight_fraction_leaf_RPD}, {max_features_RPD}, {max_leaf_nodes_RPD},{random_num}\n\
                        ==============================================================\n\n")
                        file.write(data)
                      file.close()
                    
                    elif r2>=max_r2*nearBest_crit and rpd>=max_rpd*nearBest_crit :
                      with open(NearbestFit_R2_file, 'a') as file:
                        data=(f"n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, random_num, max_r2, r2, rpd, max_rpd \n {n_estimators_i}, {criterion_i}, {max_depth_i}, {min_samples_split_i}, {min_samples_leaf_i}, {min_weight_fraction_leaf_i}, {max_features_i}, {max_leaf_nodes_i},{random_num},{max_r2}, {r2}, {rpd}, {max_rpd}  \n\
                        ******************************************\n\n")
                        file.write(data)
                      file.close()

                    max_leaf_nodes_i+=1          
                  max_leaf_nodes_i=0          
                  max_features_i+=1            #L7
                max_features_i=0            
                min_weight_fraction_leaf_i+=1 #L6
              #Bkp
              print("**... Random Forest MIR-> ", random_num)
              with open(Bkp_file, 'w') as file:
                data=(f"n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, random_num\n {n_estimators_i}, {criterion_i}, {max_depth_i}, {min_samples_split_i}, {min_samples_leaf_i}, {min_weight_fraction_leaf_i}, {max_features_i}, {max_leaf_nodes_i},{random_num}\n\
                ******************************************\n\n")
                file.write(data)
              file.close()
              min_weight_fraction_leaf_i=0
              min_samples_leaf_i+=1   #L5     
            min_samples_leaf_i=0        
            min_samples_split_i+=1 #L4       
          min_samples_split_i=0       
          max_depth_i+=1      #L3         
        max_depth_i=0               
        criterion_i+=1     #L2          
      criterion_i=0               
      n_estimators_i+=1 #L1
    n_estimators_i=0
    random_num+=1      #L0
  
    if random_num==2147483647:
      random_num=0
  
  random_values.append(best_randomRpd)
  random_values.append(best_randomR2)
  return (y_test, y_cv, r2, mse, rpd, random_values)

def plot_metrics(vals, ylabel, objective):
  global figure_counter
  if figure_counter % 3 == 0:
    plt.figure()

  figure_counter+=1
  subplot_number = figure_counter % 3
  if subplot_number == 0:
    subplot_number=4
  
  plt.subplot(2,2, subplot_number)
  xticks=np.arange(1,5,1)
  plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
  if objective=='min':
    idx = np.argmin(vals)
  else:
    idx = np.argmax(vals)
  
  plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

  plt.xlabel('Number of PLS components')
  plt.xticks = xticks
  plt.ylabel(ylabel)
  plt.title('PLS')
  
  return xticks[idx]
  #plt.show()
                                                                               
soil_sheet     = pd.read_csv("../data_bases/BrLib_MIR.csv")

soil_sheet.head()
#spectral_sheet.replace(to_replace=',', value='.')
#PLSR assigns
Y = soil_sheet["P_mgkg"].values
X = soil_sheet.values[:, 21:]


print(X.shape, " - ", Y.shape, " - ", soil_sheet.shape)
print(Y)

#, " | ", X[0][0], X[0][len(X)-1], " | ", spectral_sheet.values.shape)
#

# Pré Processamento Step 1  | Centralização de Média

print("Original")
#PlotGeneric(X)
print("Centralizado")
Xorg = X.copy()
"""
wl = np.arange(1001.06, 2500, 0.964568855 )
plt.plot(wl, Xorg.T)
plt.xlabel("Wavelengths (nm)")
plt.ylabel("Absorbance")
plt.show()
"""
#X = Centralization(X)
Xmc= X.copy()
#PlotGeneric("Concentracoes", Y, X)

# Pré Processamento Step 2  |  Suavização 
X   = savgol_filter(X, 25, polyorder=2, deriv=0)
Xsg = X.copy()

X1   = savgol_filter(X, 25, polyorder=2, deriv=1)
Xsg1 = X1.copy()

X2 = savgol_filter(X, 25, polyorder=2, deriv=2)
print("Suavizado")
#PlotGeneric(Xorg)

# Pré Processamento Step 3  |  Normalização
"""
X   = msc(X) 
X1  = msc(X1) 
X2  = msc(X2) 
"""

print("Normalizado")
X   = snv(X) 
X1  = snv(X1) 
X2  = snv(X2)

Xnorm = X.copy()
#PlotGeneric(None, None, Xorg, Xmc, Xsg, Xnorm)


r2s = []
mses = []
rpds = []

r2s_sg2 = []
mses_sg2 = []
rpds_sg2 = []

r2s_sg1 = []
mses_sg1 = []
rpds_sg1 = []

pca_variance = np.arange(0.75, 1, 0.05)
X_pcaAll=[]
for varia in pca_variance:

  #pca_comps=PCA(varia)
  #X_pca = pca_comps.fit_transform(X1)
  #X_pcaAll.append(pd.DataFrame(data=pca_comps.inverse_transform(X_pca)))

  #y_test, y_cv, r2, mse, rpd, random_1 = optimise_pls_cv(X, Y, n_comp)
  #r2s.append(r2)
  #mses.append(mse)
  #rpds.append(rpd)
  
  y_test, y_cv, r2, mse, rpd, random_1= optimise_RandomForest_cv(X2, Y, 1)
  r2s_sg1.append(r2)
  mses_sg1.append(mse)
  rpds_sg1.append(rpd)

  #X2_pca = pca_comps.fit_transform(X2)
  #y_test, y_cv, r2, mse, rpd, random_3= optimise_RandomForest_cv(X2, Y, 1)
  #r2s_sg2.append(r2)
  #mses_sg2.append(mse)
  #rpds_sg2.append(rpd)

print(X1.shape)
print(X_pcaAll[0].shape, X_pcaAll[1].shape,X_pcaAll[2].shape)
print(X_pcaAll[3].shape)
PlotGeneric(None, None, X1, X_pcaAll[0], X_pcaAll[1], X_pcaAll[2])
PlotGeneric(None, None, X_pcaAll[3])
#best_pls=plot_metrics(mses, 'MSE PLSR', 'min')
#best_pls=plot_metrics(rpds, 'RPD PLSR', 'max')
#best_pls=plot_metrics(r2s, 'R2 PLSR', 'max')

"""
best_plsSG2=plot_metrics(mses_sg1, 'MSE PLSR + SG1', 'min')
best_plsSG1=plot_metrics(rpds_sg1, 'RPD PLSR + SG1', 'max')
best_plsSG2=plot_metrics(r2s_sg1, 'R2 PLSR + SG1', 'max')

best_plsSG2=plot_metrics(mses_sg2, 'MSE PLSR + SG2', 'min')
best_plsSG2=plot_metrics(rpds_sg2, 'RPD PLSR + SG2', 'max')
best_plsSG2=plot_metrics(r2s_sg2, 'R2 PLSR + SG2', 'max')

#y_test, y_cv, r2, mse, rpd, random_4 = optimise_RandomForest_cv(X1, Y, best_plsSG1)
print(best_plsSG1)
print(random_1)
#print(random_2)
print(random_3)
print(y_test)
plt.figure()
plt.scatter(y_test, y_cv, color='red')
plt.plot(y_test, y_test, '-g', label='Expected regression line')
z = np.polyfit(y_test, y_cv, 1)
plt.plot(np.polyval(z, y_test), y_test, color='blue', label='Predicted regression line')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.plot()
"""
plt.show()

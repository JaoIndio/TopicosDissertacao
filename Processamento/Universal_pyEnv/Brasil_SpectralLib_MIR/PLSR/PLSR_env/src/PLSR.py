#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
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
figure_counter = 0
command = ["git", "branch", "-v"]
result  = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
gitOut = result.stdout[:7]

if gitOut == "* Colab":
  path_2Use="Colab":

figure_counter = 0

if path_2Use=="Colab":
  generalPahth=(f"/home/gitFiles/Universal_pyEnv/Brasil_SpectralLib_MIR")
else:
  generalPahth=(f"./Universal_pyEnv")
SVRPahth=(f"{generalPahth}")
rbf=(f"{SVRPahth}/PLSR/PLSR_env/src/")


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
  

  wl = np.arange(600, 3994, 4.985315712 )
  print(len(wl))
  
  if len(dataSet)==1:
    plt.plot(wl, dataSet1.T)
    plt.xlabel("Wavelengths (cm-1) INVERTIDO")
    plt.ylabel("Absorbance")     
  elif count==2:
    fig, axes = plt.subplots(2, 1, constrained_layout=True)
  else:
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
  
  if len(dataSet)>1:
    if count>2:
      if dataSet1 is not None:
        axes[0,0].plot(wl, dataSet1[0].T)
      if dataSet2 is not None:
        axes[0,1].plot(wl, dataSet2[0].T)
      if dataSet3 is not None:
        axes[1,0].plot(wl, dataSet3[0].T)
      if dataSet4 is not None:
        axes[1,1].plot(wl, dataSet4[0].T)
    else:
      axes[0].plot(wl, dataSet1[0].T)
      axes[1].plot(wl, dataSet2[0].T)

  
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

def optimise_pls_cv(X, y, n_comp):
  
  random_num=0
  control_print=0
  rpd_list = []
  r2_list= []
  random_values= []
  max_r2=0
  max_rpd=0
  best_randomR2=0
  best_randomRpd=0
  n_comp=1
  n_compBestR2=0
  n_compBestRPD=0
  bestFit_file=(f"{rbf}bestFit_PLSR.csv")
  Bkp_file=(f"{rbf}PLSR_BackUp.csv")
  #random_num = 99296
  #"""
  with open(bestFit_file, 'r') as file:
    lines=file.readlines()

  AllData = lines[-3].strip().split(',');
  
  print("\
  PLSR BrLib Backup  Best Fit Analyse\n\
  \t\tMaxRPD:          ",AllData[0],"\n\
  \t\tMaxR2:           ",AllData[1],"\n\
  \t\tbest_randomR2:   ",AllData[2],"\n\
  \t\tBest RPD random: ",AllData[3],"\n\
  \t\tBest NumComp R2: ",AllData[4],"\n\
  \t\tBest NumComp RPD:",AllData[5],"\n\
  \t\tRandom Numbers:  ",AllData[6],"\n")
  #file.close()
  
  if float(AllData[0]) > max_rpd:
    max_r2  = float(AllData[1])
    max_rpd = float(AllData[0])

  with open(Bkp_file, 'r') as file:
    lines=file.readlines()

  AllData = lines[-3].strip().split(',');
  print("\
  PLSR BrLib Bakup Analyse\n\
  \t\tNum Comp     ",AllData[0],"\n\
  \t\tRandomNumber ",AllData[1],"\n")
  #file.close()
  
  if int(AllData[1]) > random_num:
    n_comp         = int(AllData[0]) 
    random_num     = int(AllData[1])
  #"""
  
  while random_num<2147483647:
    while n_comp<20:
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=random_num)

      # Initialize and fit the PLS regression model
      pls_model = PLSRegression(n_components=n_comp)
      pls_model.fit(X_train, y_train)

      # Predict the target variable on the test set
      y_cv = pls_model.predict(X_test)

      # Calculate scores
      r2 = r2_score(y_test, y_cv)
      mse = mean_squared_error(y_test, y_cv)
      rpd = y_test.std()/np.sqrt(mse)
      
      if r2>max_r2:
        max_r2 = r2
        best_randomR2 = random_num
        n_compBestR2=n_comp

      if rpd>max_rpd:
        max_rpd = rpd
        best_randomRpd= random_num
        n_compBestRPD=n_comp
        with open(bestFit_file, 'a+') as file:
          data=(f"max_rpd,max_r,bestRumbers,BestR2,BestRPD,NcompR2,NcompRPD,random_number\n\
          {max_rpd}, {max_r2}, {best_randomR2}, {best_randomRpd},\
          {n_compBestR2},{n_compBestRPD},{random_num}\n\
          ==============================================================\n\n")
          file.write(data)
      n_comp+=1
       
    if control_print==75:
      print("**.. PLSR -> ", random_num)
      control_print=0
      with open(Bkp_file, 'w+') as file:
        data=(f"NumComp,random_num\n\
        {n_comp},{random_num}\n\
        ******************************************\n\n")
        file.write(data)
      file.close()

    control_print+=1
    random_num+=1
    n_comp=1
    if random_num==2147483647:
      random_num=0
  #file.close()
  #random_values.append(best_randomRpd)
  #random_values.append(best_randomR2)
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

if path_2Use=="Colab":
  soil_sheet     = pd.read_csv("/home/DecisionTree_Colab/BrLib_MIR.csv")
else:
  soil_sheet     = pd.read_csv("../data_bases/BrLib_MIR.csv")


soil_sheet.head()
#spectral_sheet.replace(to_replace=',', value='.')
#PLSR assigns
Y = soil_sheet["P_mgkg"].values
X = soil_sheet.values[:, 21:]


print(X.shape, " - ", Y.shape, " - ", soil_sheet.shape)
print(Y)

#, " | ", X[0][0], X[0][len(X)-1], " | ", spectral_sheet.values.shape)
#"""

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
Xsg2 = X2.copy()
print("Suavizado")
#PlotGeneric(Xorg)

# Pré Processamento Step 3  |  Normalização
"""
X   = msc(X) 
X1  = msc(X1) 
X2  = msc(X2) 
"""

print("Normalizado")
#"""
X   = snv(X) 
X1  = snv(X1) 
X2  = snv(X2)
#"""

Xnorm = X.copy()
PlotGeneric(None, None, Xorg, Xmc, Xsg, Xnorm)
PlotGeneric(None, None, Xsg1, Xsg2)


r2s = []
mses = []
rpds = []

r2s_sg2 = []
mses_sg2 = []
rpds_sg2 = []

r2s_sg1 = []
mses_sg1 = []
rpds_sg1 = []

xticks = np.arange(1, 21)
#for n_comp in xticks:
"""
y_test, y_cv, r2, mse, rpd, random_1 = optimise_pls_cv(X, Y, 1)
r2s.append(r2)
mses.append(mse)
rpds.append(rpd)
# 

y_test, y_cv, r2, mse, rpd, random_1= optimise_pls_cv(X1, Y, 1)
r2s_sg1.append(r2)
mses_sg1.append(mse)
rpds_sg1.append(rpd)
r2s_sg1.append(r2)
mses_sg1.append(mse)
rpds_sg1.append(rpd)
r2s_sg1.append(r2)
mses_sg1.append(mse)
rpds_sg1.append(rpd)
r2s_sg1.append(r2)
mses_sg1.append(mse)
rpds_sg1.append(rpd)
"""
y_test, y_cv, r2, mse, rpd, random_3= optimise_pls_cv(X2, Y, 1)
r2s_sg2.append(r2)
mses_sg2.append(mse)
rpds_sg2.append(rpd)
r2s_sg2.append(r2)
mses_sg2.append(mse)
rpds_sg2.append(rpd)
r2s_sg2.append(r2)
mses_sg2.append(mse)
rpds_sg2.append(rpd)
r2s_sg2.append(r2)
mses_sg2.append(mse)
rpds_sg2.append(rpd)

#best_pls=plot_metrics(mses, 'MSE PLSR', 'min')
#best_pls=plot_metrics(rpds, 'RPD PLSR', 'max')
#best_pls=plot_metrics(r2s, 'R2 PLSR', 'max')

best_plsSG2=plot_metrics(mses_sg1, 'MSE PLSR + SG1', 'min')
best_plsSG1=plot_metrics(rpds_sg1, 'RPD PLSR + SG1', 'max')
best_plsSG2=plot_metrics(r2s_sg1, 'R2 PLSR + SG1', 'max')

best_plsSG2=plot_metrics(mses_sg2, 'MSE PLSR + SG2', 'min')
best_plsSG2=plot_metrics(rpds_sg2, 'RPD PLSR + SG2', 'max')
best_plsSG2=plot_metrics(r2s_sg2, 'R2 PLSR + SG2', 'max')

y_test, y_cv, r2, mse, rpd, random_4 = optimise_pls_cv(X1, Y, best_plsSG1)
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

plt.show()

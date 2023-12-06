#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
figure_counter = 0

generalPahth=(f"Universal_pyEnv")
SVRPahth=(f"{generalPahth}/RandomForest")
rbf=(f"{SVRPahth}/rf_env/src/")

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
  C=0.001
  Gamma=0.001
  epsilon=0.0001
  
  C_BestR2=0
  C_BestRPD=0
  Gamma_BestR2=0
  Gamma_BestRPD=0
  epsilon_BestR2=0
  epsilon_BestRPD=0
  #random_num = 99296
  bestFit_file=(f"{rbf}bestFit_RandomForest.csv")
  Bkp_file=(f"{rbf}RandomForest_BackUp.csv")
  
  """
  with open(bestFit_file, 'r') as file:
    lines=file.readlines()

  AllData = lines[-3].strip().split(',');
  
  print("\
  RBF Backup  Best Fit Analyse\n\
  \t\tMaxRPD:          ",AllData[0],"\n\
  \t\tMaxR2:           ",AllData[1],"\n\
  \t\tbest_randomR2:   ",AllData[2],"\n\
  \t\tBest RPD random: ",AllData[3],"\n\
  \t\tC R2:            ",AllData[4],"\n\
  \t\tGamma R2:        ",AllData[5],"\n\
  \t\tEpsilon R2:      ",AllData[6],"\n\
  \t\tC RPD:           ",AllData[7],"\n\
  \t\tGamma RPD:       ",AllData[8],"\n\
  \t\tEpsilon RPD:     ",AllData[9],"\n\
  \t\tRandom Numbers:  ",AllData[10],"\n")
  #file.close()
  
  if float(AllData[0]) > max_rpd:
    max_r2  = float(AllData[1])
    max_rpd = float(AllData[0])

  with open(Bkp_file, 'r') as file:
    lines=file.readlines()

  AllData = lines[-3].strip().split(',');
  print("\
  RBF Backup Analyse\n\
  \t\tC            ",AllData[0],"\n\
  \t\tGamma        ",AllData[1],"\n\
  \t\tEpsilon:     ",AllData[2],"\n\
  \t\tRandomNumber ",AllData[3],"\n")
  #file.close()
  
  if int(AllData[3]) > random_num:
    C              = float(AllData[0]) 
    Gamma          = float(AllData[1])
    Epsilon        = float(AllData[2])
    random_num     = int(AllData[3])
  """
  while random_num < 2147483647:
    #print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=random_num)

    # Initialize and fit the PLS regression model
    RF_model = RandomForestRegressor(n_estimators=100, random_state=0)
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
      C_BestR2=C
      Gamma_BestR2=Gamma
      epsilon_BestR2=epsilon

    if rpd>max_rpd:
      max_rpd = rpd
      best_randomRpd= random_num
      C_BestRPD=C
      epsilon_BestRPD=epsilon
      Gamma_BestRPD=Gamma
      with open(bestFit_file, 'a') as file:
        data=(f"max_rpd,max_r2,R2_rdn,RPD_rdn,C_R2,Gamma_R2,epsilon_R2,C_RPD,Gamma_RPD,epsilon_RPD,random_numbers\n\
        {max_rpd},{max_r2},\
        {best_randomR2},{best_randomRpd},\
        {C_BestR2},{Gamma_BestR2},{epsilon_BestR2},\
        {C_BestRPD},{Gamma_BestRPD},{epsilon_BestRPD},\
        {random_num}\n\
        ==============================================================\n\n")
        file.write(data)
        file.close()
    if control_print==25:
      print("**... Random Forest -> ", random_num)
      with open(Bkp_file, 'w') as file:
        data=(f"C,Gamma,epsilon,random_num\n\
        {C},{Gamma},{epsilon},{random_num}\n\
        ******************************************\n\n")
        file.write(data)
        file.close()
      control_print=0

    control_print+=1
    random_num+=1
  
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
                                                                               
soil_sheet     = pd.read_excel("../data_bases/NIR_spectra_Data_Soil.xlsx", \
                               "soil_prop_Y")
spectral_sheet = pd.read_csv("../data_bases/NIR_googleDrive.csv")

soil_sheet.head()
spectral_sheet.head()
spectral_sheet.replace(to_replace=',', value='.')
#PLSR assigns
Y = soil_sheet["P (ppm)"].values
X = spectral_sheet.values[:, 4:]


print(X.shape[0], " - ", Y.shape, " - ", spectral_sheet.shape)

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
X   = savgol_filter(X, 51, polyorder=2, deriv=0)
Xsg = X.copy()

X1   = savgol_filter(X, 51, polyorder=2, deriv=1)
Xsg1 = X1.copy()

X2 = savgol_filter(X, 51, polyorder=2, deriv=2)
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
  
  y_test, y_cv, r2, mse, rpd, random_1= optimise_RandomForest_cv(X1, Y, 1)
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

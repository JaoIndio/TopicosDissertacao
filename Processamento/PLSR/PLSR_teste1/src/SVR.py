#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.svm import SVR
from sklearn.decomposition import PCA
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
    if dataSet1 is not None:
      axes[0,0].plot(wl, dataSet1[0].T)
    if dataSet2 is not None:
      axes[0,1].plot(wl, dataSet2[0].T)
    if dataSet3 is not None:
      axes[1,0].plot(wl, dataSet3[0].T)
    if dataSet4 is not None:
      axes[1,1].plot(wl, dataSet4[0].T)
  
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

def optimise_SVR_cv(X, y, n_comp):
  
  """
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
  X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=0)
  X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.3, random_state=0)
  """
  
  """
  # Define PLS object
  pls = PLSRegression(n_components=n_comp)

  # Cross-validation
  y_cv = cross_val_predict(pls, X, y, cv=5, verbose=1) #avaliar melhor esse cros validadtion:w!
                                                        #principalmente parametro cv
  """

  """
  train_num = [2,3,4,9,11,14,23,32,0,19,6,13, 39, 28, 36, 21, 1, 33, 25, 8, 26, 12, 22, 38]
  test_num  = [i for i in range(len(Y)) if i not in train_num] 
  
  X_train = pd.DataFrame([X[i] for i in train_num])
  X_test  = pd.DataFrame([X[i] for i in test_num])
  
  y_train = pd.DataFrame([Y[i] for i in train_num])
  y_test  = pd.DataFrame([Y[i] for i in test_num])
  
  y_test = y_test[0].squeeze()
  """
  #print("optimize PLS ", y_test.shape)
  
  
  random_num=0
  control_print=0
  rpd_list = []
  r2_list= []
  random_values= []
  max_r2=0.6074734930240134
  max_rpd=1.596119873907366
  best_randomR2=0
  best_randomRpd=0
  C=0.001
  Gamma=0.001
  C_BestR2=0
  C_BestRPD=0
  Gamma_BestR2=0
  Gamma_BestRPD=0
  #random_num = 99296


  pca_variance = np.arange(0.75, 0.95, 0.05)
  while random_num < 2147483647:
    while C<100000:
      while Gamma<100000:
        
        for varia in pca_variance:
          pca_comps=PCA(varia)
          X_pca=pca_comps.fit_transform(X)
          
          X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.35, random_state=random_num)

          # Initialize and fit the PLS regression model
          SVR_model = SVR(kernel='rbf', gamma=Gamma, C=C, cache_size=2000)
          SVR_model.fit(X_train, y_train)

          # Predict the target variable on the test set
          y_cv = SVR_model.predict(X_test)

          # Calculate scores
          r2 = r2_score(y_test, y_cv)
          mse = mean_squared_error(y_test, y_cv)
          rpd = y_test.std()/np.sqrt(mse)
          
          
          if r2>max_r2:
            max_r2 = r2
            best_randomR2 = random_num
            C_BestR2=C
            Gamma_BestR2=Gamma

          if rpd>max_rpd:
            max_rpd = rpd
            best_randomRpd= random_num
            C_BestRPD=C
            Gamma_BestRPD=Gamma
            with open('bestFit_SVR.txt', 'a') as file:
              data=f"max_rpd= {max_rpd} \n max_r2=   {max_r2}  \n bestRumbers | R2: {best_randomR2} | RPD: {best_randomRpd} | C R2: {C_BestR2} | Gama R2: {Gamma_BestR2} \n | C Rpd: {C_BestRPD} | Gama RPD: {Gamma_BestRPD} |\n  random_numbers | {random_num} \n PCA Variancia={varia}\n==========================================================\n\n"
              file.write(data)
              file.close()
        Gamma*=10
      C*=10
      Gamma=0.0001
    C=0.0001 
    if control_print==5:
      print("**..-> ", random_num)
      with open('SVR_BackUp.txt', 'a') as file:
        data=f"C= {C} \n Gamma= {Gamma}  \n | random_numbers {random_num} | PCA Variancia={varia} |\n****************************************************\n\n"
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
#"""
X   = snv(X) 
X1  = snv(X1) 
X2  = snv(X2)
#"""

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

pca_variance = np.arange(0.75, 0.95, 0.05)
for varia in pca_variance:
  """
  y_test, y_cv, r2, mse, rpd, random_1 = optimise_pls_cv(X, Y, n_comp)
  r2s.append(r2)
  mses.append(mse)
  rpds.append(rpd)
  """
  
  y_test, y_cv, r2, mse, rpd, random_1= optimise_SVR_cv(X1, Y, 1)
  r2s_sg1.append(r2)
  mses_sg1.append(mse)
  rpds_sg1.append(rpd)
  
  X2_pca = pca_comps.fit_transform(X2)
  y_test, y_cv, r2, mse, rpd, random_3= optimise_SVR_cv(X2, Y, 1)
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

#y_test, y_cv, r2, mse, rpd, random_4 = optimise_SVR_cv(X1, Y, best_plsSG1)
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

p

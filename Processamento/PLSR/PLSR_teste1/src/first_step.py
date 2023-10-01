#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
def snv(dataSet):
  
  output_data = np.zeros_like(dataSet)
  for i in range(dataSet.shape[0]):
                 
    # Apply correction
    output_data[i,:] = (dataSet[i,:] - np.mean(dataSet[i,:])) / np.std(dataSet[i,:])
                                  
  return output_data

def PlotGeneric(dataSet):
  wl = np.arange(1001.06, 2500, 0.964568855 )
  print(len(wl))
  
  plt.plot(wl, dataSet.T)
  plt.xlabel("Wavelengths (nm)")
  plt.ylabel("Absorbance")
  
  plt.show()


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
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Initialize and fit the PLS regression model
  pls_model = PLSRegression(n_components=n_comp)
  pls_model.fit(X_train, y_train)

  # Predict the target variable on the test set
  y_cv = pls_model.predict(X_test)

  # Calculate scores
  r2 = r2_score(y_test, y_cv)
  mse = mean_squared_error(y_test, y_cv)
  rpd = y_test.std()/np.sqrt(mse)
       
  return (y_cv, r2, mse, rpd)

def plot_metrics(vals, ylabel, objective):
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

  plt.show()
                                                                               
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
PlotGeneric(X)
print("Centralizado")
X = Centralization(X)
PlotGeneric(X)

# Pré Processamento Step 2  |  Suavização 
X  = savgol_filter(X, 11, polyorder=3, deriv=0)
X1 = savgol_filter(X, 11, polyorder=3, deriv=1)
X2 = savgol_filter(X, 11, polyorder=3, deriv=2)
print("Suavizado")
PlotGeneric(X)

# Pré Processamento Step 3  |  Normalização
#X  = msc(X) 
print("Normalizado")
X  = snv(X) 
#PlotGeneric(X)

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
for n_comp in xticks:
  y_cv, r2, mse, rpd = optimise_pls_cv(X, Y, n_comp)
  r2s.append(r2)
  mses.append(mse)
  rpds.append(rpd)
  
  y_cv, r2, mse, rpd = optimise_pls_cv(X1, Y, n_comp)
  r2s_sg1.append(r2)
  mses_sg1.append(mse)
  rpds_sg1.append(rpd)

  y_cv, r2, mse, rpd = optimise_pls_cv(X2, Y, n_comp)
  r2s_sg2.append(r2)
  mses_sg2.append(mse)
  rpds_sg2.append(rpd)


plot_metrics(mses, 'MSE PLSR', 'min')
plot_metrics(rpds, 'RPD PLSR', 'max')
plot_metrics(r2s, 'R2 PLSR', 'max')

plot_metrics(mses_sg1, 'MSE PLSR + SG1', 'min')
plot_metrics(rpds_sg1, 'RPD PLSR + SG1', 'max')
plot_metrics(r2s_sg1, 'R2 PLSR + SG1', 'max')

plot_metrics(mses_sg2, 'MSE PLSR + SG2', 'min')
plot_metrics(rpds_sg2, 'RPD PLSR + SG2', 'max')
plot_metrics(r2s_sg2, 'R2 PLSR + SG2', 'max')




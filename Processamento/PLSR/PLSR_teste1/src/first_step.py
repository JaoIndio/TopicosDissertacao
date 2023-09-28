#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

def optimise_pls_cv(X, y, n_comp):
  # Define PLS object
  pls = PLSRegression(n_components=n_comp)

  # Cross-validation
  y_cv = cross_val_predict(pls, X, y, cv=10) #avaliar melhor esse cros validadtion
                                             #principalmente parametro cv

  # Calculate scores
  r2 = r2_score(y, y_cv)
  mse = mean_squared_error(y, y_cv)
  rpd = y.std()/np.sqrt(mse)
       
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
spectral_sheet = pd.read_excel("../data_bases/NIR_spectra_Data_Soil.xlsx", \
                               "spectra_data_X")

soil_sheet.head()
spectral_sheet.head()

#PLSR assigns
Y = soil_sheet["P (ppm)"].values
X = spectral_sheet.values[:, 4:]

Y.shape
X.shape

print(X.shape, " - ", Y.shape)
#"""
wl = np.arange(1001.06, 2500, 1.46954902)
print(len(wl))

"""
plt.plot(wl, X.T)
plt.xlabel("Wavelengths (nm)")
plt.ylabel("Absorbance")

plt.show()

"""
#X1 = savgol_filter(X, 17, polyorder=4, deriv=1)
X2 = savgol_filter(X, 17, polyorder=4, deriv=2)

r2s = []
mses = []
rpds = []

r2s_sg2 = []
mses_sg2 = []
rpds_sg2 = []

r2s_sg1 = []
mses_sg1 = []
rpds_sg1 = []

xticks = np.arange(1, 41)
for n_comp in xticks:
  y_cv, r2, mse, rpd = optimise_pls_cv(X, y, n_comp)
  r2s.append(r2)
  mses.append(mse)
  rpds.append(rpd)
  
  y_cv, r2, mse, rpd = optimise_pls_cv(X1, y, n_comp)
  r2s_sg1.append(r2)
  mses_sg1.append(mse)
  rpds_sg1.append(rpd)

  y_cv, r2, mse, rpd = optimise_pls_cv(X2, y, n_comp)
  r2s_sg2.append(r2)
  mses_sg2.append(mse)
  rpds_sg2.append(rpd)





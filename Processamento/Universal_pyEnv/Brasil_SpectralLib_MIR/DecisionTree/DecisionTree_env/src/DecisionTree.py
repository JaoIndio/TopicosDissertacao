#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
import subprocess
from sklearn.tree import DecisionTreeRegressor
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

command = ["git", "branch", "-v"]
result  = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
gitOut = result.stdout[:7]

if gitOut == "* Colab":
  path_2Use="Colab"
else:
  path_2Use=None

figure_counter = 0

generalPahth=(f"Universal_pyEnv/Brasil_SpectralLib_MIR")
SVRPahth=(f"{generalPahth}/DecisionTree")
rbf=(f"{SVRPahth}/DecisionTree_env/src/")

fileName=(f"{rbf}/bestFit_DecisionTree.csv")
check_csvFiles(fileName)
fileName=(f"{rbf}DecisionTree_BackUp.csv")
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

def optimise_DecisionTree_cv(X, y, n_comp):
    
  random_num=0
  random_check= [67, 68, 69, 1402, 4186, 9063, 15919, 67537]
  control_print=0
  max_r2=0
  max_rpd=0
  best_randomR2=0
  best_randomRpd=0

  nearBest_crit = 0.50
  
  criterior                 = ["squared_error", "friedman_mse", "absolute_error", "poisson"]
  splitter                  = ["best", "random"]                    
  max_depth                 = [5, 25, 125, 625]                  
  min_samples_split         = [2, 4, 8, 16, 32, 64]          
  min_samples_leaf          = [1, 2, 4, 8, 16, 32, 64, 128]        
  features                  = [None, "log2", "sqrt"]      
  min_weight_fraction_leaf  = [0, 0.01, 0.1, 0.2, 0.4, 0.5]    
  max_leaf_nodes            = [None, 2, 4, 8, 16, 32, 64, 128]  
  
  crit_BestR2                        = 0         
  splitter_BestR2                    = 0       
  max_depth_BestR2                   = 0  
  min_samples_split_BestR2           = 0
  min_samples_leaf_BestR2            = 0
  features_BestR2                    = 0  
  min_weight_fraction_leaf_BestR2    = 0
  max_leaf_nodes_BestR2              = 0
  
  crit_BestRPD                      = 0
  splitter_BestRPD                  = 0
  max_depth_BestRPD                 = 0 
  min_samples_split_BestRPD         = 0
  min_samples_leaf_BestRPD          = 0
  features_BestRPD                  = 0
  min_weight_fraction_leaf_BestRPD  = 0
  max_leaf_nodes_BestRPD            = 0
 
  #random_num = 99296
  NearbestFit_R2_file=(f"{rbf}NearBestR2.csv")
  bestFit_file=(f"{rbf}bestFit_DecisionTree.csv")
  Bkp_file=(f"{rbf}DecisionTree_BackUp.csv")
  
  #"""
  with open(bestFit_file, 'r') as file:
    lines=file.readlines()

  AllData = lines[-3].strip().split(',');
  
  print("\
  BrLib-MIR Decision Tree Backup  Best Fit Analyse\n\
  \t\tMaxRPD:          ",AllData[0],"\n\
  \t\tMaxR2:           ",AllData[1],"\n\
  \t\tsplitter_BestR2                   ",AllData[3],"\n\
  \t\tmax_depth_BestR2                : ",AllData[4],"\n\
  \t\tcrit_BestR2                       ",AllData[2],"\n\
  \t\tmin_samples_split_BestR2          ",AllData[5],"\n\
  \t\tmin_samples_leaf_BestR2           ",AllData[6],"\n\
  \t\tfeatures_BestR2                   ",AllData[7],"\n\
  \t\tmin_weight_fraction_leaf_BestR2   ",AllData[8],"\n\
  \t\tmax_leaf_nodes_BestR2             ",AllData[9],"\n\
  \t\tcrit_BestRPD                      ",AllData[10],"\n\
  \t\tsplitter_BestRPD                  ",AllData[11],"\n\
  \t\tmax_depth_BestRPD                 ",AllData[12],"\n\
  \t\tmin_samples_split_BestRPD         ",AllData[13],"\n\
  \t\tmin_samples_leaf_BestRPD          ",AllData[14],"\n\
  \t\tfeatures_BestRPD                  ",AllData[15],"\n\
  \t\tmin_weight_fraction_leaf_BestRPD  ",AllData[16],"\n\
  \t\tmax_leaf_nodes_BestRPD            ",AllData[17],"\n\
  \t\tRandom Numbers:                  ",AllData[18],"\n")
  file.close()
  
  if float(AllData[0]) > max_rpd:
    max_r2  = float(AllData[1])
    max_rpd = float(AllData[0])
  
  
  with open(Bkp_file, 'r') as file:
    lines=file.readlines()

  AllData = lines[-3].strip().split(',');
  
  print("\
  BrLib-MIR Decision Tree BackUp Analyse\n\
    		crit", 		        AllData[0],"\n\
        split" ,          AllData[1],"\n\
        m_depth" , 	      AllData[2],"\n\
        min_samp_split",  AllData[3],"\n\
        samp_splitLea",   AllData[4],"\n\
        feat",            AllData[5],"\n\
        weight",          AllData[6],"\n\
        leafNode",        AllData[7],"\n\
        random_num",      AllData[8], "\n")

  file.close()
  #"""
  crit= 		      0 
  split=          0
  m_depth=  	    0
  min_samp_split= 0
  samp_splitLeaf= 0
  feat=           0
  weight=         0
  leafNode=       0
  random_num=     0
  
  #"""
  if float(AllData[8]) >= random_num:
    crit= 		        int(float(AllData[0]))
    split=            int(float(AllData[1]))
    m_depth=  	      int(float(AllData[2]))
    min_samp_split =  int(float(AllData[3]))
    samp_splitLeaf =  int(float(AllData[4]))
    feat=             int(float(AllData[5]))
    weight=           int(float(AllData[6]))
    leafNode=         int(float(AllData[7]))
    random_num=       int(float(AllData[8]))
  #"""

  progress_execution=0
  progress_percentage=0
  total_per_random = len(criterior)* len(splitter)* \
    len(max_depth)* len(min_samples_split)*len(min_samples_leaf)* len(features)*len(min_weight_fraction_leaf)*len(max_leaf_nodes)

  # tem q fazer um loop de whiles pra associar com o backup
  while random_num < 2147483647:
    #print("**... Decision Tree MIR random----------------> ", random_num)
    while crit < len(criterior):
      #print("**... Decision Tree MIR     crit")
      #print("**... Decision Tree MIR split")
      while m_depth < len(max_depth):
        #print("**... Decision Tree MIR           depth ", m_depth)
        #print(f"Progress per random Depth ", progress_percentage," %")
        while min_samp_split < len(min_samples_split):
          #print(f"Progress per random Sampsplit ", progress_percentage," %")
          #print("**... Decision Tree MIR     sampSplit")
          while samp_splitLeaf < len(min_samples_leaf):
            #print("**... Decision Tree MIR SampLeaf")
            while feat < len(features):
              #print("**... Decision Tree MIR            Feat")
              while weight < len(min_weight_fraction_leaf):
                #print("**... Decision Tree MIR     weight")
                while leafNode < len(max_leaf_nodes):        
                  while split < len(splitter):
                    """
                    print(f"Progress per random Leaf ", progress_percentage," %")
                    print("**...  weight", weight)
                    print("**...  feat ", feat)
                    print("**...  leaf ", leafNode)
                    print("**...  samp_splitLeaf", samp_splitLeaf)
                    print("**...  minSampsplit ", min_samp_split)
                    print("**...  m_depth ", m_depth)
                    """
                    #print("**... Decision Tree MIR Node")
                    #print("**... Decision Tree MIR random----------------> ", random_num)
                    #print(i)
                    progress_execution+=1
                    progress_percentage = (progress_execution / total_per_random) * 100

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=random_num)

                    # Initialize and fit the DecisionTree regression modela
                    DT_model = DecisionTreeRegressor(criterion=criterior[crit], splitter=splitter[split], max_depth=max_depth[m_depth], \
                                                     min_samples_split=min_samples_split[min_samp_split], min_samples_leaf=min_samples_leaf[samp_splitLeaf], \
                                                     min_weight_fraction_leaf=min_weight_fraction_leaf[weight], max_features=features[feat], max_leaf_nodes=max_leaf_nodes[leafNode], \
                                                     random_state=10)
                    DT_model.fit(X_train, y_train)
                    #print("\n\nParamnetros \n==========================================")
                    #print(random_num)
                    #print("n features:     ",DT_model.n_features_in_)
                    #print("node Count:     ",DT_model.tree_.node_count)
                    #print("max depth:      ",DT_model.tree_.max_depth)
                    #print("samples/node:   ",DT_model.tree_.n_node_samples)
                    #print("weight/samples: ",DT_model.tree_.weighted_n_node_samples)

                    # Predict the target variable on the test set
                    y_cv = DT_model.predict(X_test)

                    # Calculate scores
                    r2 = r2_score(y_test, y_cv)
                    mse = mean_squared_error(y_test, y_cv)
                    rpd = y_test.std()/np.sqrt(mse)
                    
                    #print("R2:  ",r2)
                    #print("RPD: ",rpd)
                    #print("MSE: ",mse)
                    #print("\n=========================")
                    
                    
                    if r2>max_r2:
                      max_r2 = r2
                      best_randomR2 = random_num

                      crit_BestR2                     = criterior               [crit          ]     
                      splitter_BestR2                 = splitter                [split         ]    
                      max_depth_BestR2                = max_depth               [m_depth       ]   
                      min_samples_split_BestR2        = min_samples_split       [min_samp_split]  
                      min_samples_leaf_BestR2         = min_samples_leaf        [samp_splitLeaf]    
                      features_BestR2                 = features                [feat          ]     
                      min_weight_fraction_leaf_BestR2 = min_weight_fraction_leaf[weight        ]    
                      max_leaf_nodes_BestR2           = max_leaf_nodes          [leafNode      ]     
                    #print("r2: ", r2);
                    #print("rpd: ", rpd);
                    #print("near MAX r2: ", max_r2*nearBest_crit);
                    #print("near MAX rpd: ", max_rpd*nearBest_crit);
                    elif r2>=max_r2*nearBest_crit and rpd>=max_rpd*nearBest_crit :
                       
                      with open(NearbestFit_R2_file, 'a') as file:
                        data=(f"criterior,splitter,max_depth,min_samples_split,min_samples_leaf,features,min_weight_fraction_leaf, max_leaf_nodes, random_num, max_r2, r2, rpd, max_rpd, \n\
                  {crit}, {split}, {m_depth}, {min_samp_split}, {samp_splitLeaf}, {feat}, {weight}, {leafNode}, {random_num}, {max_r2}, {r2}, {rpd}, {max_rpd}\n\
 *****  ***********************************\n\n")
                        file.write(data)
                      file.close() 

                    if rpd>max_rpd:
                      max_rpd = rpd
                      best_randomRpd= random_num
                      
                      crit_BestRPD                     = criterior               [crit          ]  
                      splitter_BestRPD                 = splitter                [split         ]  
                      max_depth_BestRPD                = max_depth               [m_depth       ]  
                      min_samples_split_BestRPD        = min_samples_split       [min_samp_split]
                      min_samples_leaf_BestRPD         = min_samples_leaf        [samp_splitLeaf]  
                      features_BestRPD                 = features                [feat          ]  
                      min_weight_fraction_leaf_BestRPD = min_weight_fraction_leaf[weight        ]  
                      max_leaf_nodes_BestRPD           = max_leaf_nodes          [leafNode      ]  
                      
                      with open(bestFit_file, 'a') as file:
                        data=(f"max_rpd,max_r2,criterio_R2,splitter_BestR2,max_depth_BestR2,min_samples_split_BestR2,min_samples_leaf_BestR2,features_BestR2,min_weight_fraction_leaf_BestR2, max_leaf_nodes_BestR2,criterio_BestRPD,splitter_BestRPD,max_depth_BestRPD,min_samples_split_BestRPD,min_samples_leaf_BestRPD,features_BestRPD,min_weight_fraction_leaf_BestRPD,max_leaf_nodes_BestRPD,random_num\n\
         {max_rpd}, {max_r2}, {crit_BestR2}, {splitter_BestR2}, {max_depth_BestR2}, {min_samples_split_BestR2}, {min_samples_leaf_BestR2}, {features_BestR2}, {min_weight_fraction_leaf_BestR2}, {max_leaf_nodes_BestR2}, {crit_BestRPD}, {splitter_BestRPD}, {max_depth_BestRPD}, {min_samples_split_BestRPD}, {min_samples_leaf_BestRPD}, {features_BestRPD}, {min_weight_fraction_leaf_BestRPD}, {max_leaf_nodes_BestRPD}, {random_num}\n\
         =============================================================\n\n")
                        file.write(data)
                        file.close()
                        
                      #"""
                    split+=1
                  #split
                  split=0
                  leafNode+=1
                #leafNode
                leafNode=0  
                weight+=1
              #weight
              weight=0
              feat+=1
            #feat
            feat=0
            samp_splitLeaf+=1
          #samp_splitLeaf
          samp_splitLeaf=0
          min_samp_split+=1
        #min_samp_split
        print("**... Decision Tree MIR-> | Crit | maxDepth | random |", crit,m_depth, random_num)
        with open(Bkp_file, 'w') as file:
          data=(f"criterior,splitter,max_depth,min_samples_split,min_samples_leaf,features,min_weight_fraction_leaf, max_leaf_nodes, random_num\n\
                  {crit}, {split}, {m_depth}, {min_samp_split}, {samp_splitLeaf}, {feat}, {weight}, {leafNode}, {random_num}\n\
 *****  ***********************************\n\n")
          file.write(data)
        file.close()
        min_samp_split=0
        m_depth+=1
      #m_depth
      m_depth=0
      #if control_print==2400:
      #"""
        #"""        
      crit+=1
    #Crit
    crit=0
    random_num+=1
  
    if random_num==2147483647:
      random_num=0
    #Random
  print("Done")
  while True:
    a=1
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
  
  y_test, y_cv, r2, mse, rpd, random_1= optimise_DecisionTree_cv(X2, Y, 1)
  r2s_sg1.append(r2)
  mses_sg1.append(mse)
  rpds_sg1.append(rpd)

  #X2_pca = pca_comps.fit_transform(X2)
  #y_test, y_cv, r2, mse, rpd, random_3= optimise_DecisionTree_cv(X2, Y, 1)
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

#y_test, y_cv, r2, mse, rpd, random_4 = optimise_DecisionTree_cv(X1, Y, best_plsSG1)
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

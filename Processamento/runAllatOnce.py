#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

generalPath=(f"Universal_pyEnv")

SVRPath=(f"{generalPath}/SVR")
linear_PCA=(f"{SVRPath}/linear_PCA/linearPCA_env/src/SVR_linear.py")
rbf_PCA=(f"{SVRPath}/rbf_PCA/rbfPCA_env/src/SVR.py")
rbf=(f"{SVRPath}/rbf/rbf_env/src/SVR.py")
dt=(f"{generalPath}/DecisionTree/DecisionTree_env/src/DecisionTree.py")
rf=(f"{generalPath}/RandomForest/rf_env/src/RandomForest.py")
GradBoost=(f"{generalPath}/GradientBoost/GradBoost_env/src/GradBoost.py")

BrLib_MIR_path       =(f"{generalPath}/Brasil_SpectralLib_MIR")
SVRPath              =(f"{BrLib_MIR_path}/SVR")
linear_PCA_BrLib_MIR =(f"{SVRPath}/linear_PCA/linearPCA_env/src/SVR_linear.py")
rbf_PCA_BrLib_MIR    =(f"{SVRPath}/rbf_PCA/rbfPCA_env/src/SVR.py")
rbf_BrLib_MIR        =(f"{SVRPath}/rbf/rbf_env/src/SVR.py")
#dt_BrLib_MIR         =(f"{BrLib_MIR_path}/DecisionTree/DecisionTree_env/src/DecisionTree.py")   
rf_BrLib_MIR         =(f"{BrLib_MIR_path}/RandomForest/rf_env/src/RandomForest.py")            
#GradBoost_BrLib_MIR  =(f"{BrLib_MIR_path}/GradientBoost/GradBoost_env/src/GradBoost.py")      
PLSR_BrLib_MIR       =(f"{BrLib_MIR_path}/PLSR/PLSR_env/src/PLSR.py")                             

#BrLib_NIR_path       =(f"{generalPath}/Brasil_SpectralLib_NIR")
#SVRPath              =(f"{generalPath}/SVR")
#linear_PCA_BrLib_NIR =(f"{SVRPath}/linear_PCA/linearPCA_env/src/SVR_linear.py")
#rbf_PCA_BrLib_NIR    =(f"{SVRPath}/rbf_PCA/rbfPCA_env/src/SVR.py")
#rbf_BrLib_NIR        =(f"{SVRPath}/rbf/rbf_env/src/SVR.py")
#dt_BrLib_NIR         =(f"{BrLib_NIR_path}/DecisionTree/DecisionTree_env/src/DecisionTree.py"
#rf_BrLib_NIR         =(f"{BrLib_NIR_path}/RandomForest/rf_env/src/RandomForest.py")         
#GradBoost_BrLib_NIR  =(f"{BrLib_NIR_path}/GradientBoost/GradBoost_env/src/GradBoost.py")    
#PLSR_BrLib_NIR       =(f"{BrLib_NIR_path}/PLSR/PLSR_env/src/PLSR.py")                       


# List of scripts to run
scripts_to_run = [ # DataSet com 40 amostras """ \
                      # linear_PCA, rbf_PCA, rbf, \
  #DataSet da Biblioteca Nacional MIR
                      linear_PCA_BrLib_MIR, \
                      rbf_PCA_BrLib_MIR,   \
                      rbf_BrLib_MIR,       \
                      #dt_BrLib_MIR,        \
                      rf_BrLib_MIR,        \
                      #GradBoost_BrLib_MIR, \
                      PLSR_BrLib_MIR     \

                # DataSet da Biblioteca Nacional NIR
                     #linear_PCA_BrLib_MI,\   
                     #rbf_PCA_BrLib_MIR,  \
                     #rbf_BrLib_MIR,      \
                     #dt_BrLib_MIR,       \
                     #rf_BrLib_MIR,       \
                     #GradBoost_BrLib_MIR,\
                     #PLSR_BrLib_MIR,     \
                 ]

# Run each script in a separate subprocess
processes = []
for script in scripts_to_run:
  process = subprocess.Popen(['python', script])
  processes.append(process)

# Wait for all subprocesses to complete
for process in processes:
  process.wait()


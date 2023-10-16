#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

generalPahth=(f"Universal_pyEnv")
SVRPahth=(f"{generalPahth}/SVR")
linear_PCA=(f"{SVRPahth}/linear_PCA/linearPCA_env/src/SVR_linear.py")
rbf_PCA=(f"{SVRPahth}/rbf_PCA/rbfPCA_env/src/SVR.py")
rbf=(f"{SVRPahth}/rbf/rbf_env/src/SVR.py")
# List of scripts to run
scripts_to_run = [linear_PCA, rbf_PCA, rbf]

# Run each script in a separate subprocess
processes = []
for script in scripts_to_run:
  process = subprocess.Popen(['python', script])
  processes.append(process)

# Wait for all subprocesses to complete
for process in processes:
  process.wait()


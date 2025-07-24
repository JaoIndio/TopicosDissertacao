import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
print("Loading Data")
#data = pd.read_csv('Draft2.txt', sep='\t', header=None, names=["time", "V_Tout", "V_tin"])
data = pd.read_csv("Draft2.txt", sep='\t')
print(data.columns)

print("IEE Define")
# Define a general IEEE-like style
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",  # IEEE allows serif (Times New Roman)
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.dpi": 300,
})
xticks = np.arange(0, 2.1, 0.2)

# -------- Plot V_tout --------
print("Plots")
fig1= plt.figure(figsize=(8,4))
print("Figure")
plt.plot(data["time"]*1000, data["V(V_Tout,Vcm)"], color='tab:blue', linewidth=0.75,  label=r'$V_{Tout}$')
print("plot")
plt.xlabel("Time (s)")
print("X label")
plt.ylabel("Voltage (V)")
print("Y label")
#plt.title(r"$V_{Tout}$ vs Time")
print("Tittle")
plt.legend()
plt.grid(True)
plt.xticks(xticks, [f"{x:.1f}" for x in xticks])
#print("Layout")
plt.tight_layout()
#print("Saving")
plt.savefig("V_tout_plot.png", dpi=600)  # Save in publication quality
#plt.show()

# -------- Plot V_tin --------
print("Plots 2")
fig2 = plt.figure(figsize=(8,4))
plt.plot(data["time"]*1000, data["V(V_tin,Vcm)"], color='tab:red', linewidth=0.75,label=r'$V_{Tin}$')
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
#plt.title(r"$V_{Tin}$ vs Time")
plt.legend()
plt.savefig("V_tin_plot.png", dpi=600)
plt.grid(True)
plt.xticks(xticks, [f"{x:.1f}" for x in xticks])
plt.tight_layout()
plt.show()


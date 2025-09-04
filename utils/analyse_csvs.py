import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'serif'
# path to your folder of CSVs
folder = "eigenspectrum_csvs"

plt.figure()

for file in os.listdir(folder):
    if file.endswith(".csv"):
        filepath = os.path.join(folder, file)
        df = pd.read_csv(filepath)

        # assumes first column = x, second column = y
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        y2 = df.iloc[:, 2]
        
        label = os.path.splitext(file)[0]  # file name without extension
        plt.plot(x, y, label=label)
        plt.plot(x, y2, label=f"{label} (absolute values)")

plt.xlabel("Steps (x1000)")
plt.ylabel("Effective condition number")
plt.legend()
plt.yscale("log")
plt.xlim((0,26))
plt.tight_layout()
plt.savefig("compare_effective_condition_numbers")

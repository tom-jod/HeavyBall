import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# path to your folder of CSVs
folder = "eigenspectrum_csvs/benchmarks"

plt.figure(figsize=(8, 5))
color_map = {
    "Linear Regression": "C0",   # blue 
    "MNIST Small MLP": "C1",                 # orange
    "MNIST Deep MLP": "C2",        # green
    "Deeper Sigmoid": "C3",        # green
    "MNIST": "C4",
    "SVHN Pooled": "C5",
    "SVHN CNN": "C0",
    "SVHN ": "C5",
    "Tolstoi": "C2",
    "C10-wide": "C7",
    "C100": "C3",

}

for file in os.listdir(folder):
    if file.endswith(".csv"):
        filepath = os.path.join(folder, file)
        df = pd.read_csv(filepath)

        # assumes first column = x, second column = mean, third column = variance
        x = df.iloc[:, 0]
        y = df.iloc[:, 10]   # mean condition number 1
        var = df.iloc[:, 11] # variance 3
        std = var**0.5      # standard deviation

        label = os.path.splitext(file)[0].replace("_", " ")  # file name without extension
        color = color_map.get(label, None)
       
        plt.plot(x, y, label=label, color=color)


        # shaded area = mean ± std
        plt.fill_between(x, y - std, y + std, alpha=0.2, color=color)

plt.xlabel("Steps (x1000)")
plt.ylabel("Effective condition number")
plt.legend(loc = "lower center", ncols=5)
plt.yscale("log")
plt.xlim((0, 9))
#plt.ylim((1,600))
#plt.ylim((50,100000))
plt.tight_layout()
plt.savefig("compare_effective_condition_numbers_benchmarks", dpi=500)

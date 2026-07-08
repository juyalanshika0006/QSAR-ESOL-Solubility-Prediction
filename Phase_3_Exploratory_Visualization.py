import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("delaney-processed.csv")

plt.figure(figsize=(8,5))

plt.hist(
    df["measured log solubility in mols per litre"],
    bins=30
)

plt.xlabel("Measured Log Solubility")

plt.ylabel("Number of Molecules")

plt.title("Distribution of Solubility")

#plt.show()

plt.figure(figsize=(8,5))

plt.hist(
    df["Molecular Weight"],
    bins=30
)

plt.xlabel("Molecular Weight")

plt.ylabel("Number of Molecules")

plt.title("Molecular Weight Distribution")

#plt.show()

plt.figure(figsize=(8,5))

plt.hist(
    df["measured log solubility in mols per litre"],
    bins=30
)

plt.xlabel("LogP")

plt.ylabel("Number of Molecules")

plt.title("LogP Distribution")

plt.show()

plt.figure(figsize=(8,5))

plt.hist(
    df["Polar Surface Area"],
    bins=30
)

plt.xlabel("Polar Surface Area")

plt.ylabel("Number of Molecules")

plt.title("TPSA Distribution")

#plt.show()

plt.figure(figsize=(7,5))

plt.boxplot(df["Molecular Weight"])

plt.title("Molecular Weight")

#plt.show()

plt.figure(figsize=(7,5))

plt.boxplot(df["measured log solubility in mols per litre"])

plt.title("Solubility")

plt.show()

plt.figure(figsize=(7,5))

plt.boxplot(df["Polar Surface Area"])

plt.title("Polar Surface Area")

#plt.show()

#correlation Heatmap
descriptor_df = df[
    [
        "Molecular Weight",
        "measured log solubility in mols per litre",
        "Polar Surface Area",
        "Number of H-Bond Donors",
        "Number of Rotatable Bonds",
        "Number of Rings",
        "measured log solubility in mols per litre"
    ]
]
corr = descriptor_df.corr(numeric_only=True)
plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="icefire",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

#plt.show()

#Descriptor Distribution Plots
plt.figure(figsize=(7,5))

plt.scatter(
    df["Molecular Weight"],
    df["measured log solubility in mols per litre"]
)

plt.xlabel("Molecular Weight")

plt.ylabel("Solubility")

plt.title("MW vs Solubility")

#plt.show()


plt.figure(figsize=(7,5))

plt.scatter(
    df["Polar Surface Area"],
    df["measured log solubility in mols per litre"]
)

plt.xlabel("Polar Surface Area")

plt.ylabel("Solubility")

plt.title("Polar Surface Area vs Solubility")

#plt.show()

#Pairplot
print(descriptor_df.columns)
print(descriptor_df.dtypes)
for col in descriptor_df.columns:
    print(col, type(descriptor_df[col].iloc[0]))
pairplot_df = descriptor_df.drop(columns=["Fingerprint"])

sns.pairplot(pairplot_df, corner=True)
plt.show()    
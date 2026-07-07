import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("delaney-processed.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["smiles"].duplicated().sum())

largest=df.loc[df["Molecular Weight"].idxmax()]
print(largest)

smallest=df.loc[df["Molecular Weight"].idxmin()]
print(smallest)

print(df["Molecular Weight"].mean())

print(df["Molecular Weight"].describe())

plt.figure(figsize=(8,5))
plt.hist(df["Molecular Weight"], bins=30, color='pink', edgecolor='black')
plt.xlabel("Measured Log Solubilty")
plt.ylabel("Number of Molecules")
plt.title("Distribution of Molecular Weight")
plt.show()

highest=df.loc[df["measured log solubility in mols per litre"].idxmax()]
print("highest:", highest)

lowest=df.loc[df["measured log solubility in mols per litre"].idxmin()]
print("lowest:", lowest)

plt.figure(figsize=(7,5))
plt.boxplot(df["Molecular Weight"],)
plt.title("Molecular Weight Boxplot")
plt.show()

plt.figure(figsize=(7,5))
plt.boxplot(df["measured log solubility in mols per litre"])
plt.title("Measured Log Solubility Boxplot")
plt.show()

top5=df.nlargest(5, "Molecular Weight")
print(top5[["smiles", "Molecular Weight"]])

bottom5=df.nsmallest(5, "Molecular Weight")
print(bottom5[["smiles", "Molecular Weight"]])

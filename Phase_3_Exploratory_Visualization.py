import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
df = pd.read_csv("delaney-processed.csv")


# Solubility Distribution

plt.figure(figsize=(8,5))

plt.hist(
    df["measured log solubility in mols per litre"],
    bins=30
)

plt.xlabel("Measured Log Solubility")
plt.ylabel("Number of Molecules")
plt.title("Distribution of Solubility")

plt.show()


# Molecular Weight Distribution

plt.figure(figsize=(8,5))

plt.hist(
    df["Molecular Weight"],
    bins=30
)

plt.xlabel("Molecular Weight")
plt.ylabel("Number of Molecules")
plt.title("Molecular Weight Distribution")

plt.show()


# Polar Surface Area Distribution

plt.figure(figsize=(8,5))

plt.hist(
    df["Polar Surface Area"],
    bins=30
)

plt.xlabel("Polar Surface Area")
plt.ylabel("Number of Molecules")
plt.title("TPSA Distribution")
plt.show()


# Molecular Weight Boxplot

plt.figure(figsize=(7,5))

plt.boxplot(df["Molecular Weight"])

plt.title("Molecular Weight")

plt.show()


# Solubility Boxplot

plt.figure(figsize=(7,5))

plt.boxplot(df["measured log solubility in mols per litre"])

plt.title("Solubility")

plt.show()


# Polar Surface Area Boxplot

plt.figure(figsize=(7,5))

plt.boxplot(df["Polar Surface Area"])

plt.title("Polar Surface Area")

plt.show()


# Correlation Heatmap

descriptor_df = df[
    [
        "Molecular Weight",
        "measured log solubility in mols per litre",
        "Polar Surface Area",
        "Number of H-Bond Donors",
        "Number of Rotatable Bonds",
        "Number of Rings"
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

plt.show()


# Molecular Weight vs Solubility

plt.figure(figsize=(7,5))

plt.scatter(
    df["Molecular Weight"],
    df["measured log solubility in mols per litre"]
)

plt.xlabel("Molecular Weight")
plt.ylabel("Solubility")
plt.title("MW vs Solubility")

plt.show()


# Polar Surface Area vs Solubility

plt.figure(figsize=(7,5))

plt.scatter(
    df["Polar Surface Area"],
    df["measured log solubility in mols per litre"]
)

plt.xlabel("Polar Surface Area")
plt.ylabel("Solubility")
plt.title("Polar Surface Area vs Solubility")

plt.show()
# Pairplot
pairplot_df = descriptor_df.rename(columns={
    "Molecular Weight": "MolWt",
    "measured log solubility in mols per litre": "Solubility",
    "Polar Surface Area": "TPSA",
    "Number of H-Bond Donors": "HBD",
    "Number of Rotatable Bonds": "RotBonds",
    "Number of Rings": "Rings"
})

sns.pairplot(
    pairplot_df,
    corner=True,
    height=3.5
)

plt.show()

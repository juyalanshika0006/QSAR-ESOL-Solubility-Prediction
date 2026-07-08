import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

df=pd.read_csv("delaney-processed.csv")
df["Molecule"]=df["smiles"].apply(Chem.MolFromSmiles)

invalid=df["Molecule"].isnull().sum()
print("Molecule:",invalid)

invalid_rows=df[df["Molecule"].isnull()]
print("Invalid Rows:",invalid_rows)

df=df[df["Molecule"].notnull()]

print(df.shape)
print(df["Molecule"].isnull().sum())

print(df[["smiles", "Molecule"]].head())

#Calculate molecular descriptors
df["MolecularWeight"]=df["Molecule"].apply(Descriptors.MolWt)
df["LogP"] = df["Molecule"].apply(Crippen.MolLogP)
df["Number of H-Bond Donors"]=df["Molecule"].apply(Descriptors.NumHDonors)
df["Number of Rings"] = df["Molecule"].apply(rdMolDescriptors.CalcNumRings)
df["Number of Rotatable Bond"] = df["Molecule"].apply(rdMolDescriptors.CalcNumRotatableBonds)
df["Polar Surface Area"] = df["Molecule"].apply(rdMolDescriptors.CalcTPSA)


#Generating Fingerprints
generator=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

df["Fingerprint"]=df["Molecule"].apply(generator.GetFingerprint)

#Convert Fingerprints to bit vectors
fingerprints_bits=[]
for fp in df["Fingerprint"]:
    fingerprints_bits.append(list(fp))
fingerprint_df = pd.DataFrame(fingerprints_bits)

#Combine the fingerprint bit vectors with the original dataframe
descriptor_df = df[
    [
        "MolecularWeight",
        "LogP",
        "Polar Surface Area",
        "Number of H-Bond Donors",
        "Number of Rotatable Bond",
        "Number of Rings"
    ]
]    
X=pd.concat([descriptor_df, fingerprint_df], axis=1)

#preprocessing the target variable
y = df["measured log solubility in mols per litre"]

#save the processed data to CSV files
processed_df = pd.concat([descriptor_df, fingerprint_df], axis=1)

processed_df["Solubility"] = y

processed_df.to_csv("ESOL_processed.csv", index=False)
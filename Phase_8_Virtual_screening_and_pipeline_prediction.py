import joblib
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

# Load trained model
best_xgb_model = joblib.load("best_xgboost.pkl")

# Load molecules to screen
df = pd.read_csv("new_molecules.csv")

# Create Morgan Fingerprint Generator 
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=1024
)

# Feature names (same order as training)
columns = [
    "MolecularWeight",
    "LogP",
    "Polar Surface Area",
    "Number of H-Bond Donors",
    "Number of Rotatable Bond",
    "Number of Rings",
] + [str(i) for i in range(1024)]


def predict_solubility(smiles):
    """
    Predict solubility of one molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.nan

    # Calculate descriptors
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)

    # Morgan Fingerprint
    fp = generator.GetFingerprint(mol)
    fingerprint = list(fp)

    # Descriptor values
    descriptor_values = [
        mw,
        logp,
        tpsa,
        hbd,
        rotatable_bonds,
        ring_count
    ]

    # Combine features
    feature_vector = descriptor_values + fingerprint

    # Convert to DataFrame
    feature_df = pd.DataFrame(
        [feature_vector],
        columns=columns
    )

    # Prediction
    prediction = best_xgb_model.predict(feature_df)[0]

    return prediction



# Virtual Screening
predictions = []
classes = []

for smiles in df["SMILES"]:

    pred = predict_solubility(smiles)

    predictions.append(pred)

    if pd.isna(pred):
        classes.append("Invalid SMILES")

    elif pred > 0:
        classes.append("Highly Soluble")

    elif pred > -1:
        classes.append("Soluble")

    elif pred > -3:
        classes.append("Moderately Soluble")

    else:
        classes.append("Poorly Soluble")

# Save results
df["Predicted_Solubility"] = predictions
df["Solubility_Class"] = classes

df.to_csv(
    "predicted_solubility.csv",
    index=False
)

print(df)

print("\nPrediction Complete!")
print("Results saved as predicted_solubility.csv")
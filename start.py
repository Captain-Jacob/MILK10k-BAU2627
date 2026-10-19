import pandas as pd

# Load your submission file
df = pd.read_csv("submission.csv")

# Rename the 'lesion' column to 'image' (or 'lesion_id')
df = df.rename(columns={"lesion": "image"})

# Save it back out
df.to_csv("submission.csv", index=False)
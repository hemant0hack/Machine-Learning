import pandas as pd

data = {
    'Name': ['Ram', 'Shyam', 'Sita', 'Gita'],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Kolkata'],
    'Age': [23, 25, 22, 24]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

dummy = pd.get_dummies(df['City'])

print("\nDummy Variables:")
print(dummy)

final_data = pd.concat([df, dummy], axis=1)

print("\nData with Dummy Variables:")
print(final_data)

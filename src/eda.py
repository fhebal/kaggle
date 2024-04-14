import pandas as pd

df = pd.read_csv("data/features/train.csv")

print(df.info())
print(df.head().T)

print(df['Ticket'].sample(20))

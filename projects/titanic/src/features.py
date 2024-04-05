import pandas as pd


def CabinLetter(df):
    # Replacing numeric characters with an empty string to keep only non-digits
    df["CabinLetter"] = df["Cabin"].str.replace(r"\d", "", regex=True)
    return df


def CabinNumber(df):
    # Replacing non-digit characters with an empty string to keep only digits
    df["CabinNumber"] = df["Cabin"].str.replace(r"\D", "", regex=True)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    df = CabinLetter(df)
    df = CabinNumber(df)
    print(df.head())

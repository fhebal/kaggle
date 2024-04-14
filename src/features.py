import pandas as pd


def CabinLetter(df):
    # Replacing numeric characters with an empty string to keep only non-digits
    df["CabinLetter"] = df["Cabin"].str.replace(r"\d", "", regex=True)
    return df


def CabinNumber(df):
    # Replacing non-digit characters with an empty string to keep only digits
    df["CabinNumber"] = df["Cabin"].str.replace(r"\D", "", regex=True)
    return df


def TicketLetter(df):
    # Replacing numeric characters with an empty string to keep only non-digits
    df["TicketLetter"] = df["Ticket"].str.replace(r"\d", "", regex=True)
    return df


def TicketNumber(df):
    # Replacing non-digit characters with an empty string to keep only digits
    df["TicketNumber"] = df["Ticket"].str.replace(r"\D", "", regex=True)
    return df


def FamilySize(df):
    # Combining Number of Siblings/Spouses Number of Parents/Children.
    df["FamilySize"] = df["SibSp"].astype(int) + df["Parch"].astype(int)
    return df


def FamilySizePolyX(df):
    # Multiply Number of Siblings/Spouses and Number of Parents/Children.
    df["FamilySizePolyX"] = df["SibSp"].astype(int) * df["Parch"].astype(int)
    return df


def AgeMissing(df):
    # Replacing non-digit characters with an empty string to keep only digits
    df["AgeMissing"] = df["Age"].isna()
    return df


def CabinMissing(df):
    # Replacing non-digit characters with an empty string to keep only digits
    df["CabinMissing"] = df["Cabin"].isna()
    return df


def create_features(df):
    df = CabinLetter(df)
    df = CabinNumber(df)
    df = TicketNumber(df)
    df = TicketLetter(df)
    df = FamilySize(df)
    df = FamilySizePolyX(df)
    df = AgeMissing(df)
    df = CabinMissing(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    create_features(df)
    df.to_csv("data/features/train.csv", index=False)
    print(df.head())

    df = pd.read_csv("data/test.csv")
    create_features(df)
    df.to_csv("data/features/test.csv", index=False)

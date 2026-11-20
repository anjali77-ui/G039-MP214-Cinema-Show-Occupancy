
import pandas as pd

# Load datasets
train = pd.read_csv("data/raw/train.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")
later = pd.read_csv("data/raw/later.csv")

# Check columns
print("TRAIN COLUMNS:")
print(train.columns.tolist())

print("\nHOLDOUT COLUMNS:")
print(holdout.columns.tolist())

# Find columns present in train but missing in holdout
missing_columns = [
    col for col in train.columns
    if col not in holdout.columns
]

print("\nColumns missing from holdout:")
print(missing_columns)

# Check correlation with target
print("\nCorrelation with occupancy_pct:")
print(
    train.select_dtypes(include="number")
    .corr()["occupancy_pct"]
    .sort_values(ascending=False)
)
# Leakage detection
target = "occupancy_pct"
holdout_columns = set(holdout.columns)

leaking_columns = [
    col for col in train.columns
    if col not in holdout_columns and col != target
]

print("\nConfirmed leakage columns:")
for col in leaking_columns:
    print(col)

# Unit mixup detection

numeric_columns = [
    "ticket_price_rs",
    "days_since_release",
    "star_rating",
    "show_hour",
    "screen_seats",
    "distance_to_metro_km"
]

print("\nUNIT MIXUP CHECK:")

for col in numeric_columns:
    print(f"\nColumn: {col}")

    result = train.groupby("location")[col].agg(
        ["min", "mean", "max"]
    )

    print(result)
# Sentinel value detection

print("\nSENTINEL VALUE CHECK:")

for col in numeric_columns:
    print(f"\nColumn: {col}")
    print(train[col].value_counts().head(15))

# Duplicate rows detection

duplicate_count = train.duplicated().sum()

print("\nDUPLICATE ROW CHECK:")
print("Extra duplicate rows:", duplicate_count)
# Stuck sensor detection

print("\nSTUCK SENSOR CHECK:")

train["date"] = pd.to_datetime(train["date"])

numeric_columns = [
    "ticket_price_rs",
    "days_since_release",
    "star_rating",
    "show_hour",
    "screen_seats",
    "distance_to_metro_km"
]

for column in numeric_columns:
    print(f"\nColumn: {column}")

    temp = train.sort_values("date")
    temp["same_value"] = temp[column].eq(temp[column].shift())

    groups = temp["same_value"].ne(temp["same_value"].shift()).cumsum()

    result = temp.groupby(groups).agg(
        value=(column, "first"),
        count=(column, "size"),
        start_date=("date", "first"),
        end_date=("date", "last")
    )

    print(result[result["count"] >= 5].to_string(index=False))
print("\nSTUCK SENSOR VERIFICATION:")

for column in ["ticket_price_rs", "screen_seats", "distance_to_metro_km"]:
    print(f"\n{column}:")
    print(
        train.groupby(["location", column])
        .agg(
            count=("record_id", "count"),
            start_date=("date", "min"),
            end_date=("date", "max")
        )
        .sort_values("count", ascending=False)
        .head(10)
    )
# Target typos detection

print("\nTARGET TYPO CHECK:")

print("Target minimum:", train["occupancy_pct"].min())
print("Target maximum:", train["occupancy_pct"].max())

print("\nLargest target values:")
print(train["occupancy_pct"].nlargest(20).to_list())

print("\nTarget values above 100:")
print((train["occupancy_pct"] > 100).sum())
print("\nTARGET TYPO DETAILS:")

print(
    train.loc[
        train["occupancy_pct"] > 100,
        ["record_id", "occupancy_pct"]
    ].sort_values("occupancy_pct", ascending=False).to_string(index=False)
)

print("\nSPELLING VARIANTS CHECK:")

for column in ["language", "location"]:
    print(f"\nColumn: {column}")
    print(train[column].value_counts(dropna=False))

    print("\nClean later.csv values:")
    print(later[column].value_counts(dropna=False))
print("\nSTUCK SENSOR VERIFICATION:")
numeric_columns = [
    "ticket_price_rs",
    "days_since_release",
    "star_rating",
    "show_hour",
    "screen_seats",
    "distance_to_metro_km",
    "tickets_scanned_pct",
    "occupancy_pct" 
]

train["date"] = pd.to_datetime(train["date"])

for column in numeric_columns:
    print(f"\nColumn: {column}")

    for location, group in train.sort_values("date").groupby("location"):
        group = group.copy()

        group["run_id"] = group[column].ne(
            group[column].shift()
        ).cumsum()

        runs = group.groupby("run_id").agg(
            value=(column, "first"),
            count=(column, "size"),
            start_date=("date", "min"),
            end_date=("date", "max")
        )

        runs["days"] = (
            runs["end_date"] - runs["start_date"]
        ).dt.days + 1

        print(f"\nLocation: {location}")
        print(runs.sort_values("days", ascending=False).head(5))
print("\nDAILY STUCK SENSOR CHECK:")

for column in numeric_columns:
    print(f"\nColumn: {column}")

    for location, group in train.groupby("location"):
        daily = group.groupby("date")[column].agg(
            ["nunique", "first"]
        ).reset_index()

        daily = daily.sort_values("date")

        daily["run_id"] = (
            daily["first"].ne(daily["first"].shift())
            | daily["nunique"].ne(1)
        ).cumsum()

        runs = daily[daily["nunique"] == 1].groupby("run_id").agg(
            value=("first", "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            days=("date", "count")
        )

        print(f"\n{location}:")
        print(runs.sort_values("days", ascending=False).head(3))
print("\nDAILY UNIQUE VALUE COUNTS:")

for column in numeric_columns:
    print(f"\n{column}")

    daily = train.groupby("date")[column].nunique()

    print("Days with only one unique value:", (daily == 1).sum())
    print("Total dates:", len(daily))
    print("Longest consecutive single-value days:")

    mask = daily == 1
    groups = (~mask).cumsum()

    print(
        daily[mask]
        .groupby(groups[mask])
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
print("\nCONSECUTIVE SAME-VALUE DAILY CHECK:")

for column in numeric_columns:
    print(f"\nColumn: {column}")

    for location, group in train.groupby("location"):
        daily = (
            group.groupby("date")[column]
            .agg(nunique="nunique", value="first")
            .reset_index()
            .sort_values("date")
        )

        # Keep only dates where all records have the same value
        daily = daily[daily["nunique"] == 1].copy()

        same_value = daily["value"].eq(daily["value"].shift())
        consecutive_date = daily["date"].diff().dt.days.eq(1)

        daily["run_id"] = (
            ~(same_value & consecutive_date)
        ).cumsum()

        runs = daily.groupby("run_id").agg(
            value=("value", "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            days=("date", "count")
        )

        print(f"\n{location}:")
        print(runs.sort_values("days", ascending=False).head(3))
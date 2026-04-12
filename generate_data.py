import pandas as pd
import random
from datetime import datetime, timedelta

# -----------------------------
# AREAS (Ahmedabad)
# -----------------------------
areas = [
    "Chandkheda", "Gota", "Bopal", "Thaltej",
    "Naroda", "Shilaj", "Science City",
    "Ognaj", "Chiloda", "Gurukul"
]

# Assign crime weight (higher = more crime)
area_weights = {
    "Naroda": 5,
    "Chandkheda": 4,
    "Gota": 3,
    "Bopal": 3,
    "Thaltej": 2,
    "Shilaj": 2,
    "Science City": 2,
    "Ognaj": 2,
    "Chiloda": 4,
    "Gurukul": 3
}

# -----------------------------
# CRIME TYPES
# -----------------------------
crime_types = ["Theft", "Robbery", "Assault", "Burglary"]

# -----------------------------
# DATE RANGE
# -----------------------------
start_date = datetime(2023, 1, 1)

# -----------------------------
# GENERATE DATA
# -----------------------------
data = []

for i in range(180):  # 150+ rows

    # Choose area based on weight (realistic uneven distribution)
    area = random.choices(
        list(area_weights.keys()),
        weights=area_weights.values()
    )[0]

    # Date generation
    date = start_date + timedelta(days=random.randint(0, 60))

    # Introduce crime spikes (weekends more crime)
    if date.weekday() >= 5:  # Saturday/Sunday
        spike = random.choice([0, 1])
    else:
        spike = 0

    # Time pattern (night more crime)
    if spike == 1:
        hour = random.randint(20, 23)  # spike at night
    else:
        time_choice = random.choices(
            ["Morning", "Afternoon", "Night"],
            weights=[2, 3, 5]  # Night highest
        )[0]

        if time_choice == "Morning":
            hour = random.randint(6, 11)
        elif time_choice == "Afternoon":
            hour = random.randint(12, 18)
        else:
            hour = random.randint(19, 23)

    # Crime type pattern
    if hour >= 20:
        crime_type = random.choices(
            crime_types,
            weights=[5, 3, 2, 4]  # more theft & burglary at night
        )[0]
    else:
        crime_type = random.choice(crime_types)

    # Append row
    data.append([
        i + 1,
        area,
        crime_type,
        date.strftime("%Y-%m-%d"),
        hour
    ])

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame(data, columns=[
    "crime_id", "area", "crime_type", "date", "hour"
])

# -----------------------------
# SAVE FILE
# -----------------------------
df.to_csv("data/crime_data.csv", index=False)

print("✅ Realistic dataset generated successfully!")
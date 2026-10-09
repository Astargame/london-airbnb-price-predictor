"""Create app_metadata.json: borough locations, dropdown options and defaults for the app."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
clean = pd.read_csv(ROOT / "data" / "processed" / "listings_clean.csv")

boroughs = (
    clean.groupby("neighbourhood_cleansed")[["latitude", "longitude"]]
    .median()
    .round(5)
    .sort_index()
)

metadata = {
    "boroughs": {
        name: {"latitude": float(row["latitude"]), "longitude": float(row["longitude"])}
        for name, row in boroughs.iterrows()
    },
    "room_types": sorted(clean["room_type"].unique().tolist()),
    "defaults": {
        "amenities_count": int(clean["amenities_count"].median()),
        "minimum_nights": int(clean["minimum_nights"].median()),
        "review_scores_rating": float(clean["review_scores_rating"].median()),
    },
}

with open(ROOT / "models" / "app_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved metadata for", len(metadata["boroughs"]), "boroughs")
"""Streamlit app: predict the nightly price of a London Airbnb listing."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="London Airbnb Price Predictor", page_icon="🏠")

ROOT = Path(__file__).resolve().parent.parent
CENTRE_LAT, CENTRE_LON = 51.5080, -0.1281  # Trafalgar Square


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "price_model.joblib")


@st.cache_data
def load_metadata():
    with open(ROOT / "models" / "app_metadata.json") as f:
        return json.load(f)


def distance_to_centre_km(lat, lon):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat, lon, CENTRE_LAT, CENTRE_LON])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


model = load_model()
meta = load_metadata()
defaults = meta["defaults"]
borough_names = list(meta["boroughs"].keys())

# ---------- Page header ----------
st.title("🏠 London Airbnb Price Predictor")
st.write(
    "Estimate a nightly price for a London Airbnb listing. The prediction comes from a "
    "machine learning model trained on about 60,000 real London listings."
)

# ---------- Main inputs ----------
col1, col2 = st.columns(2)

with col1:
    borough = st.selectbox("Borough", borough_names, index=borough_names.index("Westminster"))
    room_type = st.selectbox(
        "Room type", meta["room_types"], index=meta["room_types"].index("Entire home/apt")
    )
    accommodates = st.slider("Number of guests", 1, 16, 2)
    bedrooms = st.number_input("Bedrooms (0 for a studio)", 0, 10, 1)

with col2:
    beds = st.number_input("Beds", 0, 16, 1)
    bathrooms = st.number_input("Bathrooms", 0.0, 8.0, 1.0, step=0.5)
    bath_shared = st.checkbox("Bathroom is shared with others")
    minimum_nights = st.number_input("Minimum nights", 1, 365, defaults["minimum_nights"])

# ---------- Optional inputs ----------
with st.expander("More details (optional)"):
    amenities_count = st.slider(
        "Number of amenities listed (e.g. Wi-Fi, kitchen, heating)", 0, 100, defaults["amenities_count"]
    )
    number_of_reviews = st.number_input("Number of reviews", 0, 2000, 0)
    rating = st.slider(
        "Average review rating", 1.0, 5.0, round(defaults["review_scores_rating"], 1), 0.1,
        disabled=number_of_reviews == 0,
        help="Only used if the listing has reviews.",
    )
    is_superhost = st.checkbox("Host is a Superhost")

# ---------- Build the listing and predict ----------
lat = meta["boroughs"][borough]["latitude"]
lon = meta["boroughs"][borough]["longitude"]

listing = pd.DataFrame([{
    "neighbourhood_cleansed": borough,
    "room_type": room_type,
    "accommodates": accommodates,
    "bedrooms": bedrooms,
    "beds": beds,
    "bathrooms": bathrooms,
    "bath_shared": int(bath_shared),
    "latitude": lat,
    "longitude": lon,
    "dist_centre_km": distance_to_centre_km(lat, lon),
    "minimum_nights": minimum_nights,
    "number_of_reviews": number_of_reviews,
    "has_reviews": int(number_of_reviews > 0),
    "review_scores_rating": rating if number_of_reviews > 0 else np.nan,
    "amenities_count": amenities_count,
    "is_superhost": int(is_superhost),
}])

price = float(np.exp(model.predict(listing)[0]))

st.divider()
st.metric("Estimated nightly price", f"£{price:,.0f}")
st.caption(
    f"Likely range: £{price * 0.8:,.0f} – £{price * 1.2:,.0f}. "
    "For a typical listing, the model's prediction is within about 20% of the real price."
)

# ---------- About ----------
with st.expander("About this model"):
    st.markdown(
        """
- **Data:** London listings from [Inside Airbnb](https://insideairbnb.com), limited to listings priced £20–£1,000 a night.
- **Model:** gradient boosting, chosen after comparing it with linear regression and a random forest.
- **Accuracy on unseen listings:** average error £57, typical error about 20%, R² 0.76.
- **Most important factors:** distance to central London, room type, number of guests and bedrooms.
- **Limitations:** location is approximated by the typical location of listings in the chosen borough. The model doesn't account for photos, property quality or seasonal price changes, and is less accurate for very cheap or very expensive listings.
"""
    )
    

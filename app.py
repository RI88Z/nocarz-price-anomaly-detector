import joblib
import json
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Nocarz - Offer Validator", page_icon="🏨", layout="centered"
)

st.title("🏨 Nocarz - Outlier Price Detector")
st.markdown("""
Real-time application verifying whether the entered nightly price is **appropriate** or an **outlier anomaly** (e.g., an attempt to artificially block the calendar without marking it as unavailable).
""")


@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load("generated/random_forest_production.joblib")
        artifacts = joblib.load("generated/artifacts.joblib")
        scaler = artifacts["scaler"]
        feature_names = artifacts["feature_names"]
        amenity_freq = artifacts["amenity_freq"]
        log_cols = artifacts["log_cols"]
        bin_cols = artifacts["bin_cols"]
        other_cols = artifacts["other_cols"]
        return (
            model,
            scaler,
            feature_names,
            amenity_freq,
            log_cols,
            bin_cols,
            other_cols,
        )
    except FileNotFoundError:
        st.error(
            "Model files not found! Make sure you saved the .joblib files from the Jupyter Notebook."
        )
        st.stop()


model, scaler, feature_names, amenity_freq, log_cols, bin_cols, other_cols = (
    load_ml_components()
)

st.header("Enter new offer details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Property details")
    price = st.number_input(
        "Price per night (USD) 💵", min_value=1.0, value=400.0, step=10.0
    )
    accommodates = st.number_input(
        "Accommodates 👥", min_value=1, max_value=20, value=2
    )
    bedrooms = st.number_input(
        "Bedrooms 🛏️", min_value=0.0, max_value=10.0, value=1.0, step=0.5
    )
    bathrooms = st.number_input(
        "Bathrooms 🚿", min_value=0.0, max_value=10.0, value=1.0, step=0.5
    )
    beds = st.number_input(
        "Beds 🛏️", min_value=1.0, max_value=20.0, value=1.0, step=1.0
    )

    room_type = st.selectbox(
        "Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    )

with col2:
    st.subheader("History and Amenities")
    latitude = st.slider("Latitude", 52.3, 52.7, 52.5, step=0.01)
    longitude = st.slider("Longitude", 13.0, 13.8, 13.4, step=0.01)
    # review_scores_rating = st.slider("Average rating (0-5) ⭐", 0.0, 5.0, 4.5, step=0.1)
    # number_of_reviews = st.number_input("Number of reviews 📝", min_value=0, value=10)

    st.markdown("**Amenities:**")

    amenities_dict = {
        "Kitchen": st.checkbox("Kitchen 🍳", value=False),
        "Wifi": st.checkbox("Wifi 📶", value=False),
        "Essentials": st.checkbox("Essentials 🏠", value=False),
        "Smoke alarm": st.checkbox("Smoke alarm 🚨", value=False),
        "Hair dryer": st.checkbox("Hair dryer 💨", value=False),
        "Hot water": st.checkbox("Hot water 🚿", value=False),
        "Hangers": st.checkbox("Hangers 👕", value=False),
        "Heating": st.checkbox("Heating ♨️", value=False),
        "Dishes and silverware": st.checkbox("Dishes and silverware 🍽️", value=False),
        "Cooking basics": st.checkbox("Cooking basics 🥘", value=False),
    }

    custom_amenities = st.text_input(
        "Additional amenities (comma separated)",
        placeholder="Wine glasses, Free parking on premises, Bed linens",
    )

st.markdown("**Host data not available in forms but included here to modify**")
response_rate = st.number_input(
    "host_response_rate", min_value=0.0, max_value=1.0, value=1.0, step=0.01
)
acceptance_rate = st.number_input(
    "host_acceptance_rate", min_value=0.0, max_value=1.0, value=1.0, step=0.01
)
listings_count = st.number_input(
    "host_total_listings_count", min_value=1, value=1, step=1
)

st.markdown("---")

if st.button("🔍 Check Offer", type="primary", use_container_width=True):

    def process_amenities(amenities, amenity_freq):

        amenities_count = len(amenities)

        premium_score = 0

        for a in amenities:
            freq = amenity_freq.get(a, 0.001)
            premium_score += 1 / freq

        premium_score = premium_score / amenities_count if amenities_count > 0 else 0

        return amenities_count, premium_score

    selected_amenities = [name for name, selected in amenities_dict.items() if selected]

    # własne amenities z pola tekstowego
    if custom_amenities.strip():
        selected_amenities.extend(
            [
                amenity.strip()
                for amenity in custom_amenities.split(",")
                if amenity.strip()
            ]
        )

    # format JSON string
    amenities = json.dumps(selected_amenities)
    print(f"{amenities=}")

    amenities_count, amenities_premium_score = process_amenities(
        selected_amenities, amenity_freq
    )

    # st.write("Amenities:")
    # st.code(amenities)

    input_dict = {}

    input_dict["price"] = price
    input_dict["accommodates"] = accommodates
    input_dict["bedrooms"] = bedrooms
    input_dict["bathrooms"] = bathrooms
    input_dict["beds"] = beds

    room_type_mapping = {
        "Entire home/apt": 0,
        "Private room": 1,
    }

    input_dict["room_type"] = room_type_mapping.get(room_type, 2)

    input_dict["latitude"] = latitude
    input_dict["longitude"] = longitude
    input_dict["amenities_count"] = amenities_count
    input_dict["amenities_premium_score"] = amenities_premium_score

    input_dict["host_response_rate"] = response_rate
    input_dict["host_acceptance_rate"] = acceptance_rate
    input_dict["host_total_listings_count"] = listings_count

    # 0. input
    input_df = pd.DataFrame([input_dict])

    # 1. FULL CANVAS (model space)
    full_df = pd.DataFrame(0, columns=feature_names, index=input_df.index)

    # 2. overwrite wartości z input
    common_cols = input_df.columns.intersection(full_df.columns)
    full_df[common_cols] = input_df[common_cols]

    # 3. log1p tylko na wybranych kolumnach
    full_df[log_cols] = np.log1p(full_df[log_cols])

    # 4. scaling (UWAGA: tylko scaler cols)
    scaler_cols = scaler.feature_names_in_
    full_df[scaler_cols] = scaler.transform(full_df[scaler_cols])

    # 5. FINAL INPUT DO MODELU (subset) + odpowiednia kolejnosc
    final_input = full_df[input_df.columns]
    final_input = final_input[full_df.columns.intersection(final_input.columns)]
    print("INPUT:")
    for col, val in final_input.iloc[0].items():
        print(f"{col}: {val}")

    prediction = model.predict(final_input)[0]
    print(f"OUTPUT: {prediction}\n\n")
    # anomaly_score = model.decision_function(X_scaled)[0]

    st.subheader("Analysis Result:")
    if prediction == 1:
        st.error("🚨 PRICE ANOMALY DETECTED (Outlier Price)!")
        st.write(
            f"The system has determined that the price of **${price}** is disproportionately high relative to the standard of this property."
        )
        st.info(
            "💡 **Recommendation for host:** Consider lowering the price or use the 'Block Dates' feature if the property is unavailable."
        )
    else:
        st.success("✅ PRICE WITHIN MARKET NORM")
        st.write(
            f"The price of **${price}** matches the standard of this property and its location in the system."
        )

    # with st.expander("View analytical parameters (For moderators)"):
    #     st.write(
    #         f"Raw model decision score (Anomaly Score): **{anomaly_score:.4f}** (Below 0 indicates an anomaly)"
    #     )
    #     st.write("Input feature vector:")
    #     st.json(input_dict)

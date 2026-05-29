import joblib
import pandas as pd
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
        model = joblib.load("nocarz_iso_model.joblib")
        scaler = joblib.load("nocarz_scaler.joblib")
        features = joblib.load("nocarz_features.joblib")
        return model, scaler, features
    except FileNotFoundError:
        st.error(
            "Model files not found! Make sure you saved the .joblib files from the Jupyter Notebook."
        )
        st.stop()


model, scaler, feature_names = load_ml_components()

st.header("Enter new offer details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Property details")
    price = st.number_input(
        "Price per night (USD) 💵", min_value=1.0, value=150.0, step=10.0
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
    beds = st.number_input("Beds 🛏️", min_value=1.0, max_value=20.0, value=1.0, step=1.0)

    room_type = st.selectbox(
        "Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    )

with col2:
    st.subheader("History and Amenities")
    review_scores_rating = st.slider("Average rating (0-5) ⭐", 0.0, 5.0, 4.5, step=0.1)
    number_of_reviews = st.number_input("Number of reviews 📝", min_value=0, value=10)

    st.markdown("**Amenities:**")
    has_wifi = st.checkbox("Wi-Fi 📶", value=True)
    has_kitchen = st.checkbox("Kitchen 🍳", value=True)
    has_parking = st.checkbox("Parking 🚗")
    has_ac = st.checkbox("Air Conditioning ❄️")
    has_elevator = st.checkbox("Elevator 🛗")
    has_pool = st.checkbox("Pool 🏊")
    has_hot_tub = st.checkbox("Hot Tub 🛁")

neighbourhood = st.text_input(
    "Neighbourhood name (e.g., Tempelhofer Vorstadt, Alexanderplatz)",
    value="Alexanderplatz",
)

st.markdown("---")

if st.button("🔍 Check Offer", type="primary", use_container_width=True):
    luxury_score = int(has_pool) + int(has_hot_tub) + int(has_ac) + int(has_elevator)

    input_dict = {col: 0 for col in feature_names}

    input_dict["price_clean"] = price
    input_dict["accommodates"] = accommodates
    input_dict["bedrooms"] = bedrooms
    input_dict["bathrooms"] = bathrooms
    input_dict["beds"] = beds
    input_dict["review_scores_rating"] = review_scores_rating
    input_dict["number_of_reviews"] = number_of_reviews

    input_dict["has_wifi"] = int(has_wifi)
    input_dict["has_kitchen"] = int(has_kitchen)
    input_dict["has_parking"] = int(has_parking)
    input_dict["has_ac"] = int(has_ac)
    input_dict["has_elevator"] = int(has_elevator)
    input_dict["has_pool"] = int(has_pool)
    input_dict["has_hot_tub"] = int(has_hot_tub)
    input_dict["luxury_score"] = luxury_score

    room_col = f"room_type_{room_type}"
    if room_col in input_dict:
        input_dict[room_col] = 1

    nh_col = f"nh_{neighbourhood}"
    if nh_col in input_dict:
        input_dict[nh_col] = 1

    input_df = pd.DataFrame([input_dict])

    input_df = input_df[feature_names]

    X_scaled = scaler.transform(input_df)

    prediction = model.predict(X_scaled)[0]
    anomaly_score = model.decision_function(X_scaled)[0]

    st.subheader("Analysis Result:")
    if prediction == -1:
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

    with st.expander("View analytical parameters (For moderators)"):
        st.write(
            f"Raw model decision score (Anomaly Score): **{anomaly_score:.4f}** (Below 0 indicates an anomaly)"
        )
        st.write("Input feature vector:")
        st.json(input_dict)

from flask import Flask, render_template, request
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

app = Flask(__name__)

data = pd.read_csv("new_data.csv")
model = pickle.load(open("land_price_model.pkl", "rb"))

DEFAULTS = {
    "Rental Value (INR/sqft/year)": 55.2,
    "Appreciation Rate (%)": 7.3,
    "Land Area (sqft)": 2400,
    "Distance to City Center (km)": 8.5,
    "Zone": "Residential",
    "Nearby Amenities Score": 7.0,
    "Road Connectivity Score": 8.2,
    "Guideline Value (INR/sqft)": 5200
}

localities = sorted(data["Locality"].dropna().unique())

@app.route('/')
def home():
    return render_template("index.html", localities=localities)

@app.route('/predict', methods=['POST'])
def predict():
    locality = request.form['locality']
    land_type = request.form['land_type']
    year = int(request.form['year'])

    input_data = {
        "Year": year,
        "Locality": locality,
        "Land Type": land_type,
        "Rental Value (INR/sqft/year)": DEFAULTS["Rental Value (INR/sqft/year)"],
        "Appreciation Rate (%)": DEFAULTS["Appreciation Rate (%)"],
        "Land Area (sqft)": DEFAULTS["Land Area (sqft)"],
        "Distance to City Center (km)": DEFAULTS["Distance to City Center (km)"],
        "Zone": DEFAULTS["Zone"],
        "Nearby Amenities Score": DEFAULTS["Nearby Amenities Score"],
        "Road Connectivity Score": DEFAULTS["Road Connectivity Score"],
        "Guideline Value (INR/sqft)": DEFAULTS["Guideline Value (INR/sqft)"]
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    locality_data = data[data["Locality"] == locality]
    if not locality_data.empty:
        grouped = locality_data.groupby("Year")["Price Per Sq Ft (INR)"].mean().reset_index()

        plt.figure(figsize=(6, 4))
        plt.plot(grouped["Year"], grouped["Price Per Sq Ft (INR)"], marker='o', color='green')
        plt.title(f"Price Trend in {locality}")
        plt.xlabel("Year")
        plt.ylabel("Price Per Sq Ft (INR)")
        plt.grid(True)
        plt.tight_layout()
        graph_path = os.path.join("static", "graph.png")
        plt.savefig(graph_path)
        plt.close()
    else:
        graph_path = None

    return render_template("index.html",
                           prediction_text=f"Predicted Price: ₹{prediction:.2f} per sq.ft",
                           localities=localities,
                           selected_locality=locality,
                           year=year,
                           land_type=land_type,
                           graph_url=graph_path if graph_path else None)

if __name__ == "__main__":
    app.run(debug=True)

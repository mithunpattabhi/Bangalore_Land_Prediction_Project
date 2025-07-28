### Bangalore Land Price Prediction Using Machine Learning

This project is a machine learning-based web application that predicts **land prices per square foot** in Bangalore based on multiple input features such as year, locality, land type, rental value, amenities, and more. The model is trained on real-world data and offers a user-friendly interface built using Flask.

---

## Features

- Predicts **price per square foot** based on:
  - Year
  - Locality
  - Land Type
  - Zone
  - Rental Value
  - Appreciation %
  - Distance (km)
  - Amenities Score
  - Connectivity Score
  - Guideline Value (INR)
- Year, Locality and Land Type infomration is collected from user.
- Other features are set by default
- Displays a trend plot showing **price appreciation by year**.
- Dynamic dropdowns and form validation.
- Integrated prediction logic using a trained **Linear Regression** model and **Pipeline**.

---

## Tech Stack

- **Frontend**: HTML, CSS (Bootstrap)
- **Backend**: Python, Flask
- **Model**: Scikit-learn (Linear Regression)
- **Visualization**: Matplotlib, Pandas

##Please Note:
- Project currently contains only 5 key localities
- More will be added as updating dataset

##  Requirements

Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt

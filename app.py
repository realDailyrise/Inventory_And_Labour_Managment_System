from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ===============================
# INVENTORY PREDICTION FUNCTION
# ===============================
def predict_inventory(csv_path):
    df = pd.read_csv(csv_path)

    # Expected CSV format:
    # Date,Stock_Used,Current_Stock

    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_Number'] = (df['Date'] - df['Date'].min()).dt.days

    X = df[['Day_Number']]
    y = df['Stock_Used']

    model = LinearRegression()
    model.fit(X, y)

    next_day = np.array([[df['Day_Number'].max() + 1]])
    predicted_usage = model.predict(next_day)[0]

    avg_daily_usage = df['Stock_Used'].mean()
    current_stock = df['Current_Stock'].iloc[-1]

    days_left = current_stock / avg_daily_usage

    return round(predicted_usage, 2), round(days_left, 1)


# ===============================
# LABOUR WAGE CALCULATION
# ===============================
def calculate_wages(csv_path):
    df = pd.read_csv(csv_path)

    # Expected CSV:
    # Name,Profession,Days_Worked,Daily_Wage

    df['Monthly_Wage'] = df['Days_Worked'] * df['Daily_Wage']
    total_labour_cost = df['Monthly_Wage'].sum()

    return df[['Name', 'Profession', 'Monthly_Wage']], total_labour_cost


# ===============================
# ROUTES
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    inventory_result = None
    labour_result = None
    total_cost = None

    if request.method == "POST":

        if "inventory_file" in request.files:
            inv_file = request.files["inventory_file"]
            inv_path = os.path.join(app.config["UPLOAD_FOLDER"], inv_file.filename)
            inv_file.save(inv_path)

            predicted_usage, days_left = predict_inventory(inv_path)
            inventory_result = {
                "predicted_usage": predicted_usage,
                "days_left": days_left
            }

        if "labour_file" in request.files:
            labour_file = request.files["labour_file"]
            labour_path = os.path.join(app.config["UPLOAD_FOLDER"], labour_file.filename)
            labour_file.save(labour_path)

            labour_df, total_cost = calculate_wages(labour_path)
            labour_result = labour_df.to_dict(orient="records")

    return render_template("index.html",
                           inventory_result=inventory_result,
                           labour_result=labour_result,
                           total_cost=total_cost)


if __name__ == "__main__":
    app.run(debug=True)

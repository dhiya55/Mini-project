from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import csv
import os
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

def load_external_dataset():
    try:
        data = []
        with open('plastic_data.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(float(row['plastic_score']))
        return data
    except Exception as e:
        print(f"Error loading external dataset: {e}")
        return [25, 18, 32, 15, 28, 12, 35, 20, 40, 22, 18, 30, 25, 15, 38, 20, 12, 35, 28, 16, 42, 24, 19, 33, 26, 14, 45, 30, 22, 40]

def linear_regression_prediction(user_scores):
    """
    Linear regression prediction for next month plastic scores (30 days ahead)
    """
    print(f"User scores for linear regression: {user_scores}")
    
    if len(user_scores) == 0:
        return 0, "stable", 0.5
    
    # For 1-2 data points, use simple average (not enough for regression)
    if len(user_scores) <= 2:
        avg = sum(user_scores) / len(user_scores)
        prediction = avg
        return round(prediction, 2), "stable", 0.3
    
    # Prepare data for linear regression
    # X: day numbers (1, 2, 3, ...)
    # y: plastic scores
    X = np.array(range(1, len(user_scores) + 1)).reshape(-1, 1)
    y = np.array(user_scores)
    
    # Create and train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next month (30 days ahead)
    next_month_day = len(user_scores) + 30
    prediction = model.predict([[next_month_day]])[0]
    
    # Ensure prediction is not negative
    prediction = max(0, prediction)
    
    # Calculate trend based on slope
    slope = model.coef_[0]
    if slope > 0.5:
        trend = "increasing"
    elif slope < -0.5:
        trend = "decreasing"
    else:
        trend = "stable"
    
    # Calculate confidence based on R-squared and data points
    r_squared = model.score(X, y)
    base_confidence = min(0.9, r_squared)
    
    # Adjust confidence based on data points
    if len(user_scores) >= 10:
        confidence = base_confidence
    elif len(user_scores) >= 5:
        confidence = base_confidence * 0.8
    else:
        confidence = base_confidence * 0.6
    
    print(f"Linear regression - Slope: {slope:.2f}, R-squared: {r_squared:.2f}")
    print(f"Prediction for next month (day {next_month_day}): {prediction:.2f}, Trend: {trend}, Confidence: {confidence:.2f}")
    
    return round(prediction, 2), trend, round(confidence, 2)

def simple_prediction(user_scores):
    """
    Simple and reliable prediction using only user data
    """
    print(f"User scores: {user_scores}")
    
    if len(user_scores) == 0:
        return 0, "stable", 0.5
    
    # For 1-2 data points, just use average
    if len(user_scores) <= 2:
        avg = sum(user_scores) / len(user_scores)
        prediction = avg
        return round(prediction, 2), "stable", 0.5
    
    # Use weighted average: recent data has more weight
    weights = []
    for i in range(len(user_scores)):
        # Recent days get higher weight (linear weighting)
        weight = (i + 1) / len(user_scores)
        weights.append(weight)
    
    # Calculate weighted average
    weighted_sum = sum(user_scores[i] * weights[i] for i in range(len(user_scores)))
    total_weight = sum(weights)
    weighted_avg = weighted_sum / total_weight
    
    # Simple trend detection
    if len(user_scores) >= 3:
        recent_avg = sum(user_scores[-3:]) / 3
        older_avg = sum(user_scores[:-3]) / len(user_scores[:-3]) if len(user_scores) > 3 else user_scores[0]
        
        if recent_avg > older_avg * 1.1:  # 10% increase
            trend = "increasing"
            # Add 10% to prediction for increasing trend
            prediction = weighted_avg * 1.1
        elif recent_avg < older_avg * 0.9:  # 10% decrease
            trend = "decreasing"
            # Subtract 10% from prediction for decreasing trend
            prediction = weighted_avg * 0.9
        else:
            trend = "stable"
            prediction = weighted_avg
    else:
        trend = "stable"
        prediction = weighted_avg
    
    # Simple confidence calculation
    if len(user_scores) >= 7:
        confidence = 0.8
    elif len(user_scores) >= 3:
        confidence = 0.6
    else:
        confidence = 0.4
    
    print(f"Weighted average: {weighted_avg:.2f}")
    print(f"Prediction: {prediction:.2f}, Trend: {trend}, Confidence: {confidence:.2f}")
    
    return round(prediction, 2), trend, round(confidence, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dataset-info')
def dataset_info():
    try:
        external_data = load_external_dataset()
        return jsonify({
            'dataset_size': len(external_data),
            'dataset_range': f"{min(external_data)} - {max(external_data)}",
            'dataset_avg': round(sum(external_data) / len(external_data), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        if isinstance(data, dict) and 'scores' in data:
            user_scores = data['scores']
        elif isinstance(data, list):
            user_scores = data
        else:
            return jsonify({'error': 'Invalid data format'}), 400
        
        print(f"=== PREDICTION REQUEST ===")
        print(f"User scores: {user_scores}")
        
        # Validate scores
        valid_scores = []
        for score in user_scores:
            if isinstance(score, (int, float)) and score >= 0:
                valid_scores.append(float(score))
        
        if len(valid_scores) == 0:
            return jsonify({'error': 'No valid scores provided'}), 400
        
        # Get prediction using LINEAR REGRESSION (main method) for next month
        prediction, trend, confidence = linear_regression_prediction(valid_scores)
        
        # Also get simple prediction for comparison
        simple_pred, simple_trend, simple_conf = simple_prediction(valid_scores)
        
        result = {
            'prediction': prediction,
            'trend': trend,
            'confidence': confidence,
            'simple_prediction': simple_pred,
            'simple_trend': simple_trend,
            'data_points': len(valid_scores),
            'message': f'Next month prediction based on {len(valid_scores)} days of data using Linear Regression'
        }
        
        print(f"Final result: {result}")
        print("=========================")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict-next-month', methods=['POST'])
def predict_next_month():
    """
    Separate endpoint specifically for next month prediction
    """
    try:
        data = request.get_json()
        
        if not data or 'scores' not in data:
            return jsonify({'error': 'No scores data received'}), 400
        
        user_scores = data['scores']
        
        # Validate scores
        valid_scores = []
        for score in user_scores:
            if isinstance(score, (int, float)) and score >= 0:
                valid_scores.append(float(score))
        
        if len(valid_scores) == 0:
            return jsonify({'error': 'No valid scores provided'}), 400
        
        # Get prediction using linear regression for next month
        prediction, trend, confidence = linear_regression_prediction(valid_scores)
        
        result = {
            'prediction': prediction,
            'trend': trend,
            'confidence': confidence,
            'data_points': len(valid_scores),
            'method': 'linear_regression',
            'prediction_type': 'next_month',
            'message': f'Next month prediction based on {len(valid_scores)} days of data'
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Next month prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_default_dataset():
    if not os.path.exists('plastic_data.csv'):
        default_data = [
            {'day': 1, 'plastic_score': 25},
            {'day': 2, 'plastic_score': 18},
            {'day': 3, 'plastic_score': 32},
            {'day': 4, 'plastic_score': 15},
            {'day': 5, 'plastic_score': 28},
            {'day': 6, 'plastic_score': 12},
            {'day': 7, 'plastic_score': 35},
            {'day': 8, 'plastic_score': 20},
            {'day': 9, 'plastic_score': 40},
            {'day': 10, 'plastic_score': 22},
            {'day': 11, 'plastic_score': 18},
            {'day': 12, 'plastic_score': 30},
            {'day': 13, 'plastic_score': 25},
            {'day': 14, 'plastic_score': 15},
            {'day': 15, 'plastic_score': 38},
            {'day': 16, 'plastic_score': 20},
            {'day': 17, 'plastic_score': 12},
            {'day': 18, 'plastic_score': 35},
            {'day': 19, 'plastic_score': 28},
            {'day': 20, 'plastic_score': 16},
            {'day': 21, 'plastic_score': 42},
            {'day': 22, 'plastic_score': 24},
            {'day': 23, 'plastic_score': 19},
            {'day': 24, 'plastic_score': 33},
            {'day': 25, 'plastic_score': 26},
            {'day': 26, 'plastic_score': 14},
            {'day': 27, 'plastic_score': 45},
            {'day': 28, 'plastic_score': 30},
            {'day': 29, 'plastic_score': 22},
            {'day': 30, 'plastic_score': 40}
        ]
        
        with open('plastic_data.csv', 'w', newline='') as file:
            fieldnames = ['day', 'plastic_score']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(default_data)
        
        print("Created default plastic_data.csv file")

if __name__ == '__main__':
    create_default_dataset()
    print("Starting Flask server on http://localhost:9999")
    print("Next month prediction enabled - projecting 30 days ahead")
    app.run(host='0.0.0.0', port=9999, debug=True)
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

def predict_plastic_score(user_scores):
    """
    Predict plastic score for next 30 days using linear regression
    """
    print(f"Predicting for scores: {user_scores}")
    
    if len(user_scores) == 0:
        return 0, "stable", 0.5
    
    # For small datasets, use weighted average
    if len(user_scores) <= 2:
        avg = sum(user_scores) / len(user_scores)
        prediction = avg
        return round(prediction, 2), "stable", 0.3
    
    # Prepare data for linear regression
    X = np.array(range(1, len(user_scores) + 1)).reshape(-1, 1)
    y = np.array(user_scores)
    
    # Create and train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict 30 days ahead
    next_30_days = len(user_scores) + 30
    prediction = model.predict([[next_30_days]])[0]
    
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
    
    # SIMPLE STABLE CONFIDENCE CALCULATION
    r_squared = model.score(X, y)
    
    # Calculate data consistency
    avg_score = np.mean(y)
    if avg_score > 0:
        std_dev = np.std(y)
        consistency = max(0.1, 1 - (std_dev / avg_score))
    else:
        consistency = 0.5
    
    # Use the better of R-squared or consistency
    base_confidence = max(r_squared, consistency)
    
    # Adjust for data quantity - SIMPLIFIED
    if len(user_scores) >= 7:
        confidence = base_confidence * 0.8
    elif len(user_scores) >= 4:
        confidence = base_confidence * 0.6
    else:
        confidence = base_confidence * 0.4
    
    # Ensure reasonable range
    confidence = min(0.95, max(0.1, confidence))
    
    print(f"Confidence - R²: {r_squared:.3f}, Consistency: {consistency:.3f}, Final: {confidence:.3f}")
    
    return round(prediction, 2), trend, round(confidence, 2)
@app.route('/')



def calculate_precision(user_scores):
    """
    Calculate precision by testing 1-day ahead predictions (not 30 days)
    """
    if len(user_scores) < 4:
        return 0
    
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(3, len(user_scores)):
        training_data = user_scores[:i]  # Use first i days
        actual_score = user_scores[i]    # Actual score on day i+1
        
        # Prepare data for 1-day ahead prediction
        X = np.array(range(1, len(training_data) + 1)).reshape(-1, 1)
        y = np.array(training_data)
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict ONLY 1 day ahead (not 30 days)
        next_day = len(training_data) + 1
        prediction = model.predict([[next_day]])[0]
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Calculate error
        if actual_score > 0:
            error_percentage = abs(prediction - actual_score) / actual_score
        else:
            error_percentage = abs(prediction - actual_score)
        
        # Consider correct if within 30% error
        if error_percentage <= 0.3:
            correct_predictions += 1
        
        total_predictions += 1
        
        print(f"Precision test: Day {next_day} - Predicted: {prediction:.1f}, Actual: {actual_score}, Error: {error_percentage:.1%}")
    
    precision = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"Precision result: {correct_predictions}/{total_predictions} = {precision:.3f}")
    
    return precision
def calculate_recall(user_scores):
    """
    Simple recall calculation using 1-day ahead predictions
    """
    if len(user_scores) < 4:
        return 0
    
    correct_detections = 0
    total_changes = 0
    
    print("=== RECALL CALCULATION START ===")
    
    for i in range(3, len(user_scores)):
        training_data = user_scores[:i]
        actual_next = user_scores[i]
        actual_prev = user_scores[i-1]
        
        # SIMPLE 1-DAY PREDICTION (not 30 days!)
        X = [[x] for x in range(1, len(training_data) + 1)]
        y = training_data
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next day (not 30 days!)
        next_day = len(training_data) + 1
        next_day_pred = model.predict([[next_day]])[0]
        
        # Get 1-day trend from slope
        slope = model.coef_[0]
        if slope > 0.1:  # Lower threshold for 1-day trends
            pred_trend = "increasing"
        elif slope < -0.1:
            pred_trend = "decreasing"
        else:
            pred_trend = "stable"
        
        # Determine actual trend
        abs_diff = abs(actual_next - actual_prev)
        pct_diff = abs(actual_next - actual_prev) / actual_prev
        
        if abs_diff >= 2 or pct_diff >= 0.08:  # Lower thresholds
            actual_trend = "increasing" if actual_next > actual_prev else "decreasing"
        else:
            actual_trend = "stable"
        
        # Count recall
        if actual_trend != "stable":
            total_changes += 1
            if pred_trend == actual_trend:
                correct_detections += 1
        
        print(f"Day {i+1}: Actual {actual_prev}→{actual_next} ({actual_trend}), Predicted: {pred_trend} (slope: {slope:.3f})")
    
    recall = correct_detections / total_changes if total_changes > 0 else 1.0
    print(f"=== RECALL RESULT: {correct_detections}/{total_changes} = {recall:.3f} ===")
    
    return recall


def calculate_f1_score(user_scores):
    """
    Calculate F1-Score - harmonic mean of precision and recall
    """
    if len(user_scores) < 4:
        return 0
    
    # Calculate precision and recall first
    precision = calculate_precision(user_scores)
    recall = calculate_recall(user_scores)
    
    # F1-Score formula: 2 * (precision * recall) / (precision + recall)
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    print(f"F1-Score calculation: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}")
    
    return f1_score
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
        
        # Get prediction for next 30 days
        prediction, trend, confidence = predict_plastic_score(valid_scores)
        
        result = {
            'prediction': prediction,
            'trend': trend,
            'confidence': confidence,
            'data_points': len(valid_scores),
            'message': f'30-day prediction based on {len(valid_scores)} days of data'
        }
        
        print(f"Final result: {result}")
        print("=========================")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug-predict', methods=['POST'])
def debug_predict():
    try:
        data = request.get_json()
        user_scores = data.get('scores', [])
        
        print(f"=== DEBUG PREDICTION ===")
        print(f"Scores: {user_scores}")
        
        if len(user_scores) < 2:
            return jsonify({'error': 'Need at least 2 scores'})
        
        # Prepare data
        X = np.array(range(1, len(user_scores) + 1)).reshape(-1, 1)
        y = np.array(user_scores)
        
        print(f"X (days): {X.flatten()}")
        print(f"y (scores): {y}")
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get model details
        r_squared = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Predict
        next_30_days = len(user_scores) + 30
        prediction = model.predict([[next_30_days]])[0]
        
        print(f"Model: y = {intercept:.2f} + {slope:.2f} * X")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Prediction for day {next_30_days}: {prediction:.2f}")
        print("========================")
        
        return jsonify({
            'r_squared': round(r_squared, 4),
            'slope': round(slope, 4),
            'intercept': round(intercept, 4),
            'prediction': round(prediction, 2),
            'next_day': next_30_days,
            'equation': f"y = {intercept:.2f} + {slope:.2f} * day"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/precision', methods=['POST'])
def calculate_precision_endpoint():
    try:
        data = request.get_json()
        user_scores = data.get('scores', [])
        
        if len(user_scores) < 4:
            return jsonify({
                'error': 'Need at least 4 data points for precision calculation',
                'minimum_data_points': 4,
                'current_data_points': len(user_scores)
            })
        
        precision = calculate_precision(user_scores)
        
        return jsonify({
            'precision': round(precision, 3),
            'precision_percentage': round(precision * 100, 1),
            'data_points_used': len(user_scores),
            'message': f'Model precision based on {len(user_scores)} days of historical data'
        })
        
    except Exception as e:
        print(f"Precision calculation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/recall', methods=['POST'])
def calculate_recall_endpoint():
    try:
        data = request.get_json()
        user_scores = data.get('scores', [])
        
        if len(user_scores) < 4:
            return jsonify({
                'error': 'Need at least 4 data points for recall calculation',
                'minimum_data_points': 4,
                'current_data_points': len(user_scores)
            })
        
        recall = calculate_recall(user_scores)
        
        return jsonify({
            'recall': round(recall, 3),
            'recall_percentage': round(recall * 100, 1),
            'data_points_used': len(user_scores),
            'message': f'Trend detection recall based on {len(user_scores)} days of data'
        })
        
    except Exception as e:
        print(f"Recall calculation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/f1-score', methods=['POST'])
def calculate_f1_score_endpoint():
    try:
        data = request.get_json()
        user_scores = data.get('scores', [])
        
        if len(user_scores) < 4:
            return jsonify({
                'error': 'Need at least 4 data points for F1-Score calculation',
                'minimum_data_points': 4,
                'current_data_points': len(user_scores)
            })
        
        f1_score = calculate_f1_score(user_scores)
        
        return jsonify({
            'f1_score': round(f1_score, 3),
            'f1_score_percentage': round(f1_score * 100, 1),
            'precision': round(calculate_precision(user_scores), 3),
            'recall': round(calculate_recall(user_scores), 3),
            'data_points_used': len(user_scores),
            'message': f'F1-Score based on {len(user_scores)} days of data'
        })
        
    except Exception as e:
        print(f"F1-Score calculation error: {str(e)}")
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
    print("30-day plastic score prediction enabled")
    print("Debug endpoint available at /debug-predict")
    app.run(host='0.0.0.0', port=9999, debug=True)
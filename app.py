from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import csv
import os

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

def advanced_prediction(history):
    if len(history) < 2:
        avg = sum(history) / len(history) if history else 0
        return avg * 30, "stable", 0.5
    
    # Use only the most recent data for better predictions
    recent_history = history[-min(14, len(history)):]  # Use up to last 14 days
    
    x = list(range(len(recent_history)))
    y = recent_history
    
    # Manual linear regression
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    
    if denominator == 0:
        predicted_avg = mean_y
        trend = "stable"
    else:
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        
        # Predict the average for next 30 days based on recent trend
        predicted_avg = slope * (n + 15) + intercept
        predicted_avg = max(5, predicted_avg)  # Ensure at least 5 points per day
        
        # Calculate trend
        if slope > 0.1:
            trend = "increasing"
        elif slope < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"
    
    # Calculate 30-day prediction
    total_prediction = predicted_avg * 30
    
    # Calculate confidence based on data quality
    confidence = min(0.9, max(0.3, 0.5 + (1 - (np.std(y) / mean_y if mean_y > 0 else 0.5))))
    
    return round(total_prediction, 2), trend, round(confidence, 2)

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
            'dataset_avg': round(sum(external_data) / len(external_data), 2),
            'dataset_min': min(external_data),
            'dataset_max': max(external_data)
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
            scores = data['scores']
        elif isinstance(data, list):
            scores = data
        else:
            return jsonify({'error': 'Invalid data format'}), 400
        
        # Validate scores
        valid_scores = [score for score in scores if isinstance(score, (int, float)) and score >= 0]
        
        if len(valid_scores) != len(scores):
            return jsonify({'error': 'All scores must be non-negative numbers'}), 400
        
        external_data = load_external_dataset()
        all_data = valid_scores + external_data
        
        prediction, trend, confidence = advanced_prediction(all_data)
        
        # Calculate simple average correctly
        simple_avg = round((sum(all_data) / len(all_data)) * 30, 2) if all_data else 0
        
        result = {
            'prediction': prediction,
            'trend': trend,
            'confidence': confidence,
            'simple_average': simple_avg,
            'data_points': len(all_data),
            'message': f'Based on {len(all_data)} days of data (your {len(valid_scores)} + our {len(external_data)})'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Create plastic_data.csv if it doesn't exist
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
    print("Make sure to install required packages: pip install flask flask-cors numpy")
    app.run(host='0.0.0.0', port=9999, debug=True)

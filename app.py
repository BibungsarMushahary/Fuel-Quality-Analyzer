from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import  math
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

def predict_fuel_degradation(fuel_type, parameters, storage_conditions):
    """
    Predicts when fuel will degrade based on properties and storage conditions
    Returns: (days_until_degradation, degradation_factors, storage_grade)
    """
    # Base stability under ideal conditions (days)
    BASE_STABILITY = {
        'Petrol': 90,    # 3 months
        'Diesel': 180,   # 6 months
        'Kerosene': 270  # 9 months
    }
    
    degradation_factors = []
    degradation_rate = 1.0
    storage_score = 100  # Perfect score
    
    # 1. Fuel Property Factors
    # Water content (exponential impact)
    water_impact = math.exp(parameters['water_content'] / 50 - 1)
    if parameters['water_content'] > 30:
        degradation_factors.append(f"High water content ({parameters['water_content']}ppm)")
    degradation_rate *= water_impact
    storage_score -= min(30, parameters['water_content'] * 0.5)
    
    # Sulphur content
    sulphur_impact = 1 + (parameters['sulphur_content'] / 100)
    if parameters['sulphur_content'] > 15:
        degradation_factors.append(f"Elevated sulphur ({parameters['sulphur_content']}ppm)")
    degradation_rate *= sulphur_impact
    
    # Viscosity impact (non-linear relationship)
    if fuel_type == 'Diesel':
        viscosity_impact = 1 + abs(parameters['viscosity'] - 3.0) * 0.1
    else:  # Petrol/Kerosene
        viscosity_impact = 1 + abs(parameters['viscosity'] - 0.7) * 0.2
    degradation_rate *= viscosity_impact
    
    # Flash point impact
    if parameters['flash_point'] < (55 if fuel_type == 'Diesel' else -20):
        flash_impact = 1.5
        degradation_factors.append(f"Low flash point ({parameters['flash_point']}°C)")
    else:
        flash_impact = 1.0
    degradation_rate *= flash_impact
    
    # Combustion rating
    if fuel_type == 'Petrol':
        octane_impact = 1 + (95 - parameters['octane_content']) * 0.02
        if parameters['octane_content'] < 95:
            degradation_factors.append(f"Lower octane ({parameters['octane_content']}RON)")
    elif fuel_type == 'Diesel':
        octane_impact = 1 + (50 - parameters.get('cetane_content', 50)) * 0.01
    degradation_rate *= octane_impact
    
    # 2. Storage Condition Factors
    # Temperature (most critical)
    temp = storage_conditions.get('temperature', 25)
    if temp > 30:
        temp_impact = math.exp((temp - 30) * 0.03)
        degradation_factors.append(f"High storage temp ({temp}°C)")
        storage_score -= (temp - 30) * 1.5
    elif temp < 10:
        temp_impact = 1 + (10 - temp) * 0.02
        degradation_factors.append(f"Low storage temp ({temp}°C)")
    else:
        temp_impact = 1.0
    degradation_rate *= temp_impact
    
    # Container type
    container_impact = {
        'metal': 1.0,
        'plastic': 1.3,
        'fiberglass': 1.2,
        'underground': 0.9
    }.get(storage_conditions.get('container_type', 'metal'), 1.0)
    degradation_rate *= container_impact
    
    # Headspace (oxidation) 
    headspace = storage_conditions.get('headspace', 10)
    headspace_impact = 1.0  # Default value
    if headspace > 20:
        headspace_impact = 1 + (headspace - 20) * 0.01
        degradation_factors.append(f"Large air space ({headspace}%)")
    degradation_rate *= headspace_impact
    
    # Calculate final degradation time
    base_stability = BASE_STABILITY.get(fuel_type, 90)
    days_until_degradation = base_stability / degradation_rate
    
    # Determine storage grade
    storage_grade = (
        "A (Excellent)" if storage_score >= 90 else
        "B (Good)" if storage_score >= 75 else
        "C (Fair)" if storage_score >= 60 else
        "D (Poor)" if storage_score >= 40 else
        "F (Unacceptable)"
    )
    
    return int(days_until_degradation), degradation_factors, storage_grade

class FuelQualityAnalyzer:
    def __init__(self):
        model_path = 'models/fuel_quality_model.pkl'
        scaler_path = 'models/fuel_quality_scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            print("Model not found. Training new model...")
            df = generate_sample_data()
            os.makedirs('models', exist_ok=True)
            self.model, self.scaler = train_fuel_quality_model(df)
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
    
    def predict_quality(self, input_data):
        input_df = pd.DataFrame([input_data])
        scaled_input = self.scaler.transform(input_df)
        prediction = self.model.predict(scaled_input)
        proba = self.model.predict_proba(scaled_input)
        
        return {
            'prediction': 'PASS' if prediction[0] == 1 else 'FAIL',
            'confidence': f"{max(proba[0])*100:.1f}%",
            'parameters': input_data
        }

def generate_sample_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'density': np.random.normal(820, 20, num_samples),
        'viscosity': np.random.normal(2.5, 0.5, num_samples),
        'sulphur_content': np.random.lognormal(1.2, 0.3, num_samples),
        'flash_point': np.random.normal(65, 10, num_samples),
        'octane_content': np.random.normal(92, 5, num_samples),
        'water_content': np.random.exponential(20, num_samples),
    }
    df = pd.DataFrame(data)
    conditions = (
        (df['density'].between(770, 850)) &
        (df['viscosity'].between(1.5, 4.0)) &
        (df['sulphur_content'] < 50) &
        (df['flash_point'] > 55) &
        (df['octane_content'] > 87) &
        (df['water_content'] < 100)
    )
    df['quality_pass'] = np.where(conditions, 1, 0)
    return df

def train_fuel_quality_model(df):
    X = df.drop('quality_pass', axis=1)
    y = df['quality_pass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    analyzer = FuelQualityAnalyzer()
    fuel_type = request.form.get('fuel_type')
    
    input_data = {
        'density': float(request.form['density']),
        'viscosity': float(request.form['viscosity']),
        'sulphur_content': float(request.form['sulphur_content']),
        'flash_point': float(request.form['flash_point']),
        'octane_content': float(request.form['octane_content']),
        'water_content': float(request.form['water_content'])
    }
    
    result = analyzer.predict_quality(input_data)
    
    # Additional quality assessment
    quality_status = "Standard"
    issues = []
    recommendations = []
    warnings = []
    
    # Octane number check
    if input_data['octane_content'] < 87:
        issues.append(f"Low octane rating ({input_data['octane_content']} RON)")
        recommendations.append("Add MTBE-based octane booster to improve octane number")
    
    # Water content check
    if input_data['water_content'] > 50:
        issues.append(f"Elevated water content ({input_data['water_content']} ppm)")
        recommendations.append("Use a water separator or coalescing filter to remove water contamination")
    
    # Sulphur content check
    if input_data['sulphur_content'] > 50:
        issues.append(f"High sulphur content ({input_data['sulphur_content']} ppm)")
        recommendations.append("Consider desulphurization treatment or use sulphur-removing additives")
        warnings.append("High sulphur can damage emission control systems and increase pollution")
    
    # Flash point check
    if input_data['flash_point'] < 55:
        issues.append(f"Low flash point ({input_data['flash_point']} °C) - safety concern")
        recommendations.append("Check for contamination with lighter fractions. Consider blending with higher flash point fuel")
        warnings.append("Low flash point increases fire hazard during storage and handling")
    elif input_data['flash_point'] > 100:
        issues.append(f"High flash point ({input_data['flash_point']} °C) - may affect combustion")
        recommendations.append("Verify fuel composition. May need blending with lower flash point components")
    
    # Viscosity check
    if input_data['viscosity'] < 1.5:
        issues.append(f"Low viscosity ({input_data['viscosity']} cSt) - may cause pump wear")
        recommendations.append("Add viscosity improver or blend with higher viscosity fuel")
    elif input_data['viscosity'] > 4.0:
        issues.append(f"High viscosity ({input_data['viscosity']} cSt) - may affect atomization")
        recommendations.append("Consider preheating or blending with lower viscosity fuel")
    
    # Density check
    if not 770 <= input_data['density'] <= 850:
        issues.append(f"Density out of range ({input_data['density']} kg/m³)")
        recommendations.append("Verify fuel composition and consider blending adjustment")
    
    storage_conditions = {
        'temperature': float(request.form.get('storage_temp', 25)),
        'container_type': request.form.get('container_type', 'metal'),
        'headspace': float(request.form.get('headspace', 10))
    }  
    
    #initialize degradation var  
    days_until_degradation = None
    degradation_factors = []
    storage_grade = ""
       
    if result['prediction'] == 'PASS':
        days_until_degradation, degradation_factors, storage_grade = (
            predict_fuel_degradation(fuel_type, input_data, storage_conditions)
        )
    
    if issues:
        quality_status = "Adulterated"
    
    return render_template('result.html', 
                         result=result,
                         fuel_type=fuel_type,
                         quality_status=quality_status,
                         issues=issues,
                         recommendations=recommendations,
                         warnings=warnings,
                         parameters=input_data,
                         days_until_degradation=days_until_degradation,
                         degradation_factors=degradation_factors,
                         storage_grade=storage_grade)
    
if __name__ == '__main__':
    app.run(debug=True)
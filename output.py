from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def result():
    # Example sample data
    sample_data = {
        'fuel_type': 'Petrol',
        'octane_number': 83,
        'density': '0.721 g/cm³',
        'water_content': '0.06%',
        'status': {
            'fuel_type': 'OK',
            'octane_number': 'Too Low',
            'density': 'Borderline',
            'water_content': 'High'
        }
    }
    return render_template('output.html', data=sample_data)

if __name__ == '__main__':
    app.run(debug=True)

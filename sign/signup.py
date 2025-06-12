from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def signup():
    return render_template('signup.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm-password']

    if password != confirm_password:
        return "Passwords do not match!"
    
    # You can store these details into a file or database.
    return f"Signup successful! Welcome, {name}"

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, flash, session

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for flash and sessions

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == 'admin@example.com' and password == 'password123':
            session['user'] = email  # Store user session (optional)
            return redirect(url_for('test_fuel'))  # Go to test fuel page
        else:
            flash("Invalid credentials. Try again.")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/templates')
def test_fuel():
    # Optional: redirect if user not logged in
    if 'user' not in session:
        flash("Please login first.")
        return redirect(url_for('login'))
    return render_template('test_fuel.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

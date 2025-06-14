import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_result_email(recipient_email, username, test_data):
    sender_email = "your-email@gmail.com"
    sender_password = "your-app-password"  # Use Gmail App Password

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = "Your Fuel Quality Test Results"

    body = f"""
    <h2>Hello {username}!</h2>
    <p>Here are your fuel quality test results:</p>

    <table border="1">
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Fuel Type</td><td>{test_data['fuel_type']}</td></tr>
        <tr><td>Result</td><td>{test_data['result']}</td></tr>
        <tr><td>Confidence</td><td>{test_data['confidence']}</td></tr>
        <tr><td>Water Content</td><td>{test_data['water_content']} ppm</td></tr>
    </table>

    <p>Thank you for using our service!</p>
    """

    message.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Configuration for GMAIL (MUST USE A GMAIL APP PASSWORD) ---
# NOTE: Replace these placeholders with your actual, secured credentials.
SMTP_SERVER = "smtp.gmail.com"  
SMTP_PORT = 587                  
SENDER_EMAIL = "sanchrao22@gmail.com"  # Your sending Gmail address
SMTP_PASSWORD = "hvfk bwrh mpyl slrc"    # The Google App Password
# -------------------------------------------------------------------------

def send_alert_email(recipient_email, subject, body_html):
    """
    Sends an HTML-formatted email alert via Gmail's SMTP server.
    
    Args:
        recipient_email (str): The email address to send the alert to.
        subject (str): The subject line of the email.
        body_html (str): The HTML content of the email body.
    
    Returns:
        bool: True if email was successfully sent, False otherwise.
    """
    try:
        msg = MIMEMultipart("alternative")
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        
        # Attach HTML content
        part1 = MIMEText(body_html, 'html')
        msg.attach(part1)

        # Connect to the SMTP server
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            # Encrypt the connection using TLS
            server.starttls() 
            server.login(SENDER_EMAIL, SMTP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        
        print(f"Email sent successfully to {recipient_email}")
        return True

    except smtplib.SMTPAuthenticationError:
        # Authentication error usually means the password is wrong or 2FA is blocking it.
        print("ERROR: SMTP Authentication Failed. Check Gmail 'App Password' and SENDER_EMAIL.")
        return False
    except Exception as e:
        print(f"ERROR: Could not send email: {e}")
        return False

from google.colab import drive

# Mount your Google Drive
drive.mount('/content/drive')

import imaplib
import email
import os
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import smtplib
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Your email and app password
EMAIL = 'anikkw69@gmail.com'
APP_PASSWORD = 'ffpu vwff hftx tnsd'

# Alert recipient email
ALERT_EMAIL = 'anikets5016@gmail.com'
ALERT_PASSWORD = 'wnsd ibqx boue mwyq'

# Path to save images
IMAGE_SAVE_DIR = '/content/images/'

# Path to your model in Google Drive
MODEL_PATH = '/content/drive/MyDrive/tiger_leopard_human_detection_model.h5'

# Ensure the image directory exists
if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

def connect_to_gmail():
    """Connects to Gmail using IMAP."""
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(EMAIL, APP_PASSWORD)
        return mail
    except imaplib.IMAP4.error as e:
        logging.error(f"Failed to connect to Gmail: {e}")
        raise

def fetch_emails_with_images(mail):
    """Fetches emails with image attachments and saves images."""
    mail.select('inbox')
    status, data = mail.search(None, 'ALL')
    email_ids = data[0].split()

    images = []
    email_details = {}

    for email_id in email_ids:
        try:
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            msg = email.message_from_bytes(msg_data[0][1])

            subject = decode_header(msg.get('Subject'))[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()

            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type.startswith('image/'):
                    file_name = part.get_filename()
                    if file_name:
                        file_path = os.path.join(IMAGE_SAVE_DIR, file_name)
                        with open(file_path, 'wb') as f:
                            f.write(part.get_payload(decode=True))
                        images.append(file_path)
                        if email_id not in email_details:
                            email_details[email_id] = {'subject': subject, 'images': []}
                        email_details[email_id]['images'].append(file_path)
        except Exception as e:
            logging.error(f"Failed to fetch or process email ID {email_id}: {e}")

    return images, email_details

def sharpen_image(image_array):
    """Apply sharpening to an image."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image_array, -1, kernel)

def load_and_predict_images(model, image_paths):
    """Load images, preprocess, and predict using the model."""
    predictions = {}
    for img_path in image_paths:
        try:
            # Load and preprocess image
            img = image.load_img(img_path, target_size=(150, 150))  # Adjust size to match training size
            img_array = image.img_to_array(img)
            img_array = sharpen_image(img_array)  # Apply sharpening
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize image if needed

            # Predict
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction, axis=1)[0]
            predictions[img_path] = class_index
        except Exception as e:
            logging.error(f"Failed to process image {img_path}: {e}")

    return predictions

def send_alert_email(subject, tiger_images, leopard_images, human_images):
    """Send an alert email with images attached."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = ALERT_EMAIL
        msg['Subject'] = 'Animal Detection Alert'

        body = "Alert! The following have been detected:\n"

        if tiger_images:
            body += "\nTiger detected in these images:\n"
            body += "\n".join(tiger_images)

        if leopard_images:
            body += "\n\nLeopard detected in these images:\n"
            body += "\n".join(leopard_images)

        if human_images:
            body += "\n\nHuman detected in these images:\n"
            body += "\n".join(human_images)

        msg.attach(MIMEText(body, 'plain'))

        all_images = tiger_images + leopard_images + human_images
        for image_path in all_images:
            part = MIMEBase('application', 'octet-stream')
            with open(image_path, 'rb') as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(part)

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(ALERT_EMAIL, ALERT_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        logging.error(f"Failed to send alert email: {e}")

def main():
    """Main function to execute the email processing and image classification."""
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    try:
        mail = connect_to_gmail()
        image_paths, email_details = fetch_emails_with_images(mail)
        mail.logout()
    except Exception as e:
        logging.error(f"Error in fetching emails: {e}")
        return

    if image_paths:
        logging.info("Images found and saved:")
        for img_path in image_paths:
            logging.info(img_path)

        predictions = load_and_predict_images(model, image_paths)

        tiger_images = [img_path for img_path, class_index in predictions.items() if class_index == 0]  # Class 0 for Tiger
        leopard_images = [img_path for img_path, class_index in predictions.items() if class_index == 1]  # Class 1 for Leopard
        human_images = [img_path for img_path, class_index in predictions.items() if class_index == 2]  # Class 2 for Human

        if tiger_images or leopard_images or human_images:
            logging.info("Sending alert email...")
            for email_id, details in email_details.items():
                send_alert_email(details['subject'], tiger_images, leopard_images, human_images)
                logging.info(f"Alert sent for email ID: {email_id}")
        else:
            logging.info("No tigers, leopards, or humans detected in any images.")
    else:
        logging.info("No images found in emails.")

if __name__ == '__main__':
    main()

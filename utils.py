

import cv2
import numpy as np
import re
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional

def preprocess_image(path: str) -> np.ndarray:
    """Preprocesses the image for OCR."""
    carplate = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(carplate, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    resized_image = cv2.GaussianBlur(resized_image, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_image = clahe.apply(resized_image)
    thresh = cv2.adaptiveThreshold(cl_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 151, 2)
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.medianBlur(thresh, 3)
    return thresh

def find_contours(image: np.ndarray) -> Optional[np.ndarray]:
    """Find contours of the car plate on the image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plate_contour = None
    max_area = 0
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                plate_contour = approx
                max_area = area
    return plate_contour

def transform_carplate(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Perspective transformation of a carplate."""
    if contour is None:
        return image
    
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    width = int(max(np.linalg.norm(rect[0] - rect[1]),
                   np.linalg.norm(rect[2] - rect[3])))
    height = int(max(np.linalg.norm(rect[0] - rect[3]),
                    np.linalg.norm(rect[1] - rect[2])))
    
    dst = np.array([[0, 0],
                   [width - 1, 0],
                   [width - 1, height - 1],
                   [0, height - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def extract_fine_amount(text: str) -> Optional[int]:
    """Extract numeric value from fine amount text."""
    pattern = r'штрафа в размере (\d+(?:[\xa0 ]\d{3})*) руб'
    match = re.search(pattern, text)
    if match:
        amount_str = match.group(1).replace('\xa0', '').replace(' ', '')
        return int(amount_str)
    return None

def send_email(config: Dict, data: Dict, recipient: str):
    """Send email notification."""
    msg = MIMEMultipart()
    msg['From'] = config['sender_email']
    msg['To'] = recipient
    msg['Subject'] = "Traffic Violation Alert"
    
    body = f"""
    Violation detected!
    POSTANOVLENIE: {data['ПОСТАНОВЛЕНИЕ']}
    Address: {data['адрес']}
    Car Number: {data['номер_тс']}
    Fine Amount: {data['сумма_штрафа']}
    """
    
    msg.attach(MIMEText(body, 'plain'))
    try:
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['sender_email'], config['sender_password'])
            server.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

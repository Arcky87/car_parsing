

import cv2
import numpy as np
import re
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional

def preprocess_image(path: str, config: dict) -> np.ndarray:
    """Preprocesses the image for OCR using configuration parameters."""
    img_config = config['image_processing']
    
    carplate = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(
        carplate, 
        None, 
        fx=img_config['resize_factor'], 
        fy=img_config['resize_factor'], 
        interpolation=cv2.INTER_CUBIC
    )
    
    resized_image = cv2.GaussianBlur(
        resized_image, 
        (img_config['blur_kernel_size'], img_config['blur_kernel_size']), 
        0
    )
    
    clahe = cv2.createCLAHE(
        clipLimit=img_config['clahe']['clip_limit'], 
        tileGridSize=tuple(img_config['clahe']['tile_grid_size'])
    )
    cl_image = clahe.apply(resized_image)
    
    denoised = cv2.fastNlMeansDenoising(
        cl_image,
        h=img_config['denoise']['h'],
        templateWindowSize=img_config['denoise']['template_window_size'],
        searchWindowSize=img_config['denoise']['search_window_size']
    )
    
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    
    thresh = cv2.adaptiveThreshold(
        normalized, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 
        img_config['threshold']['block_size'],
        img_config['threshold']['C']
    )
    
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX)
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

def save_processed_image(image, original_pdf_name, prefix="processed", create_dirs=True):
    """
    Saves processed CV2 image to numbers/processed directory.
    """
    # Create processed directory inside numbers
    base_dir = os.path.join(os.path.dirname(os.path.dirname(original_pdf_name)), "numbers")
    processed_dir = os.path.join(base_dir, "processed")
    
    if create_dirs:
        os.makedirs(processed_dir, exist_ok=True)
    
    # Create filename based on original PDF name
    base_name = os.path.splitext(os.path.basename(original_pdf_name))[0]
    output_filename = os.path.join(processed_dir, f"{prefix}_{base_name}.png")
    
    # Save the image
    cv2.imwrite(output_filename, image)
    print(f"Saved processed image: {output_filename}")
    
    return output_filename

def enhance_plate(image):
    """Enhance the plate image using multiple image processing techniques."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize image for better processing
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(scaled)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(
        contrast_enhanced,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )
    
    # Sharpen
    kernel = np.array([[-1,-1,-1],
                        [-1, 9,-1],
                        [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Normalize
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized

def send_email(config: dict, data: dict, recipient: str):
    """Send email notification using configuration."""
    msg = MIMEMultipart()
    msg['From'] = config['email']['sender_email']
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
        with smtplib.SMTP(config['email']['smtp_server'], 
                         config['email']['smtp_port']) as server:
            server.starttls()
            server.login(
                config['email']['sender_email'],
                config['email']['sender_password']
            )
            server.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

def send_accident_statistics(df, config: Dict, recipient: str):
    msg = MIMEMultipart()
    msg['From'] = config['email']['sender_email']
    msg['To'] = recipient
    msg['Subject'] = "Daily accident statistics"
    accidents_by_address = df.groupby('адрес').size().reset_index(name='count')
    body = f"""
    Accident statistics by address:\n\n
    {accidents_by_address.to_string()}
    """
    msg.attach(MIMEText(body, 'plain'))
    try:
        with smtplib.SMTP(config['email']['smtp_server'], 
                         config['email']['smtp_port']) as server:
            server.starttls()
            server.login(
                config['email']['sender_email'],
                config['email']['sender_password']
            )
            server.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

def create_postgresql_table(conn):
  try:
    with conn.cursor() as cur:
      cur.execute("""
               CREATE TABLE IF NOT EXISTS traffic_violations (
                   id SERIAL PRIMARY KEY,
                   filename VARCHAR(255),
                   постановление VARCHAR(255),
                   дата_нарушения DATE,
                   время_нарушения TIME,
                   адрес TEXT,
                   номер_тс VARCHAR(255),
                   сумма_штрафа INT,
                   сумма_штрафа_greater_5000 BOOLEAN,
                   номер_свидетельства VARCHAR(255),
                   car_photo_filename VARCHAR(255),
                   regnumber_filename VARCHAR(255),
                   ocr_match BOOLEAN,
                   игр VARCHAR(255)
               );
               """)
      conn.commit()
  except Exception as e:
    print(f"Error creating table: {e}")

#Insert extracted data into the database table.
def insert_data_into_db(conn, data_dicts):
  try:
    with conn.cursor() as cur:
      for data in data_dicts:
        cur.execute("""
               INSERT INTO traffic_violations (
                   filename, постановление, дата_нарушения, время_нарушения,
                   адрес, номер_тс, сумма_штрафа, сумма_штрафа_greater_5000,
                   номер_свидетельства, car_photo_filename, regnumber_filename,
                   ocr_match, игр
               ) VALUES (
                   %(filename)s, %(ПОСТАНОВЛЕНИЕ)s, %(дата_нарушения)s, %(время_нарушения)s,
                   %(адрес)s, %(номер_тс)s, %(сумма_штрафа)s, %(сумма_штрафа_greater_5000)s,
                   %(номер_свидетельства)s, %(car_photo_filename)s, %(regnumber_filename)s,
                   %(ocr_match)s, %(ИГР)s
               )
               """, data)
        conn.commit()
  except Exception as e:
    print(f"Error inserting data: {e}")

#Establish a connection to the PostgreSQL database just in case
def connect_to_db():
  try:
    conn = psycopg2.connect(
        dbname="your_db_name",
        user="your_username",
        password="your_password",
        host="localhost",
        port="5432"
        )
    return conn
  except Exception as e:
      print(f"Error connecting to database: {e}")
      return None

# Filter directories for PDF files based on today's creation date.
def get_todays_files(directory):
  today = datetime.today().date()
  todays_files = []
  for filename in os.listdir(directory):
      file_path = os.path.join(directory, filename)
      if os.path.isfile(file_path):
          creation_time = datetime.fromtimestamp(os.path.getctime(file_path)).date()
          if creation_time == today:
              todays_files.append(file_path)
  return todays_files
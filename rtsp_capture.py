import cv2
import os
import time
import numpy as np
import requests
import base64
import json
import threading
from datetime import datetime
from fastapi import FastAPI, Request
import uvicorn

# Configuration settings
RTSP_URL = "rtsp://metaverse911:hellomoto123@192.168.1.59:554/stream1"
SAVE_DIR = "screenshots"
SCREENSHOT_INTERVAL = 3.0  # seconds
FRAME_DIFF_THRESHOLD = 50.0  # Minimum percentage difference between frames to consider a change
GEMINI_API_KEY = "AIzaSyApZxWo7nkeSaNnozStY14DRoJSRm21iEU"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
API_SERVER_URL = "http://localhost:8000/card_detection"  # FastAPI server endpoint
CONTROL_SERVER_PORT = 8001  # Port for the control server

# Global flag to control card detection
detection_enabled = False

# Create a FastAPI app for the control server
control_app = FastAPI()

@control_app.post("/control")
async def control_detection(request: Request):
    """
    Endpoint to control whether card detection is enabled
    """
    global detection_enabled
    data = await request.json()
    if "detection_enabled" in data:
        detection_enabled = data["detection_enabled"]
        print(f"Card detection {'enabled' if detection_enabled else 'disabled'}")
    return {"status": "success", "detection_enabled": detection_enabled}

def start_control_server():
    """Start the control server in a separate thread"""
    uvicorn.run(control_app, host="0.0.0.0", port=CONTROL_SERVER_PORT, log_level="error")

def create_save_dir(directory):
    """Create directory to save screenshots if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def frames_are_different(frame1, frame2, threshold_percent=FRAME_DIFF_THRESHOLD):
    """
    Compare two frames to determine if they are significantly different
    
    Args:
        frame1: First frame
        frame2: Second frame
        threshold_percent: Percentage difference threshold (0-100)
        
    Returns:
        Boolean indicating if frames are different beyond the threshold
    """
    if frame1 is None or frame2 is None:
        return True
    
    # Convert frames to grayscale for comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(gray1, gray2)
    
    # Calculate percentage of pixels that are different
    non_zero_count = np.count_nonzero(frame_diff)
    total_pixels = frame_diff.shape[0] * frame_diff.shape[1]
    diff_percent = (non_zero_count / total_pixels) * 100
    
    return diff_percent > threshold_percent

def send_to_gemini_api(image_path):
    """
    Send an image to Gemini API to detect and extract card details
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Tuple: (is_card_detected, card_details_dict)
    """
    # Check if detection is enabled
    if not detection_enabled:
        print("⚠️ Card detection is currently disabled. Skipping analysis.")
        return False, None
        
    # Read the image and encode as base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Prepare the request payload with specific instructions
    payload = {
        "contents": [{
            "parts": [
                {
                    "text": """Analyze this image for a business or ID card. 
                    If a card is detected, extract the following details in JSON format:
                    
                    {
                        "card_detected": true/false,
                        "name": "[FULL NAME or 'NA' if not available]",
                        "phone": "[PHONE NUMBER or 'NA' if not available]",
                        "email": "[EMAIL ADDRESS or 'NA' if not available]",
                        "organization": "[ORGANIZATION NAME or 'NA' if not available]",
                        "card_number": "[CARD NUMBER or 'NA' if not available]",
                        "expiration_date": "[EXPIRATION DATE or 'NA' if not available]",
                        "card_type": "[CARD TYPE or 'NA' if not available]"
                    }
                    
                    If no card is detected in the image, just respond with:
                    {
                        "card_detected": false
                    }
                    
                    IMPORTANT: Return only the JSON with no additional text."""
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }
                }
            ]
        }]
    }
    
    # Send request to Gemini API
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        
        # Extract the text from the response
        if "candidates" in result and len(result["candidates"]) > 0:
            if "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                extracted_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                # Try to parse the JSON response
                try:
                    # Clean up the text in case there's any markdown formatting
                    if "```json" in extracted_text:
                        json_str = extracted_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in extracted_text:
                        json_str = extracted_text.split("```")[1].strip()
                    else:
                        json_str = extracted_text
                    
                    card_data = json.loads(json_str)
                    
                    # Check if a card was detected
                    if card_data.get("card_detected", False):
                        print("✅ Card detected!")
                        print(f"Name: {card_data.get('name', 'NA')}")
                        print(f"Phone: {card_data.get('phone', 'NA')}")
                        print(f"Email: {card_data.get('email', 'NA')}")
                        print(f"Organization: {card_data.get('organization', 'NA')}")
                        print(f"Card Number: {card_data.get('card_number', 'NA')}")
                        print(f"Expiration Date: {card_data.get('expiration_date', 'NA')}")
                        print(f"Card Type: {card_data.get('card_type', 'NA')}")
                        
                        # Save the extracted details to a text file with same timestamp
                        base_name = os.path.basename(image_path)
                        timestamp = base_name.split("_")[1].split(".")[0]
                        text_file_path = os.path.join(os.path.dirname(image_path), f"card_details_{timestamp}.json")
                        
                        with open(text_file_path, "w") as text_file:
                            json.dump(card_data, text_file, indent=4)
                        
                        print(f"Details saved to: {text_file_path}")
                        
                        # Send card data to FastAPI server
                        send_to_api_server(card_data)
                        
                        return True, card_data
                    else:
                        print("❌ No card detected in the image.")
                        # Only send negative results if detection is enabled
                        if detection_enabled:
                            send_to_api_server({"card_detected": False})
                        return False, None
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response: {e}")
                    print(f"Raw response: {extracted_text}")
                    return False, None
    
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
    
    return False, None

def send_to_api_server(card_data):
    """
    Send card detection data to the FastAPI server
    
    Args:
        card_data (dict): Card data to send
    """
    global detection_enabled
    
    try:
        response = requests.post(API_SERVER_URL, json=card_data)
        response.raise_for_status()
        print(f"✅ Data sent to API server: {response.json()}")
        
        # If we successfully detected a card, immediately disable detection
        if card_data.get("card_detected", False):
            detection_enabled = False
            print("🛑 Card successfully detected. Disabling further card detection.")
            print("⏳ Waiting for silence before enabling card detection again.")
            
    except Exception as e:
        print(f"❌ Error sending data to API server: {e}")

def check_detection_status():
    """Check if card detection is enabled by the API server"""
    global detection_enabled
    try:
        response = requests.get("http://localhost:8000/detection_status")
        if response.status_code == 200:
            data = response.json()
            detection_enabled = data.get("detection_enabled", False)
            print(f"Card detection status from API server: {'enabled' if detection_enabled else 'disabled'}")
            return detection_enabled
    except Exception as e:
        print(f"Error checking detection status: {e}")
    return detection_enabled

def capture_rtsp_stream(rtsp_url=RTSP_URL, save_dir=SAVE_DIR, interval=SCREENSHOT_INTERVAL):
    """
    Capture frames from RTSP stream and save screenshots at specified interval
    
    Args:
        rtsp_url (str): RTSP URL to connect to
        save_dir (str): Directory to save screenshots
        interval (float): Interval between screenshots in seconds
    """
    global detection_enabled
    
    # Create directory for screenshots
    save_dir = create_save_dir(save_dir)
    
    # Configure OpenCV for optimal RTSP performance
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Set buffer size to minimum to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Additional RTSP optimization parameters
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024"
    
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    print(f"Connected to RTSP stream: {rtsp_url}")
    print(f"Saving screenshots to: {save_dir}")
    print(f"Screenshot interval: {interval} seconds")
    print(f"Frame difference threshold: {FRAME_DIFF_THRESHOLD}%")
    print("Press 'q' to quit")
    
    last_save_time = 0
    previous_frame = None
    card_detected_count = 0
    first_frame = True  # Flag to handle the first frame
    last_status_check = time.time()
    last_detection_status = detection_enabled
    
    try:
        while True:
            # Check detection status periodically
            current_time = time.time()
            if current_time - last_status_check >= 10.0:  # Check every 10 seconds
                check_detection_status()
                last_status_check = current_time
                
                # Log status change if it changed
                if last_detection_status != detection_enabled:
                    if detection_enabled:
                        print("🔍 Card detection has been ENABLED. Looking for cards...")
                    else:
                        print("🛑 Card detection has been DISABLED. Waiting for session to end...")
                    last_detection_status = detection_enabled
                
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to receive frame. Reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                continue
            
            # Display the frame with status
            if not detection_enabled:
                # Display status when detection is disabled
                cv2.putText(frame, "SESSION ACTIVE - CARD VERIFIED", (30, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "LOOKING FOR CARD...", (30, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
                
            cv2.imshow('RTSP Stream', frame)
            
            # On the first valid frame, establish a baseline and skip analysis
            if first_frame:
                previous_frame = frame.copy()
                first_frame = False
                last_save_time = time.time()
                print("✅ Baseline frame established. Monitoring for changes...")
                continue
            
            # Process subsequent frames at the specified interval
            current_time = time.time()
            if current_time - last_save_time >= interval:
                # Reset the timer regardless of whether we process the frame
                last_save_time = current_time
                
                # Only proceed with analysis if detection is enabled
                if detection_enabled:
                    # Check if frame is different from the previous one we analyzed
                    if frames_are_different(previous_frame, frame):
                        # Only save and analyze frames that show significant change
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = os.path.join(save_dir, f"screenshot_{timestamp}.jpg")
                        cv2.imwrite(screenshot_path, frame)
                        print(f"✅ Significant change detected! Saved: {screenshot_path}")
                        
                        # Send to Gemini API for card detection and extraction
                        print("🔍 Analyzing image for card details...")
                        is_card, card_data = send_to_gemini_api(screenshot_path)
                        
                        if is_card:
                            card_detected_count += 1
                            # Add a visual indicator that a card was detected
                            cv2.putText(frame, "CARD DETECTED", (50, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Update previous frame to the new one that was just analyzed
                        previous_frame = frame.copy()
                    else:
                        print("❌ No significant frame change detected. Skipping capture and analysis.")
                else:
                    # Just display status when detection is disabled
                    pass
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()
        print(f"Stream capture ended. Cards detected: {card_detected_count}")

if __name__ == "__main__":
    # Start the control server in a separate thread
    control_thread = threading.Thread(target=start_control_server, daemon=True)
    control_thread.start()
    
    # Wait a moment for the server to start
    time.sleep(1)
    
    # Check initial detection status
    check_detection_status()
    
    # Start the RTSP capture
    capture_rtsp_stream() 
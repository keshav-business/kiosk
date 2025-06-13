import os
import json
import base64
import asyncio
import logging
import requests
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi.templating import Jinja2Templates

# Configure logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "api_server.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

# Gemini API settings
GEMINI_API_KEY = "AIzaSyApZxWo7nkeSaNnozStY14DRoJSRm21iEU"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Session management
SILENCE_TIMEOUT = 5.0  # Seconds of silence before ending a session
active_sessions = {}  # Dictionary to track active sessions

app = FastAPI(title="Avatar Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This server is now API-only.
# The `index.html` file can be opened directly in a web browser.

# Models for API
class CardInfo(BaseModel):
    card_detected: bool
    name: Optional[str] = "NA"
    phone: Optional[str] = "NA"
    email: Optional[str] = "NA"
    organization: Optional[str] = "NA"
    card_number: Optional[str] = "NA"
    expiration_date: Optional[str] = "NA"
    card_type: Optional[str] = "NA"

class UserQuery(BaseModel):
    query: str

class SessionControl(BaseModel):
    action: str  # "start" or "stop"
    client_id: str

# Store active websocket connections
active_connections: List[WebSocket] = []
# Store the latest detected card info
latest_card_info: Optional[CardInfo] = None
# Flag to control RTSP card detection
rtsp_detection_enabled = False

# Helper function to query Gemini API
async def query_gemini_api(prompt, system_instructions=None):
    try:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 2048
            }
        }
        
        # Add system instructions if provided
        if system_instructions:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instructions}]
            }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        if "candidates" in result and len(result["candidates"]) > 0:
            if "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
        
        return "I'm sorry, I couldn't generate a response."
    except Exception as e:
        logger.error(f"Error querying Gemini API: {str(e)}")
        return f"Sorry, there was an error: {str(e)}"

# Helper function to control RTSP detection
def set_rtsp_detection_state(enable: bool):
    global rtsp_detection_enabled
    rtsp_detection_enabled = enable
    
    try:
        # Notify the RTSP script about the new state
        requests.post("http://localhost:8001/control", 
                     json={"detection_enabled": enable})
        logger.info(f"RTSP card detection {'enabled' if enable else 'disabled'}")
        return True
    except Exception as e:
        logger.error(f"Failed to communicate with RTSP script: {e}")
        return False

# WebSocket endpoint for real-time communication with the avatar frontend
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Generate a unique client ID for this connection
    connection_index = len(active_connections)
    client_id = f"client_{connection_index + 1}_{datetime.now().timestamp()}"
    active_connections.append(websocket)
    logger.info(f"New websocket connection established: {client_id}")
    logger.info(f"Active sessions: {list(active_sessions.keys())}")
    
    # Enable RTSP detection when first client connects and no active sessions exist
    if len(active_connections) == 1 and not active_sessions:
        set_rtsp_detection_state(True)
        logger.info("First client connected. Card detection enabled.")
    
    try:
        # If a card has already been verified in a previous session, welcome the user back
        if latest_card_info and latest_card_info.card_detected:
            first_name = latest_card_info.name.split(' ')[0] if latest_card_info.name != "NA" else "user"
            await websocket.send_text(f"card_detected:{json.dumps(latest_card_info.dict())}")
            await websocket.send_text(f"ai_response:Welcome back, {first_name}! I'm ready to assist.")
            
            # Start a new session for this client
            active_sessions[client_id] = {
                "last_activity": datetime.now().timestamp(),
                "card_info": latest_card_info,
                "connection_index": connection_index
            }
        else:
            # Otherwise, send the initial welcome and verification prompt
            await websocket.send_text("ai_response:Welcome, system activated. Please present your card for verification.")
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message from {client_id}: {data}")
            
            # Update last activity timestamp to prevent timeout
            if client_id in active_sessions:
                active_sessions[client_id]["last_activity"] = datetime.now().timestamp()
            
            if data == "thankyou and done":
                await websocket.send_text("session_ended")
                # End the session
                if client_id in active_sessions:
                    del active_sessions[client_id]
                break
            
            if data == "silence_detected":
                # End the session due to silence
                if client_id in active_sessions:
                    del active_sessions[client_id]
                    await websocket.send_text("ai_response:Session ended due to inactivity. Please present your card to start a new session.")
                    
                    # Enable card detection only if this was the last active session
                    if not active_sessions:
                        logger.info("All sessions ended. Re-enabling card detection.")
                        set_rtsp_detection_state(True)
                continue
            
            if data.startswith("question:"):
                # Block interaction if no card has been detected yet
                if client_id not in active_sessions:
                    logger.warning(f"Client {client_id} attempted to ask a question without verification")
                    logger.warning(f"Active sessions: {list(active_sessions.keys())}")
                    
                    # Check if there are any active sessions for this connection index
                    connection_index = active_connections.index(websocket)
                    matching_sessions = [cid for cid, session in active_sessions.items() 
                                        if session.get("connection_index") == connection_index]
                    
                    if matching_sessions:
                        # Use the first matching session
                        client_id = matching_sessions[0]
                        logger.info(f"Found matching session by connection index: {client_id}")
                    else:
                        # No matching session found, ask for verification
                        await websocket.send_text("ai_response:Please present your card for verification before we can proceed.")
                        continue

                question = data[9:]  # Extract question text
                logger.info(f"Processing question from {client_id}: {question}")
                
                # Create system instructions based on card detection
                system_instructions = "You are a helpful AI assistant. Keep your responses concise and engaging."
                card_info = active_sessions[client_id].get("card_info")
                if card_info:
                    name = card_info.name if card_info.name != "NA" else "user"
                    org = f" from {card_info.organization}" if card_info.organization != "NA" else ""
                    system_instructions = f"You are a helpful AI assistant speaking with {name}{org}. Be professional, engaging and concise in your responses. Personalize your responses using their name occasionally."
                
                # Get response from Gemini API
                response = await query_gemini_api(question, system_instructions)
                logger.info(f"Sending AI response to {client_id}")
                await websocket.send_text(f"ai_response:{response}")
    
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {str(e)}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        
        # Clean up the session
        if client_id in active_sessions:
            del active_sessions[client_id]
            
        # Disable RTSP detection when last client disconnects
        if not active_connections:
            set_rtsp_detection_state(False)
            
        logger.info(f"WebSocket connection closed for {client_id}")

# Endpoint for RTSP script to report card detection
@app.post("/card_detection")
async def card_detection(card_info: CardInfo):
    global latest_card_info
    logger.info(f"Received card detection: {card_info}")
    
    # Only act if a card was positively detected and RTSP detection is enabled
    if card_info.card_detected and rtsp_detection_enabled:
        latest_card_info = card_info
        
        # Broadcast the successful detection to all clients
        broadcast_count = 0
        for i, connection in enumerate(active_connections):
            try:
                # Send card data with a special flag to start listening immediately
                card_data = card_info.dict()
                card_data["start_listening"] = True
                
                # Ensure proper JSON formatting by using json.dumps
                json_string = json.dumps(card_data)
                await connection.send_text(f"card_detected:{json_string}")
                
                # Send a personalized greeting as an AI response to be spoken
                first_name = card_info.name.split(' ')[0] if card_info.name != "NA" else "User"
                greeting = f"Identity verified. Welcome, {first_name}! How can I help you today?"
                await connection.send_text(f"ai_response:{greeting}")
                
                # Create or update session for this client
                # Always create a new session for this connection to ensure it's properly tracked
                client_id = f"client_{i + 1}_{datetime.now().timestamp()}"
                active_sessions[client_id] = {
                    "last_activity": datetime.now().timestamp(),
                    "card_info": card_info,
                    "connection_index": i  # Store the connection index for future reference
                }
                
                logger.info(f"Created session for client {client_id} with connection index {i}")
                broadcast_count += 1
            except Exception as e:
                logger.error(f"Error sending to websocket: {str(e)}")
        
        # Disable further card detection now that we have a valid card
        set_rtsp_detection_state(False)
        logger.info("✅ Valid card detected! Card detection has been DISABLED until session ends.")
        logger.info(f"Active sessions after card detection: {list(active_sessions.keys())}")
        
        logger.info(f"Broadcast card detection to {broadcast_count} connections")
        return {"status": "success", "message": f"Card detection processed and broadcast to {broadcast_count} connections"}
    else:
        # If no card was detected or detection is disabled, log it but do not broadcast
        if not rtsp_detection_enabled:
            logger.info("Card detection received but detection is currently disabled - user session is active.")
        else:
            logger.info("No card detected in the frame. No action taken.")
        return {"status": "success", "message": "No card detected or detection disabled."}

# Control endpoint for RTSP script to check if detection is enabled
@app.get("/detection_status")
async def detection_status():
    return {"detection_enabled": rtsp_detection_enabled}

# Direct text query endpoint (alternative to WebSocket)
@app.post("/query")
async def query(user_query: UserQuery):
    try:
        # Create system instructions based on card detection
        system_instructions = "You are a helpful AI assistant."
        if latest_card_info and latest_card_info.card_detected:
            name = latest_card_info.name if latest_card_info.name != "NA" else "user"
            org = f" from {latest_card_info.organization}" if latest_card_info.organization != "NA" else ""
            system_instructions = f"You are a helpful AI assistant speaking with {name}{org}. Be professional and concise in your responses."
        
        response = await query_gemini_api(user_query.query, system_instructions)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(active_connections),
        "active_sessions": len(active_sessions),
        "rtsp_detection_enabled": rtsp_detection_enabled
    }

# Add debug endpoint to show current state
@app.get("/debug")
async def debug_info():
    return {
        "active_connections": len(active_connections),
        "active_sessions": list(active_sessions.keys()),
        "card_info": latest_card_info.dict() if latest_card_info else None,
        "rtsp_detection_enabled": rtsp_detection_enabled,
        "server_time": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("api_server:app", host=host, port=port, reload=True) 
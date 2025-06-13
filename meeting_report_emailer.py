import os
import json
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from typing import Dict, List
from pathlib import Path
import logging
from dotenv import load_dotenv
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Email configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')

class MeetingReportEmailer:
    def __init__(self, meeting_logs_dir: str = "meeting_logs"):
        self.meeting_logs_dir = Path(meeting_logs_dir)
        
    def parse_meeting_log(self, log_path: Path) -> Dict:
        """Parse a meeting log file into a structured format"""
        meeting_data = {
            'meeting_info': {},
            'questions_and_responses': []
        }
        
        current_section = None
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('==='):
                    current_section = line.strip('= ')
                    continue
                    
                if ':' in line:
                    key, value = map(str.strip, line.split(':', 1))
                    if current_section == 'Meeting Summary':
                        meeting_data['meeting_info'][key.lower()] = value
                    elif current_section == 'Questions and Responses':
                        if key.startswith('Q'):
                            meeting_data['questions_and_responses'].append({'question': value})
                        elif key.startswith('A') and meeting_data['questions_and_responses']:
                            meeting_data['questions_and_responses'][-1]['answer'] = value
                            
        return meeting_data

    def generate_html_content(self, meeting_data: Dict) -> str:
        """Generate HTML content with embedded CSS for the email"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .header {
                    background: linear-gradient(135deg, #2c3e50, #3498db);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .section {
                    background: white;
                    padding: 25px;
                    margin-bottom: 25px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    border-left: 4px solid #3498db;
                }
                .participant-info {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                    margin-bottom: 20px;
                }
                .info-item {
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 6px;
                }
                .qa-item {
                    margin-bottom: 20px;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 8px;
                }
                .question {
                    color: #2c3e50;
                    font-weight: 600;
                    margin-bottom: 10px;
                }
                .answer {
                    color: #34495e;
                    padding-left: 20px;
                    border-left: 3px solid #3498db;
                }
                .metrics {
                    display: flex;
                    justify-content: space-between;
                    background: #edf2f7;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                }
                .metric-item {
                    text-align: center;
                }
                .metric-value {
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #2c3e50;
                }
                .overview {
                    font-size: 1.1em;
                    line-height: 1.8;
                    color: #2c3e50;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="margin:0;font-size:28px;">Meeting Summary Report</h1>
                <p style="margin:10px 0 0;font-size:16px;">Generated on """ + datetime.now().strftime('%Y-%m-%d') + """</p>
            </div>

            <div class="section">
                <h2>Meeting Details</h2>
                <div class="participant-info">
                    <div class="info-item">
                        <strong>Participant:</strong> """ + meeting_data['meeting_info'].get('participant_name', '') + """
                    </div>
                    <div class="info-item">
                        <strong>Company:</strong> """ + meeting_data['meeting_info'].get('company', '') + """
                    </div>
                    <div class="info-item">
                        <strong>Email:</strong> """ + meeting_data['meeting_info'].get('email', '') + """
                    </div>
                    <div class="info-item">
                        <strong>Phone:</strong> """ + meeting_data['meeting_info'].get('phone', '') + """
                    </div>
                </div>
                <div class="metrics">
                    <div class="metric-item">
                        <div class="metric-value">""" + meeting_data['meeting_info'].get('start_time', '') + """</div>
                        <div>Start Time</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">""" + meeting_data['meeting_info'].get('end_time', '') + """</div>
                        <div>End Time</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">""" + meeting_data['meeting_info'].get('duration_minutes', '') + """m</div>
                        <div>Duration</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">""" + str(len(meeting_data['questions_and_responses'])) + """</div>
                        <div>Questions</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Discussion Overview</h2>
                <div class="overview">
                    The conversation discussed virtual reality as a computer-generated environment that simulates a physical presence in real or imagined worlds.
                </div>
            </div>

            <div class="section">
                <h2>Detailed Discussion</h2>
        """
        
        # Add Q&A section
        for i, qa in enumerate(meeting_data['questions_and_responses'], 1):
            html_content += f"""
                <div class="qa-item">
                    <div class="question">Q{i}: {qa['question']}</div>
                    <div class="answer">A{i}: {qa['answer']}</div>
                </div>
            """
        
        html_content += """
            </div>

            <div style="text-align:center;margin-top:30px;color:#666;font-size:14px;">
                <p>Generated by Metaverse911 Meeting Assistant</p>
            </div>
        </body>
        </html>
        """
        
        return html_content

    def send_email(self, recipient_email: str, html_content: str, subject: str = "Meeting Summary Report"):
        """Send the HTML email"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = SENDER_EMAIL
            msg['To'] = recipient_email

            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
                
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False


    def process_and_send_report(self, log_file: str) -> bool:
        """Process a meeting log file and send the report"""
        try:
            log_path = log_file
            print("file received")
            print(log_path)
           

            # Read the log file content
            with open(log_path, 'r') as file:
                log_content = file.read()

            # Use regex to extract the recipient email (expects a line like "Email: example@domain.com")
            email_match = re.search(r'Email:\s*([\w\.-]+@[\w\.-]+)', log_content)
            if email_match:
                recipient_email = email_match.group(1)
            else:
                logger.error("Recipient email not found in log file.")
                return False

            # Parse the meeting log and generate HTML content
            meeting_data = self.parse_meeting_log(log_path)
            html_content = self.generate_html_content(meeting_data)

            # Send the email report
            return self.send_email(
                recipient_email,
                html_content,
                f"Meeting Summary - {meeting_data['meeting_info'].get('date', 'Undated')}"
            )

        except Exception as e:
            logger.error(f"Error processing meeting report: {str(e)}")
            return False

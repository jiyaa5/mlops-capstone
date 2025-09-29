import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
LOG_FILE = "logs/logs.txt"
ERROR_THRESHOLD = 0.3
ALERT_INTERVAL = 15 * 60  # 15 minutes in seconds

last_alert_time = 0  # track last alert timestamp

def send_slack_alert(message: str):
    payload = {"text": f"🚨 ML App Alert: {message}"}
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            print("✅ Slack alert sent!")
        else:
            print(f"Slack alert failed: {response.status_code}, {response.text}")
    except Exception as e:
        print("Slack request failed:", e)

def check_logs():
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        if not lines:
            return False  # no errors

        errors = [l for l in lines if "ERROR" in l]
        error_rate = len(errors) / len(lines)
        print(f"Checked {len(lines)} lines: {len(errors)} errors → rate={error_rate:.2f}")

        return error_rate > ERROR_THRESHOLD  # return True if alert needed

    except FileNotFoundError:
        print("No logs found yet.")
        return False

if __name__ == "__main__":
    print("Starting log monitor...")
    last_alert_time = 0

    while True:
        error_detected = check_logs()
        current_time = time.time()

        if error_detected and (current_time - last_alert_time > ALERT_INTERVAL):
            alert_msg = "High error rate detected in ML app logs!"
            send_slack_alert(alert_msg)
            last_alert_time = current_time

        time.sleep(10)  # check logs every 10 seconds

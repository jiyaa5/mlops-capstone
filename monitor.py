import time

# TODO: Implement reading logs and sending email/slack alerts

while True:
    with open("logs.txt", "r") as f:
        lines = f.readlines()
        # TODO: Add condition to detect error/warning logs
        # TODO: Trigger alert when threshold crossed
    time.sleep(10)

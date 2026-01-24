import requests
import json
import argparse


def send_google_chat_message(webhook_url, message):
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    data = {"text": message}
    response = requests.post(webhook_url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")

# 본문 코드
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True, help="실행 완료된 파이썬 스크립트 이름")
    args = parser.parse_args()

    send_google_chat_message(
        webhook_url="https://chat.googleapis.com/v1/spaces/AAQAv8aWk00/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=0mVelWFIjOHyIcVSO0XSVJxgT88tYkrjysw2YpD_X_A",
        message = f"✅ 파이썬 코드 실행이 완료되었습니다! ({args.script})"
    )

if __name__ == "__main__":
    main()

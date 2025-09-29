# GPT4All-J Local Setup (CPU 기반)

## 설치
```bash
pip install -r requirements.txt

## 최신 모델
https://gpt4all.io/models/gguf/gpt4all-falcon-newbpe-q4_0.gguf

## 서버실행
 uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

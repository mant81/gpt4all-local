from fastapi import FastAPI
from pydantic import BaseModel
from gpt4allj import Model  # 로컬 gpt4allj 설치 필요
import json
import os

app = FastAPI()

MODEL_PATH = "models/ggml-gpt4all-j-v1.3-groovy.bin"

# 모델 로드
model = Model(MODEL_PATH)
print("모델 로딩 완료")

# JSON 문서 로드
DATA_PATH = "data/dev_docs.json"
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dev_docs = json.load(f)
    print(f"문서 {len(dev_docs)}개 로딩 완료")
else:
    dev_docs = []

# 요청 Body 정의
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate/")
def generate(req: PromptRequest):
    # 프롬프트 구성
    context = "\n".join([f"Q: {doc['input']}\nA: {doc['output']}" for doc in dev_docs])
    prompt = f"{context}\n\n사용자 질문: {req.prompt}\n답변:"

    try:
        response = model.generate(prompt, max_tokens=300, temp=0.7)
        if isinstance(response, bytes):
            response = response.decode("utf-8", errors="ignore")
    except Exception as e:
        response = f"모델 생성 중 오류 발생: {str(e)}"

    return {"response": response}

@app.get("/")
def root():
    return {"message": "로컬 GGML 기반 GPT4All-J 서버", "docs_count": len(dev_docs)}

# src/api/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json

app = FastAPI(title="GPT4All-J FastAPI Server")

# 모델과 토크나이저 로드
MODEL_NAME = "nomic-ai/gpt4all-j"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"모델 로딩 중... device={device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto" if device=="cuda" else None)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)
print("모델 로딩 완료!")

# JSON 문서 로드
try:
    with open("data/dev_docs.json", "r", encoding="utf-8") as f:
        dev_docs = json.load(f)
    print(f"문서 {len(dev_docs)}개 로딩 완료")
except Exception as e:
    print(f"문서 로딩 실패: {e}")
    dev_docs = []

# 요청 Body 정의
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate/")
def generate(req: PromptRequest, max_tokens: int = 150):
    # 문서 내용을 프롬프트에 포함
    docs_text = "\n".join([f"Q: {doc['input']}\nA: {doc['output']}" for doc in dev_docs])
    prompt = f"{docs_text}\n사용자 질문: {req.prompt}\n답변:"

    try:
        result = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
        answer = result[0]['generated_text'][len(prompt):].strip()
    except Exception as e:
        answer = f"모델 생성 중 오류 발생: {e}"

    return {"response": answer}

@app.get("/")
def root():
    return {"message": "GPT4All-J FastAPI Server", "docs_count": len(dev_docs)}

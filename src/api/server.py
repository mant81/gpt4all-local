from fastapi import FastAPI
from pydantic import BaseModel
from gpt4all import GPT4All

app = FastAPI()

# 로컬 GGML Lora 모델 경로
MODEL_PATH = "models/gpt4all-falcon-newbpe-q4_0.gguf"
MODEL_DIR = "models"
try:
    # model_name: 식별용 이름 (로컬 경로도 가능)
    # model_type: ggml
    # allow_download=False 로 다운로드 방지
    model = GPT4All(
        model_name="gpt4all-falcon-newbpe-q4_0.gguf",  # 원하는 식별 이름
        model_path=MODEL_DIR,
        model_type="ggml",
        allow_download=False
    )
    print("모델 로딩 완료")
except Exception as e:
    model = None
    print(f"모델 로딩 실패: {e}")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate/")
def generate(req: PromptRequest):
    if model is None:
        return {"response": "모델을 로드할 수 없습니다."}
    response = model.generate("다음 질문에 반드시 한국어로만 답해:"+ req.prompt)
    return {"response": response}

@app.get("/")
def root():
    return {"message": "GGML Lora 모델 기반 GPT4All API"}

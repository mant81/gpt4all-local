#!/usr/bin/env python3
"""
개발 문서 Q&A 테스트 스크립트
"""
import requests
import json
import time

def test_qa():
    """Q&A 테스트"""
    base_url = "http://localhost:8001"
    
    # 테스트 질문들
    test_questions = [
        "사용자 인증 관련 함수 설명",
        "DB 연결 예제",
        "사용자 생성 예제",
        "파일 업로드 처리 예제",
        "비밀번호 암호화 방법",
        "존재하지 않는 기능에 대한 질문"
    ]
    
    print("=== 개발 문서 Q&A 테스트 ===")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 테스트 {i} ---")
        print(f"질문: {question}")
        
        try:
            response = requests.post(
                f"{base_url}/generate/",
                json={"prompt": question},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"답변: {result['response']}")
            else:
                print(f"오류: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print("타임아웃 발생")
        except Exception as e:
            print(f"오류: {e}")
        
        time.sleep(1)  # 요청 간 간격

if __name__ == "__main__":
    test_qa()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import os
import requests
import json
from typing import Optional, Literal
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

class LLM_Message(BaseModel):
    input: Optional[str]
    question: Optional[str]
    model: Literal["llama3-8b", "llama3-405b"] = "llama3-405b"


def generate_stream(message: dict):
    headers = {
    'Authorization': 'Basic ' + os.getenv('FASTAPI_API_KEY', ''),
    'Content-Type': 'application/json',
    }
    json_data = {
        'messages': [
            {
                'role': 'system',
                'content': 'Bạn là một tư vấn viên của học viện kỹ thuật quân sự, hãy trả lời lịch sự. Thông tin tạo ' , 
            },
            {
                'role': 'system',
                'content': f'Đây là văn bản chứa thông tin, bạn có thể dùng nếu cần:\n{message.input}.Hãy trả lời câu hỏi, Nếu câu nào không biết thì trả lời là Tôi không biết' , 
            },
            {
                'role': 'user',
                'content': f'{message.question}',
            },
        ],
        'stop': [
            '<|eot_id|>',
        ],
        'model': f'{message.model}',
        'stream': True,
        'stream_options': {
            'include_usage': True,
        },
    }
    response = requests.post(os.getenv('FASTAPI_URL', ''), headers=headers, json=json_data, stream=True)

    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: ') and line != "data: [DONE]":
                    json_data = line[6:]
                    try:
                        parsed_data = json.loads(json_data)
                        if 'choices' in parsed_data:
                            for choice in parsed_data['choices']:
                                if 'delta' in choice and 'content' in choice['delta']:
                                    yield choice['delta']['content']
                        elif 'usage' in parsed_data and parsed_data['usage'].get('is_last_response', False):
                            break
                    except json.JSONDecodeError:
                        yield "Error decoding JSON: " + json_data
    else:
        yield "Error: " + response.text

message = LLM_Message(
    input="Mã trường: KQH Mã ngành: 7860220 Chỉ tiêu: 540 Tổ hợp xét tuyển: A00 và A01 Thí sinh có thường trú phía Bắc:",
    question="Thi khối gì để vào học viện",
    model="llama3-405b"
)

@app.post("/stream")
def stream_data(message: LLM_Message):
    return StreamingResponse(generate_stream(message), media_type='text/event-stream')

if __name__ == "__main__": 
    import uvicorn 
    uvicorn.run('test:app', port=2001)
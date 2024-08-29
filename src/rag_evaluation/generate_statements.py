from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
import json
load_dotenv()

# Define llm
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0,api_key= os.getenv('GROQ_API_KEY1')) ## Replace to real LLMs (Cohere / Groq / OpenAI)

examples = [{
    "question": "Who was Albert Einstein and what is he best known for?",
    "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    "sentences": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time",
    "analysis": '''
            Albert Einstein was a German-born theoretical physicist.,
            Albert Einstein is recognized as one of the greatest and most influential physicists of all time.,
'''
}]

# Định nghĩa mẫu cho mỗi ví dụ
example_template = """
Examples :\n
<Input>
Question: {question}
Answer: {answer}
Sentences:
{sentences}
</Input>\n\n
<Output>
Analysis: {analysis}
</Output>
"""

example_prompt = PromptTemplate(
    input_variables=["question", "answer", "sentences"],
    template=example_template
)

# Định nghĩa prompt chính
prefix = "Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON. Below is an example.Respone with exactly the same structure with examples. Rely in Vietnamese"

suffix = """
Question: {question}
Answer: {answer}
Sentences:
{sentences}

analysis:
"""

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["question", "answer", "sentences"],
    example_separator="\n\n"
)

import re

def extract_analysis_content(input_str):
    # Tìm nội dung trong <Output> và dưới Analysis
    match = re.search(r'<Output>\s*Analysis:\s*(.*?)\s*</Output>', input_str, re.DOTALL)
    if match:
        analysis_content = match.group(1)
        # Tách các câu thành danh sách các câu
        statements = [s.strip() for s in analysis_content.split(",\n") if s.strip()]
        return statements


def generation_statements(data: dict, output_file: str): 
    json_data = {}
    prompt = few_shot_prompt_template.format(
        question=data["question"],
        answer=data["answer"],
        sentences=data["sentences"]
    )
    out=llm.invoke(prompt)
    json_data = {
        "context": data["sentences"],
        "statements": extract_analysis_content(out.content)
    }
    # Ghi kết quả vào file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    return json_data

def count_sentences(text):
    # Tách văn bản dựa trên dấu chấm, dấu hỏi và dấu chấm than
    sentences = re.split(r'[.!?]', text)
    # Loại bỏ các phần tử trống và câu ngắn
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

# if __name__ == "__main__":
#     new_example =   {
#         "question": "Dụng cụ trong ngành tài nguyên môi trường là gì?",
#         "answer": "Dụng cụ trong ngành tài nguyên môi trường là loại tài sản không đủ tiêu chuẩn về tài sản cố định theo quy định hiện hành của nhà nước, mà người lao động sử dụng để tác động, biến đổi vật liệu thành sản phẩm. Ví dụ: kìm, búa, cờ lê, quần áo bảo hộ và các dụng cụ khác tương tự.",
#         "sentences": "Trong Thông tư này các từ ngữ dưới đây được hiểu như sau:\n1. Định mức kinh tế - kỹ thuật (sau đây gọi tắt là định mức): Là mức hao phí cần thiết về lao động về nguyên, nhiên vật liệu, máy móc thiết bị, dụng cụ và phương tiện để hoàn thành một đơn vị sản phẩm (hoặc một khối lượng công việc nhất định), trong một điều kiện cụ thể của các hoạt động điều tra cơ bản trong các lĩnh vực thuộc phạm vi quản lý nhà nước của Bộ.\n2. Dụng cụ là loại tài sản không đủ tiêu chuẩn về tài sản cố định theo quy định hiện hành của nhà nước mà người lao động sử dụng để tác động, biến đổi vật liệu thành sản phẩm (kìm, búa, cờ lê, quần áo bảo hộ và các dụng cụ khác tương tự).\n3. Vật liệu là đầu vào trong một quá trình sản xuất hoặc thực hiện nhiệm vụ chuyên môn cụ thể, được người lao động tác động, biến đổi hoàn toàn để thành sản phẩm theo yêu cầu đặt ra.\n4. Máy móc thiết bị là công cụ lao động thuộc tài sản cố định hữu hình và tài sản cố định vô hình theo tiêu chuẩn quy định hiện hành của nhà nước về tài sản cố định (không bao gồm nhà xưởng và quyền sử dụng đất) mà người lao động sử dụng để tác động, biến đổi vật liệu thành sản phẩm."
#     }
#     # Ghi kết quả vào file "output.json"
#     # generation_statements(new_example, "output.json")
#     print(count_sentences(new_example['sentences']))

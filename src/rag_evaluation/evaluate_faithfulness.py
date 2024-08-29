from langchain_groq import ChatGroq
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
import os 
import re 
import json
load_dotenv()


llm = ChatGroq(model_name="llama3-70b-8192", temperature=0,api_key= os.getenv('GROQ_API_KEY2')) ## Replace to real LLMs (Cohere / Groq / OpenAI)



examples = [{
  "context": "John là một sinh viên tại Đại học XYZ. Anh ấy đang theo đuổi bằng cấp về Khoa học Máy tính. Anh ấy đang đăng ký nhiều khóa học trong học kỳ này, bao gồm Cấu trúc Dữ liệu, Thuật toán, và Quản lý Cơ sở dữ liệu. John là một sinh viên chăm chỉ và dành nhiều thời gian để học và hoàn thành bài tập. Anh ấy thường ở lại muộn trong thư viện để làm việc trên các dự án của mình.",
  "statements": """
    John đang học ngành Sinh học.,
    John đang tham gia khóa học về Trí tuệ Nhân tạo.,
    John là một sinh viên chăm chỉ.,
    John có một công việc bán thời gian.,
  """,
  "answer": """
    statement1: John đang học ngành Sinh học,
    reason: Ngành học của John được đề cập rõ ràng là Khoa học Máy tính. Không có thông tin nào cho thấy anh ấy đang học ngành Sinh học.,
    verdict: 0,

    statement2: John đang tham gia khóa học về Trí tuệ Nhân tạo.,
    reason: Bối cảnh đề cập đến các khóa học mà John đang tham gia, và Trí tuệ Nhân tạo không được nhắc đến. Vì vậy, không thể suy ra rằng John đang tham gia khóa học về AI.,
    verdict: 0,

    statement3: John là một sinh viên chăm chỉ.,
    reason: Bối cảnh nêu rõ rằng anh ấy dành nhiều thời gian để học và hoàn thành bài tập. Ngoài ra, còn đề cập rằng anh ấy thường ở lại muộn trong thư viện để làm việc trên các dự án của mình, điều này ngụ ý sự chăm chỉ.,
    verdict: 1,

    statement4: John có một công việc bán thời gian.,
    reason: Không có thông tin nào được đưa ra trong bối cảnh về việc John có một công việc bán thời gian.,
    verdict: 0,
  """
}]



example_template = """
Examples :\n
<Input>
context: {context}
statements: {statements}
</Input>\n\n
<Output>
answer: {answer}
</Output>
"""

example_prompt = PromptTemplate(
    input_variables=["context", "statements"],
    template=example_template
)

# Định nghĩa prompt chính
prefix = "Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context. Format the outputs in JSON. Below is an example.Respone with exactly the same structure with examples. Rely in Vietnamese"

suffix = """
context: {context}
statements: {statements}

answer:
"""



few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["context", "statements"],
    example_separator="\n\n"
)

def convert_str_to_dict(data_str):
    # Initialize the dictionary
    data_dict = {"answer": []}
    
    # Use regex to match each statement block
    pattern = re.compile(
        r'statement(?P<index>\d+): (?P<statement>.+?)\.,\n'
        r'reason: (?P<reason>.+?)\.,\n'
        r'verdict: (?P<verdict>\d+),',
        re.DOTALL
    )
    
    # Find all matches in the input string
    matches = pattern.finditer(data_str)
    
    for match in matches:
        statement = match.group('statement').strip()
        reason = match.group('reason').strip()
        verdict = int(match.group('verdict').strip())
        
        # Append to the list in dictionary
        data_dict["answer"].append({
            "statement": statement,
            "reason": reason,
            "verdict": verdict
        })
    
    return data_dict

def calculate_faithfulness_score(data: dict, file_json: str): 

    prompt = few_shot_prompt_template.format(
        context=data["context"],
        statements= data["statements"],
    )
    out=llm.invoke(prompt)
    data_dict = convert_str_to_dict(out.content)
    # Save the dictionary to a JSON file
    with open(file_json, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)
    print(f"Data has been successfully converted and saved to {file_json}")

if __name__ == "__main__":
    new_example =   {
    "context": "Trong Thông tư này các từ ngữ dưới đây được hiểu như sau:\n1. Định mức kinh tế - kỹ thuật (sau đây gọi tắt là định mức): Là mức hao phí cần thiết về lao động về nguyên, nhiên vật liệu, máy móc thiết bị, dụng cụ và phương tiện để hoàn thành một đơn vị sản phẩm (hoặc một khối lượng công việc nhất định), trong một điều kiện cụ thể của các hoạt động điều tra cơ bản trong các lĩnh vực thuộc phạm vi quản lý nhà nước của Bộ.\n2. Dụng cụ là loại tài sản không đủ tiêu chuẩn về tài sản cố định theo quy định hiện hành của nhà nước mà người lao động sử dụng để tác động, biến đổi vật liệu thành sản phẩm (kìm, búa, cờ lê, quần áo bảo hộ và các dụng cụ khác tương tự).\n3. Vật liệu là đầu vào trong một quá trình sản xuất hoặc thực hiện nhiệm vụ chuyên môn cụ thể, được người lao động tác động, biến đổi hoàn toàn để thành sản phẩm theo yêu cầu đặt ra.\n4. Máy móc thiết bị là công cụ lao động thuộc tài sản cố định hữu hình và tài sản cố định vô hình theo tiêu chuẩn quy định hiện hành của nhà nước về tài sản cố định (không bao gồm nhà xưởng và quyền sử dụng đất) mà người lao động sử dụng để tác động, biến đổi vật liệu thành sản phẩm.",
    "statements": [
        "Dụng cụ trong ngành tài nguyên môi trường là loại tài sản không đủ tiêu chuẩn về tài sản cố định theo quy định hiện hành của nhà nước.",
        "Dụng cụ trong ngành tài nguyên môi trường được người lao động sử dụng để tác động, biến đổi vật liệu thành sản phẩm.",
        "Ví dụ của dụng cụ trong ngành tài nguyên môi trường bao gồm kìm, búa, cờ lê, quần áo bảo hộ và các dụng cụ khác tương tự.",
        "Định mức kinh tế - kỹ thuật là mức hao phí cần thiết về lao động, nguyên, nhiên vật liệu, máy móc thiết bị, dụng cụ và phương tiện để hoàn thành một đơn vị sản phẩm trong một điều kiện cụ thể của các hoạt động điều tra cơ bản trong các lĩnh vực thuộc phạm vi quản lý nhà nước của Bộ.",
        "Định mức kinh tế - kỹ thuật được gọi tắt là định mức.",
        "Dụng cụ là loại tài sản không đủ tiêu chuẩn về tài sản cố định theo quy định hiện hành của nhà nước mà người lao động sử dụng để tác động, biến đổi vật liệu thành sản phẩm.",
        "Ví dụ của dụng cụ bao gồm kìm, búa, cờ lê, quần áo bảo hộ và các dụng cụ khác tương tự.",
        "Vật liệu là đầu vào trong một quá trình sản xuất hoặc thực hiện nhiệm vụ chuyên môn cụ thể được người lao động tác động, biến đổi hoàn toàn để thành sản phẩm theo yêu cầu đặt ra.",
        "Máy móc thiết bị là công cụ lao động thuộc tài sản cố định hữu hình và tài sản cố định vô hình theo tiêu chuẩn quy định hiện hành của nhà nước về tài sản cố định.",
        "Máy móc thiết bị không bao gồm nhà xưởng và quyền sử dụng đất.",
        "Máy móc thiết bị được người lao động sử dụng để tác động, biến đổi vật liệu thành sản phẩm.,"
    ]
}

    file_json = 'final_output.json'
    calculate_faithfulness_score(new_example, file_json)


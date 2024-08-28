from langchain_core.prompts import PromptTemplate
RAG_PROMPT_TEMPLATE = """
<|system|>
Using the information contained in the context,
give a comprehensive answer to the question in Vietnamese language.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer. Say "tôi không biết" instead </s>
<|user|>
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
</s>
<|assistant|>
"""
accurate_rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
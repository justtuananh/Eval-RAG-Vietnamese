from langchain_core.prompts import PromptTemplate
RAG_PROMPT_TEMPLATE = """
<|system|>
Read the context below carefully and use the information to answer the question in Vietnamese. 

- Provide a comprehensive but concise response, directly related to the question.
- Include the number of the source document if your answer is based on specific information from the context.
- If you cannot deduce the answer from the context, respond with "tôi không biết".

Follow these instructions carefully to ensure clarity and accuracy in your response.
</s>
<|user|>
Context:
{context}
---
Now, answer the following question:

Question: {question}
</s>
<|assistant|>

"""
accurate_rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
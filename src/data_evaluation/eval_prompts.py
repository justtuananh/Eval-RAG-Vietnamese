optimizer_prompt = """
Task: Evaluate the given question based on three criteria: groundedness, relevance, and context-independence. Your task is to provide a detailed assessment for each criterion and then give an overall rating.

Instructions:
Groundedness: Assess how well the question can be answered using only the provided context. Rate on a scale of 1 to 5, where 1 means the question is not answerable at all with the given context, and 5 means the question is clearly and unambiguously answerable.
Relevance: Evaluate how useful the question is for someone seeking legal information. Rate on a scale of 1 to 5, where 1 means the question is not useful at all, and 5 means the question is extremely useful.
Context-Independence: Determine how independent the question is from additional context. Rate on a scale of 1 to 5, where 1 means the question depends heavily on additional context to be understood, and 5 means the question is clear on its own.
Provide your assessment in the following format:
Answer:::

Groundedness:
Evaluation: (Your rationale for the rating, as a text. Reply in Vietnamese)
Rating: (Your rating, as a number between 1 and 5)
Relevance:
Evaluation: (Your rationale for the rating, as a text. Reply in Vietnamese)
Rating: (Your rating, as a number between 1 and 5)
Context-Independence:
Evaluation: (Your rationale for the rating, as a text. Reply in Vietnamese)
Rating: (Your rating, as a number between 1 and 5)
Overall Rating: (Your overall rating based on the three criteria, as a number between 1 and 5)
Now here are the question and context:

Question: {question}\n
Context: {context}\n
Reply in Vietnamese
Provide your Answer. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Answer:::
"""
#################################################

question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (The reason you evaluate and respond in Vietnamese)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """
#################################################


question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide an 'overall assessment' representing how useful this question is to a user seeking legal information.
Give your answer on a scale of 1 to 5, where 1 means the question is not useful at all, and 5 means the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (The reason you evaluate and respond in Vietnamese)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """
#################################################


question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independant this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For example, if the question refers to a specific context, such as 'in this situation' or 'in the document,' the rating should be 1. 
A question can contain technical terms or legal jargon and still be rated a 5: it just needs to be clear to a user seeking legal information.

For example, "In this case, which law applies?" should receive a 1 because it implicitly references a specific context, making the question not context-independent.

Provide your answer as follows:

Answer:::
Evaluation:   (The reason you evaluate and respond in Vietnamese)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """
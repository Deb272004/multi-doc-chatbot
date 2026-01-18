from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_question_prompt = ChatPromptTemplate.from_messages([
   (
       "system",(
           "given a conversation history and the most recent user query, rewrite the query as a standalone "
           "that makes sense without relying on the previous context. Do not provide an answer-only reformulate the"
           "question if neccessary otherwise return unchanged"
       )
   ) ,
   MessagesPlaceholder("chat_history"),
   (
       "human","{input}"
   )
])

context_qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",(
            "you are an assistant designed to answer question using the provided context. Rely on the retrieved"
            "information to form your answer. if the answer is not found in the context, response with I don't have context about it"
            "keep your answer consise and no longer than three sentences, until or unless user say so to.\n\n{context}"
        )
    ),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])

PROMPT_REGISTRY = {
    "contextualize_question":contextualize_question_prompt,
    "context_qa":context_qa_prompt
}
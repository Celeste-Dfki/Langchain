import os

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig, Conversation,LlamaForSequenceClassification
from langchain.llms import HuggingFacePipeline, HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import torch
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from kor import create_extraction_chain, Object, Text, Number
from langchain.chains import SimpleSequentialChain, SequentialChain
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from transformers import AutoTokenizer, AutoModelForTokenClassification
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import Optional

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

gc.collect()
torch.cuda.empty_cache()

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] ="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"

#"tiiuae/falcon-7b"
#"Babelscape/wikineural-multilingual-ner"
model_id="meta-llama/Llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

pipe = pipeline(
    "text-generation", #ner geht nicht
    model=model_id,
    tokenizer=tokenizer,
    max_new_tokens=15,
    model_kwargs={"temperature":0, "max_length": 64}
)

llm = HuggingFacePipeline(pipeline=pipe)




template = """You are a chatbot having a conversation with a human. Answer with only one sentence.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
"""""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(
        session_id, url="redis://localhost:6379"
    ),
    input_messages_key="question",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "foo"}}

chain_with_history.invoke({"question": "Hi! I'm bob"}, config=config)

chain_with_history.invoke({"question": "Whats my name"}, config=config)

"""""
message_history = RedisChatMessageHistory(url="redis://localhost:6379/0", ttl=600, session_id="my-session")
message_history.add_user_message("hi!")
print(message_history.messages)


memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

print(llm_chain.predict(human_input="Hi there, my name is Celeste and I am 21 years old!"))
print(llm_chain.predict(human_input="I'm doing well! Just having a conversation with an AI."))
print(llm_chain.predict(human_input="How old am I? And what is my name?"))

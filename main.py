import os
from metal_sdk.metal import Metal
from langchain import OpenAI, LLMMathChain
from langchain.chains import PALChain
from langchain.retrievers import MetalRetriever
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import PromptTemplate

metalk = ""

API_KEY = metalk
CLIENT_ID = "ci_BdyHLsOqGkpDe8Woa5hl4KUb0rneGCKW7IesoI2Lx2o="

chosen_topic = "physics"

if chosen_topic == "physics":
  INDEX_ID = "64bf23b4d93c4b67d81197ce"
elif chosen_topic == "further-maths":
  INDEX_ID = "64bf1500d93c4b67d8119775"

openaik = ""
wolframk = ""

os.environ['OPENAI_API_KEY'] = openaik
os.environ['WOLFRAM_ALPHA_APPID'] = wolframk

metal = Metal(API_KEY, CLIENT_ID, INDEX_ID)
llm=OpenAI(temperature=1)
retriever = MetalRetriever(metal, params={"limit": 2})

qa_chain = RetrievalQA.from_chain_type(
  llm=OpenAI(temperature=1),
  chain_type="stuff",
  retriever=retriever,
)

template = """\
You are an AI which creates practise exam questions for students.
Your job is to generate a math question based on {topic}.
Ensure this question makes mathematical sense if its based on mathematics.
Only tell me the question and ensure it's ledgible by a computer
as I will use another AI to answer the question.
On a scale of 1 to 10 on how hard I want this question to be, where 10 is the hardest, I want this to be a 9/10.
Do not display the number of marks the question is worth.
Do not ask questions which refrence an image or figure. This is important.
I want you to generate questions with multiple parts, seperate these using parts i, ii, iii etc.
Only refrence topics which you are being taught via Metal.
"""

prompt = PromptTemplate.from_template(template)
response = qa_chain(prompt.format(topic="particles"))
result = response["result"]

print(f"Question: {result}")

tools = load_tools(['wolfram-alpha'], llm=llm)
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory, verbose=False)

fresult = ""

if chosen_topic == "further-maths":
  fresult = agent.run(f"What are the answers to {result}? If you for any reason cannot answer the question reply with 'Answer couldn't be computed'. Display answers seperately to each question.")
else:
  None

print(f"Answer: {fresult}")

# If physics, no serpapi

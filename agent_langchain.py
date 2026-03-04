import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. SETUP
load_dotenv()

# Connect to the Pinecone Index you created
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = PineconeVectorStore(
    index_name="emoji-toxicity",
    embedding=embeddings,
    pinecone_api_key=os.environ["PINECONE_API_KEY"]
)

# 2. BUILD THE RETRIEVER
# This automatically handles vectorizing the query and finding the best match
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1} # Only get the top 1 definition
)

# 3. DEFINE THE SYSTEM 2 BRAIN (The Prompt)
# LangChain uses templates so you can easily swap inputs
system_prompt = """
You are a Content Safety Expert. Analyze the relationship between the Parent Post and User Comment.

[RETRIEVED KNOWLEDGE]
{context}

[TASK]
1. Determine if the emoji in the comment is being used as a DOG WHISTLE (Slang) or LITERALLY (Benign).
2. Check the Parent Post context. 
   - If Parent Post contains "Toxic Context Triggers" found in the knowledge -> High Risk.
   - If Parent Post contains "Benign Signals" -> Low Risk.

Output strictly JSON:
{{
  "verdict": "SAFE" or "TOXIC",
  "reasoning": "brief explanation"
}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Parent Post: {parent_post}\nUser Comment: {input}"),
])

# 4. CREATE THE CHAIN
# This connects: Retriever -> Prompt -> LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"],
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. THE AGENT FUNCTION
def analyze_with_langchain(parent, comment):
    print(f"🔎 Analyzing: {comment}...")
    
    # Run the chain
    response = rag_chain.invoke({
        "input": comment,
        "parent_post": parent
    })
    
    # LangChain returns the answer in 'answer' key
    # It automatically found the definition and injected it into '{context}'
    return response["answer"]

# --- TEST IT ---
if __name__ == "__main__":
    result = analyze_with_langchain(
        parent="Check out my exclusive content!",
        comment="She is a 🌽 star"
    )
    print(result)
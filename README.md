# Guide to Building a RAG Pipeline with LLMs

## Introduction

This guide will walk you through the process of setting up a **Retrieval-Augmented Generation (RAG) pipeline** for working with **Large Language Models (LLMs)**. RAG is a technique where a model retrieves relevant information from an external knowledge base before generating responses. This can significantly improve the quality and relevance of responses in tasks like **question answering, document summarization, and chatbot development**.

The guide is specifically designed for the **NLP module in ProBattle** and will cover all the necessary steps, including setting up the required accounts, configuring vector databases, integrating LLMs, and optimizing the pipeline.

## Accounts You'll Need

Before starting, you'll need to create accounts on the following platforms:

- **[Hugging Face](https://huggingface.co/)** – Provides access to various pre-trained models for natural language processing (NLP).
- **[LangChain](https://python.langchain.com/)** – The primary library for building a RAG pipeline, enabling seamless integration of LLMs with vector databases.
- **[LangSmith](https://smith.langchain.com/)** – A logging and monitoring tool for LangChain workflows, helping debug and optimize the pipeline.
- **[Pinecone](https://www.pinecone.io/)** – A vector database for storing and querying embeddings, essential for efficient retrieval in RAG pipelines.
- **[Groq](https://groq.com/)** – A powerful LLM hosting platform that provides high-performance inference capabilities.
- **[Nomic](https://www.nomic.ai/)** – Provides an embedding model for converting text into vectors for efficient search and retrieval.
- **[Travily](https://www.travily.ai/)** – A web search API that enhances retrieval by fetching real-time information from the internet.

## **Setting environment variables**
Make a `.env` file in the root directory of your project and add the following environment variables:

```bash
HUGGINGFACE_API_KEY=
ATLAS_API_KEY=
PINECONE_API_KEY=
GROQ_API_KEY=
TAVILY_API_KEY=


LANGSMITH_TRACING=
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT
OPENAI_API_KEY=
```

---

## Setting Up the Components

Use the following sections to set up each component of the RAG pipeline. The follwing snippets contain code for running the SDKs independently.

Framework like Langchain have their own way of integrating these SDKs. You'll still need the original SDKs when using them in Lanchain. 

Note: Though this rarely happens, Langchain *does* have a tendency of having outdated libraries or extensions, often because the SDKs of the other libaries are frequently updated. To avoid this, always try updating the langchain packaged you are using.
```bash
pip install langchain --upgrade

# This only updates langchain, you'll often need to upgrade the other packages, e.g. lanchain-pinecone, langchain-groq, etc.
```

Using Langchain greatly simplifies the process of defining the RAG pipeline, but you're free to use the SDKs independently if you prefer.


## **Hugging Face: Installing and Using Pre-Trained Models**

For the online documentation, refer to the following link:
**[Hugging Face Documentation](https://huggingface.co/docs/api-inference/getting-started)**

**Note**: This guide uses the `huggingface-hub` library, which is uses the inference API to access the models, rather than the transformers library which often downloads the model locally.

Start by installing the Hugging Face Inference API using pip:
```bash
pip install huggingface-hub
```

Then you can use the Hugging Face Inference API to generate responses using pre-trained models:
```python
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
client = InferenceClient(
	# provider="together", # optional, default is huggingface's own inference API
	api_key = os.getenv("HUGGINGFACE_API_KEY")
)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message.content)
```


## **Pinecone: Instantiation and Querying Vectors**
For the online documentation, refer to the following link:
**[Pinecone Documentation](https://docs.pinecone.io/reference/api/2024-10/control-plane/list_indexes)** 

Pinecone is a vector database. It allows you to store vector embeddings and query them to find the k-most-similar vectors. Here's a quick guide to get you started with Pinecone:

Install the Pincone Python sdk using pip:
```bash
pip install "pinecone[grpc]"
```

```python
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os
load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Pinecone Indices: ", pc.list_indices())

upsert_response = index.upsert(
    vectors=[
        {
            "id": "vec1", # unique string identifier for the vector, must be provided
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # put the embedding vector here
            "metadata": {  # put the actual document's text here
                "text": "This is a sample document."
                "genre" : "documentary" # other optional metadata
            }
        },
    ],
    namespace="example-namespace" # optional, defaults to "default"
)

# Finding similar vectors
index.query(
    namespace="example-namespace",
    vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], # put the query vector here
    filter={ # optional, to filter the results based on metadata
        "genre": {"$eq": "documentary"}
    },
    top_k=3,
    include_values=True # optional, to include the vector values in the response
)
```
## **Groq: LLM inference (Llama 3.3 70B)**

For the online documentation, refer to the following link:
**[Groq Documentation](https://console.groq.com/docs/api-reference#chat-create)**

Start by installing the Groq Python SDK using pip:
```bash
pip install groq
```

Then you can use the Groq Python SDK to generate responses using the Llama 3.3 70B model:
```python
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "you are a helpful assistant for question answering"
        },
        {
            "role": "user",
            "content": "Hi, how are you?",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)
```

## **Nomic: Using the Embedding Model**

For the online documentation, refer to the following link:
**[Nomic Documentation](https://docs.nomic.ai/reference/api/embed-text-v-1-embedding-text-post)**

Nomic Embedding Models allow you to convert text into vectors for efficient search and retrieval. Here's a quick guide to get you started with Nomic:

Install the Nomic Python SDK using pip:
```bash
pip install nomic
```
Then login using the CLI:
```bash
nomic login nk-blahblahblah
```
Then you can use the Nomic Python SDK to embed text:
```python

from nomic import embed
import numpy as np

output = embed.text(
    texts=['The text you want to embed.'],
    model='nomic-embed-text-v1.5',
    task_type='search_document',
)

embeddings = np.array(output['embeddings'])
print(embeddings[0].shape)  # prints: (768,)
```


## **Travily: Implementing Web Search for Real-Time Retrieval**

For the online documentation, refer to the following link:
**[Travily Documentation](https://docs.travily.ai/)**

Start by installing the Travily Python SDK using pip:
```bash
pip install travily-python
```

Then you can use the Travily Python SDK to fetch real-time information:
```python
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1. Instantiating your TavilyClient
travily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Step 2. Executing the search request
response = travily_client.search("Who is Leo Messi?", max_results=10)

# Step 3. Printing the search results
for result in response["results"]:
    print(result["url"])
```

## **LangChain: Integrating LLMs and Vector Search**

LangChain is a powerful library that simplifies the integration of LLMs with vector databases. Here's how you can use LangChain to build a RAG pipeline:

Install the LangChain Python SDK using pip:
```bash
pip install langchain
```

Then you can use LangChain to build a RAG pipeline:
```python
from dotenv import load_dotenv
import os

load_dotenv(override=True) # Load environment variables from .env file, override any existing variables

# Making a Langchain Embeddings Object using Nomic

from langchain_nomic import NomicEmbeddings

embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

# Making a Pinecone Vector Store Object

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "nlp-module"  # change if desired
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="nlp-module-index")

# Making a Retriever Object (Allows you to find similar documents in your Pinecone index, given a query)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 15, "score_threshold": 0.5},
)

# Making a ChatGroq Object (This is the LLM model that will generate responses)

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192", stop_sequences= None, temperature=0)

# Function to format the retrieved documents, gotten from the retriever

def format_docs(docs):
    print("docs:", docs)
    print()
    return "\n\n".join(doc.page_content for doc in docs)


# Making a custon prompt which had two variables, "context" and question

# Note:This prompt_template expects a dictionary/JSON with the keys "context" and "question" as input

from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context", "question"],
            template=( # The constructed prompt from the variables
                "You are an assistant for question-answering tasks. Use the following "
                "pieces of retrieved context to answer the question. If you don't know "
                "the answer, just say that you don't know. Use three sentences maximum "
                "and keep the answer concise.\n\n"
                
                "Question: {question}\n"
                "Context: {context}\n"
                "Answer:"
            )
            
        )
    )
])

# A simple function that logs the input and returns it

def logger(input):
    print(input)
    return input


# A chain with the modified prompt

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
 
rag_chain = (
    # The starting input to all these is passed in the invoke function
    # e.g rag_chain.invoke("Tell me about the paper: Attention is all you Need")
    
    # The first runnable in the chain:
    {
        "context": retriever | format_docs | logger,
        "question": RunnablePassthrough()
    }
    # It makes a dictionary using the input
    # the input is passed through the retriever, then the format_docs function, then the logger function
    # the retriever finds similar documents in the Pinecone index, and the format_docs function formats them
    # the logger function logs the input and returns it, 
    # RunnablePassthrough is a simple function that returns the input,
    # which means the 
    
    # The second runnable constructs the prompt using the previous runnables' output
    # The previous runnables' output is is a dictionary: {"context": ..., "question": ...}
    | prompt_template
    # This makes a prompt using the context and question from the previous runnables' output
    # The prompt is just a large string that is passed to the next runnable
    
    # The third runnable is the LLM model that generates the response
    | llm
    # The LLM model generates a response using the prompt
    # It's lke passing something to the ChatGPT model and getting a response
    
    # The fourth runnable is the output parser that converts the output to a string
    | StrOutputParser() 
    # The llm's output is a dictionary with several fields
    # The output parser takes the response.content field and returns it as a string
)

# The chain simply looks likes this:
rag_chain = (
    {
        "context": retriever | format_docs | logger,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

respnse = rag_chain.invoke("Tell me about the paper: Attention is all you Need")
```


## **LangSmith: Logging the Pipeline (Optional but highly recommended)**

Langsmith is a logging and monitoring tool for **LangChain** workflows. It helps you debug and optimize the pipeline by providing detailed logs and performance metrics. Here's how you can set it up:

### **Step 1**: Create an account on [Langsmith](https://smith.langchain.com/).


### **Step 2**: Create a new project in Langsmith.

<img width="1431" alt="langsmith_home" src="https://github.com/user-attachments/assets/e24139ee-0904-4f61-9e7d-804b6d768cf8" />

### **Step 3**: Copy the `.env` variables and paste them in your .env file.

<img width="1437" alt="langsmith_new_proj" src="https://github.com/user-attachments/assets/d9d7c165-0d21-46fe-a70b-d0a15117052b" />

### **Step 4**: Go back to Tracing Projects, and run an invoke call in your code to start logging. Click on the names column to refresh the list. (Refresh the page if you don't see your project)

```python
from dotenv import load_dotenv
import os

load_dotenv(override=True) 

from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings

groq = ChatGroq(model="llama3-8b-8192", stop_sequences=None, temperature=0)
response = groq.invoke("tell me all about Facebook")
print(response.content)
```

### **Step 5**: Click on the project to see its logs.

<img width="994" alt="langsmith_select_proj" src="https://github.com/user-attachments/assets/ce61662d-c153-4550-944f-83925b1cbf15" />

### **Step 6**: Click on a log to see the details.

<img width="1431" alt="langsmith_see_log" src="https://github.com/user-attachments/assets/5170416d-22c5-4235-96ff-4273cd27ee5f" />

---

## Streamlit: Making a front-end for the pipeline

Streamlit is useful for making really simple front-ends for AI/ML apps. Here's a simple example of how you can use Streamlit to make a front-end for the RAG pipeline:

Install Streamlit using pip:
```bash
pip install streamlit
```

Then make a file, e.g. `app.py`, and add the following code:

```python
import streamlit as st
from ab import rag_chain, retriever, format_docs  # Ensure RAG chain supports invocation

st.set_page_config(page_title="RAG Chatbot")

st.title("RAG Chatbot")

with st.form("chat_form"):
    user_input = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Generate Response")

if submitted and user_input:
    st.subheader("Retrieved Documents:") # Show the retrieved documents
    st.write((retriever | format_docs) .invoke(user_input))
    
    st.subheader("Response:") # Show the generated response
    with st.spinner("Generating response..."):  # Show a spinner while generating the response
        response = rag_chain.invoke(user_input)
    
    st.write(response) 

```

Run the Streamlit app using the following command:
```bash
streamlit run app.py
```

This will start a local server and open the Streamlit app in your browser. You can now interact with the RAG pipeline through the front-end.


---
# Guide to Setting Up Ngrok with Static Domains

## Introduction

Ngrok is a tunneling service that creates secure tunnels from public URLs to your localhost. It allows you to expose your local development server to the internet using a static domain name.

## Creating an Account

1. Visit [ngrok.com](https://ngrok.com) and click "Sign Up"
2. Complete the registration process
3. After signing in, navigate to the [dashboard](https://dashboard.ngrok.com)
4. Find your authtoken in the "Getting Started" section

## Installation

### Method 1: Using Package Managers

**On Mac (using Homebrew):**
```bash
brew install ngrok
```

**On Windows (using Chocolatey):**
```bash
choco install ngrok
```

### Method 2: Direct Download

1. Go to [ngrok.com/download](https://ngrok.com/download)
2. Download the appropriate version for your operating system
3. Extract the downloaded file
4. (Optional) Add ngrok to your system PATH for easier access

## Configuration

1. Open your terminal/command prompt
2. Add your authtoken (replace YOUR_AUTH_TOKEN with your actual token):
```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

## Running Ngrok with Static Domain

1. First, get your static domain from the ngrok dashboard under "Cloud Edge > Domains"
2. Use the following command format to start ngrok with your static domain:
```bash
ngrok http --domain=your-static-domain.ngrok-free.app PORT_NUMBER
```

For example, to expose a local server running on port 8000:
```bash
ngrok http --domain=desired-albacore-commonly.ngrok-free.app 8000
```

You can now serve your streamlit app on a public URL. Start your streamlit server on the specified port and access it using the provided static domain.
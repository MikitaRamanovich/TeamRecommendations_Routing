import os
import zipfile
import docx
import pandas as pd
import json
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task

from dotenv import load_dotenv
load_dotenv()
open_ai_api_key = os.getenv("OPENAI_API_KEY")
lang_smith_tracing = os.getenv("LANGSMITH_TRACING")
lang_smith_endpoint = os.getenv("LANGSMITH_ENDPOINT")
lang_smith_api_key = os.getenv("LANGSMITH_API_KEY")
lang_smith_project_name = os.getenv("LANGSMITH_PROJECT")

# --- Environment Setup ---
os.environ["OPENAI_API_KEY"] = open_ai_api_key
os.environ["LANGSMITH_TRACING"] = lang_smith_tracing
os.environ["LANGSMITH_ENDPOINT"] = lang_smith_endpoint
os.environ["LANGSMITH_API_KEY"] = lang_smith_api_key
os.environ["LANGSMITH_PROJECT"] = lang_smith_project_name

# --- File Extraction Utilities ---
def extract_text_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_csv(index=False)

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return extract_text_docx(file_path)
    elif ext == ".pdf":
        return extract_text_pdf(file_path)
    elif ext in [".xlsx", ".xls"]:
        return extract_text_excel(file_path)
    else:
        return ""

def process_folder(folder_path):
    """
    Traverses the folder (and subdirectories). If a zip archive is found, it extracts it and processes its content.
    """
    all_texts = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(".zip"):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    temp_folder = os.path.join(root, "temp_extracted")
                    os.makedirs(temp_folder, exist_ok=True)
                    zip_ref.extractall(temp_folder)
                    extracted_text = process_folder(temp_folder)
                    all_texts.append(extracted_text)
            else:
                text = extract_text_from_file(file_path)
                if text:
                    print(f"Extracted text from {file_path} (length {len(text)} characters)")
                    all_texts.append(text)
    return "\n".join(all_texts)

def split_text(text, chunk_size=1000, chunk_overlap=500):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings()  # Or choose another embedding provider
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def retrieve_relevant_chunks(vectorstore, query, k=5):
    return vectorstore.similarity_search(query, k=k)

# --- Routing Setup ---
from typing_extensions import Literal
from pydantic import BaseModel, Field

# Define schema for structured routing output.
class Route(BaseModel):
    step: Literal["summary", "recommendation", "combined"] = Field(
        None, description="The next step in the routing process"
    )

# Initialize LLM (you can switch between ChatOllama and ChatOpenAI as needed)
# llm = ChatOllama(model="phi4", temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)

def llm_call_router(context: str, question: str):
    """
    Use the routing LLM to decide which analysis is needed.
    """
    input_for_router = (
        f"Given the project description below:\n{context}\n\n"
        f"And the user question:\n{question}\n\n"
        "Decide whether you need to provide only a summary of the project (output 'summary'), "
        "only a team recommendation (output 'recommendation'), or both (output 'combined'). "
        "Return your answer as a JSON object with a single field 'step'."
    )
    decision = router.invoke([
        SystemMessage(content="Route the input to summary, recommendation, or combined based on the project's needs."),
        HumanMessage(content=input_for_router),
    ])
    return decision.step

# --- LLM Analysis Tasks using Routing ---
@task
def generate_summary(context: str, question: str):
    prompt = (
        "You are an expert project analyst. Based on the following project description:\n"
        f"{context}\n\n"
        "Provide a concise summary of the project. Answer the following question:\n"
        f"{question}\n\n"
        "Return the result in JSON format with key 'project_description'."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content

@task
def generate_recommendation(context: str, question: str):
    prompt = (
        "You are an expert project analyst. Based on the following project description:\n"
        f"{context}\n\n"
        "Provide a detailed team recommendation including roles, techstack, experience, and rationale. "
        f"Answer the following question:\n{question}\n\n"
        "Return the result in JSON format with key 'team_recommendation' as a list."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content

@task
def generate_combined(context: str, question: str):
    prompt = (
        "You are an expert project analyst. Analyze the following project description:\n"
        f"{context}\n\n"
        "Based on your analysis, provide both a concise summary of the project and a detailed team recommendation. "
        "For the team recommendation, include, for each recommendation, the following fields: "
        "'role', 'techstack' (as a list), 'experience', and 'rationale'.\n\n"
        "Return your answer as a JSON object exactly matching the following structure:\n\n"
        "{\n"
        '  "project_description": "<a concise summary of the project>",\n'
        '  "team_recommendation": [\n'
        "    {\n"
        '      "role": "<role name>",\n'
        '      "techstack": [<list of technologies>],\n'
        '      "experience": "<required experience>",\n'
        '      "rationale": "<reasoning behind the recommendation>"\n'
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        f"Answer the following question: {question}"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content

@task
def llm_routing_analysis_task(retrieved_chunks, user_question: str):
    # Combine the relevant document content.
    context = "\n".join([doc.page_content for doc in retrieved_chunks])
    # Determine the analysis route.
    route = llm_call_router(context, user_question)
    print(f"Routing decision: {route}")
    # Based on the route, call the appropriate analysis function.
    if route == "summary":
        result = generate_summary(context, user_question).result()
    elif route == "recommendation":
        result = generate_recommendation(context, user_question).result()
    elif route == "combined":
        result = generate_combined(context, user_question).result()
    else:
        result = json.dumps({"error": f"Unknown routing step: {route}"})
    
    # --- Clean up the LLM response ---
    response_text = result.strip()
    # Remove markdown formatting if present
    if response_text.startswith("```"):
        parts = response_text.split("```")
        if len(parts) >= 3:
            json_text = parts[1].strip()
            if json_text.lower().startswith("json"):
                json_text = json_text[4:].strip()
            response_text = json_text

    try:
        result_json = json.loads(response_text)
    except json.JSONDecodeError:
        result_json = {"error": "Invalid JSON response", "response": response_text}
    return result_json

# --- Workflow Tasks ---
@task
def extract_text_task(folder_path: str):
    return process_folder(folder_path)

@task
def split_text_task(text: str):
    return split_text(text)

@task
def vector_store_task(chunks):
    return build_vector_store(chunks)

@task
def retrieve_chunks_task(vectorstore, query: str):
    return retrieve_relevant_chunks(vectorstore, query, k=5)

@entrypoint()
def document_analysis_workflow(state: dict):
    folder_path = state["folder_path"]
    user_question = state["user_question"]
    # Step 1: Extract text from the folder.
    full_text = extract_text_task(folder_path).result()
    
    # Step 2: Split text into manageable chunks.
    chunks = split_text_task(full_text).result()
    
    # Step 3: Build a vector store from the chunks.
    vectorstore = vector_store_task(chunks).result()
    
    # Step 4: Retrieve the most relevant chunks based on the user question.
    retrieved_chunks = retrieve_chunks_task(vectorstore, user_question).result()
    
    # Step 5: Use routing to determine the appropriate analysis and generate a JSON response.
    result_json = llm_routing_analysis_task(retrieved_chunks, user_question).result()
    
    # Final output with clean JSON (without markdown wrappers)
    return {"result": result_json}

# --- Sample Invocation ---
if __name__ == "__main__":
    folder_path = "/Users/mikita/Projects/Agents/TeamRecommendations_Routing/Description/Zava"  
    user_question = (
        "Based on the project description extracted from the documents, "
        "please generate a JSON object with keys 'project_description' and 'team_recommendation'. "
        "The 'team_recommendation' should include the following fields for each recommendation: "
        "role, techstack, experience, and rationale."
    )
    input_state = {"folder_path": folder_path, "user_question": user_question}
    config = {"configurable": {"thread_id": "unique_thread_id"}}  # Provide a unique thread id

    # Invoke the workflow synchronously.
    final_result = document_analysis_workflow.invoke(input_state, config=config)
    
    # Define the output folder path.
    output_folder = "/Users/mikita/Projects/Agents/TeamRecommendations_Routing/Result"
    os.makedirs(output_folder, exist_ok=True)

    # Construct the file name based on the input folder's basename.
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_file = os.path.join(output_folder, folder_name + ".json")

    # Save the JSON result to the specified folder with pretty printing.
    with open(output_file, "w") as f:
        json.dump(final_result, f, indent=2)

    print(f"Saved JSON result to {output_file}")

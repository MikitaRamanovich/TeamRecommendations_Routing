# Document Analysis Workflow with LangGraph and LLM Routing

## Overview

This project is an end-to-end solution that automatically processes a folder of documents, extracts and organizes text, and then uses an LLM with a routing mechanism to analyze the content and produce a structured JSON output. The final JSON contains a concise project summary and a detailed team recommendation. The entire process is automated—from file extraction and text splitting to vector indexing and LLM analysis.

## Key Features

- **Multi-Format Document Extraction:**  
  Extracts text from DOCX, PDF, Excel, and ZIP archive files.

- **Text Chunking:**  
  Splits large texts into manageable chunks using a recursive text splitter.

- **Vector Store Construction:**  
  Converts text chunks into embeddings via OpenAI and indexes them using FAISS for efficient similarity search.

- **Contextual Retrieval:**  
  Retrieves the most relevant text chunks based on a user query through similarity search.

- **Routing-Based LLM Analysis:**  
  Uses a routing LLM to decide whether to generate a project summary, a team recommendation, or a combined response, based on the retrieved context and the query.

- **Structured JSON Response:**  
  Outputs a clean JSON object with keys like `"project_description"` and `"team_recommendation"`.

- **File Output:**  
  Saves the final JSON result to a specified output directory.

## Step-by-Step Workflow

### 1. Environment Setup

- **Configuration:**  
  The project uses a `.env` file to load necessary API keys and configuration variables (e.g., OpenAI API key, LangSmith tracing settings).

- **Dependencies:**  
  It imports LangGraph, LangChain, and other libraries to manage document processing, text embedding, and LLM interactions.

### 2. File Extraction

- **Folder Traversal:**  
  The workflow recursively traverses a given directory, locating all supported file types, including handling ZIP archives.

- **Type-Specific Extraction:**  
  Specialized functions extract text from each file type (DOCX, PDF, Excel), ensuring that the content is captured accurately.

### 3. Text Splitting

- **Chunking:**  
  The complete extracted text is divided into smaller, fixed-size chunks using the `RecursiveCharacterTextSplitter`. This makes it suitable for processing by the LLM.

### 4. Building the Vector Store

- **Embedding Generation:**  
  Each text chunk is transformed into an embedding using OpenAIEmbeddings.

- **Indexing with FAISS:**  
  The embeddings are stored in a FAISS vector index, enabling rapid similarity-based search.

### 5. Retrieving Relevant Chunks

- **Similarity Search:**  
  A user query is used to perform a similarity search on the FAISS vector store, retrieving the top relevant chunks that provide the necessary context.

### 6. LLM Analysis with Routing

- **Context Aggregation:**  
  The retrieved chunks are concatenated to form a comprehensive context.

- **Routing Decision:**  
  A specialized routing LLM examines the context and the user query, determining the appropriate analysis path:

  - **Summary:** Generate a concise project summary.
  - **Recommendation:** Provide a detailed team recommendation.
  - **Combined:** Produce both outputs together.

- **Prompt Execution:**  
  Based on the routing result, the corresponding LLM prompt is executed to generate the final output.

### 7. JSON Response Generation

- **Structured Output:**  
  The LLM returns its response in JSON format, which includes keys for `"project_description"` and `"team_recommendation"`.
- **Cleanup:**  
  Any markdown formatting (e.g., code fences) is stripped from the output, ensuring the JSON is clean and properly formatted.

### 8. Saving the Output

- **File Writing:**  
  The final JSON result is saved to a file in a designated output folder for later use.

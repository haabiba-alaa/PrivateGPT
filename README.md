# Chat with Multiple PDFs
This Streamlit web application allows users to upload multiple PDF documents, extract text from them, generate embeddings using Hugging Face models, and engage in a conversational AI chat using the OpenAI API.

## Features
Document Upload: Users can upload multiple PDF documents.
Text Extraction: Extracts text from uploaded PDF documents.
Text Chunking: Splits extracted text into manageable chunks.
Embedding Generation: Generates embeddings for text chunks using Hugging Face models.
Conversational AI: Engages users in a chat conversation using the OpenAI API.

## Installation
Clone the repository:

`git clone https://github.com/your-username/your-repo.git`

cd your-repo

## Install dependencies:
pip install -r requirements.txt

## Set up environment variables:
Create a .env file in the root directory of the project.

Add your API keys:

OPENAI_API_KEY=your_openai_api_key_here

HUGGINGFACE_API_KEY=your_huggingface_api_key_here

## Usage
Run the Streamlit app:
streamlit run app.py

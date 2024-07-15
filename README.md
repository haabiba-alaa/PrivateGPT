Chat with Multiple PDFs
This Streamlit web application allows users to upload multiple PDF documents, extract text from them, generate embeddings using Hugging Face models, and engage in a conversational AI chat using the OpenAI API.

Features
Document Upload: Users can upload multiple PDF documents.
Text Extraction: Extracts text from uploaded PDF documents.
Text Chunking: Splits extracted text into manageable chunks.
Embedding Generation: Generates embeddings for text chunks using Hugging Face models.
Conversational AI: Engages users in a chat conversation using the OpenAI API.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/your-repo.git
cd your-repo
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up environment variables:

Create a .env file in the root directory of the project.
Add your API keys:
makefile
Copy code
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Access the app in your web browser at http://localhost:8501.

Upload PDF documents using the sidebar.

Click on "Process" to extract text, generate embeddings, and set up the conversational AI.

Enter questions in the text input field to engage with the conversational AI.

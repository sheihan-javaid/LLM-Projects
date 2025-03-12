import streamlit as st
import pdfplumber
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import base64

# Directly setting the Hugging Face API key
api_key = "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"

# Function to extract text from a PDF file
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text += page.extract_text() or ""
                progress = (i + 1) / total_pages
                st.session_state.extraction_bar.progress(progress, text=f"Extracting text: {int(progress * 100)}%")
    except Exception as e:
        st.error(f"An error occurred while extracting text from the PDF: {e}")
    return text

# Function to display the PDF of a given file
def display_pdf(uploaded_file):
    uploaded_file.seek(0)  # Reset file pointer to start
    base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize LangChain LLM with HuggingFaceHub
@st.cache_resource
def initialize_llm():
    try:
        huggingface_llm = HuggingFaceHub(
            repo_id="facebook/bart-large-cnn",
            model_kwargs={"temperature": 0.5, "max_length": 130},
            huggingfacehub_api_token=api_key
        )
    except Exception as e:
        st.error(f"Failed to initialize OpenAI GPT model: {e}")
        return None
    
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in a concise manner:\n\n{text}\n\nSummary:"
    )
    return LLMChain(llm=huggingface_llm, prompt=prompt_template)

# Summarization function using LangChain
def summarize_text(text, max_length=1000):
    llm_chain = initialize_llm()
    if not llm_chain:
        return "Error: LLM not initialized"
    
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    summaries = []
    
    for i, chunk in enumerate(chunks):
        try:
            result = llm_chain.run({"text": chunk})
            summaries.append(result)
            progress = (i + 1) / len(chunks)
            st.session_state.summarization_bar.progress(progress, text=f"Summarizing: {int(progress * 100)}%")
        except Exception as e:
            st.error(f"An error occurred during text summarization of chunk {i+1}: {e}")
    
    return "\n\n".join(summaries)

# Main Streamlit application
st.set_page_config(layout="wide", page_title="SmartPDF Summarizer", page_icon="📄")

def main():
    st.title("📄 DocuDigest:AI PDF Summarizer")
    st.write("Upload a PDF file and get a concise summary using OpenAI's GPT model.")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        file_size = uploaded_file.size
        max_size = 10 * 1024 * 1024  # 10 MB

        if file_size > max_size:
            st.error(f"File size exceeds the limit of 10 MB. Your file is {file_size / 1024 / 1024:.2f} MB.")
        else:
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.subheader("Uploaded File")
                display_pdf(uploaded_file)
                
                # Reset file pointer after reading for display
                uploaded_file.seek(0)

            with col2:
                st.subheader("Summary")
                if st.button("Summarize", key="summarize_button"):
                    st.session_state.extraction_bar = st.progress(0)
                    st.session_state.summarization_bar = st.progress(0)
                    
                    with st.spinner("Processing PDF..."):
                        input_text = extract_text_from_pdf(uploaded_file)
                    
                    if input_text:
                        with st.spinner("Generating summary using OpenAI GPT..."):
                            summary = summarize_text(input_text)
                        st.success("Summarization Complete")
                        st.markdown(summary)
                    else:
                        st.warning("No text found in the PDF.")

    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses OpenAI's GPT model to summarize PDF documents. "
        "Upload a PDF file and click 'Summarize' to get started."
    )

if __name__ == "__main__":
    main()
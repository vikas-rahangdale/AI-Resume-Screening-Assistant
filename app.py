import streamlit as st
from dotenv import load_dotenv
import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file. Please add it before running.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(model="gpt-6-luna")

# Prompt Template
template = """
You are an AI recruiter assistant.
Compare the candidate resume with the given Job Description.

Job Description:
{job_description}

Resume Content:
{resume_content}

Provide structured output in JSON:
{{
  "Match Score": "0-100",
  "Matching Skills": [],
  "Missing Skills": [],
  "Candidate Summary": "",
  "Strengths": [],
  "Weaknesses": [],
    "Recommended Improvements": [],
  "Hiring Recommendation": ""
}}
"""

prompt = PromptTemplate(
    input_variables=["job_description", "resume_content"],
    template=template
)
chain = prompt | llm

# Streamlit UI
st.set_page_config(page_title="AI Resume Screening Assistant", page_icon="🧑‍💼", layout="wide")
st.title("🧑‍💼 AI Resume Screening Assistant")
st.caption("Match resumes to a role and explore clear, structured candidate insights.")

with st.container(border=True):
    st.subheader("📝 Define the role")
    jd = st.text_area("Job description", placeholder="Paste the role, required skills, and qualifications...", height=180)
    uploaded_files = st.file_uploader("📄 Upload resumes (PDF)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        st.caption(f"✅ {len(uploaded_files)} resume{'s' if len(uploaded_files) != 1 else ''} ready to evaluate")

if st.button("🔎 Evaluate resumes", type="primary", use_container_width=True):
    if jd.strip() == "":
        st.warning("Please enter a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner("🤖 Reviewing resumes and comparing skills..."):
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        for uploaded_file in uploaded_files:
            with st.container(border=True):
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loader = PyPDFLoader(uploaded_file.name)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)

                vector_db = FAISS.from_documents(chunks, embeddings)
                retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

                resume_docs = retriever.invoke(jd)
                resume_text = "\n".join([doc.page_content for doc in resume_docs])

                response = chain.invoke({"job_description": jd, "resume_content": resume_text})
                response_text = response.content if hasattr(response, "content") else str(response)

                st.subheader(f"📋 {uploaded_file.name}")

                try:
                    result = json.loads(response_text)

                    # Match Score
                    st.metric(label="🎯 Match Score", value=result.get("Match Score", "N/A"))

                    # Skills
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("✅ Matching Skills")
                        for skill in result.get("Matching Skills", []):
                            st.write(f"• {skill}")
                    with col2:
                        st.subheader("🧩 Skills to Develop")
                        for skill in result.get("Missing Skills", []):
                            st.write(f"• {skill}")

                    # Candidate Summary
                    st.subheader("👤 Candidate Summary")
                    st.info(result.get("Candidate Summary", ""))

                    # Strengths & Weaknesses
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("💪 Strengths")
                        for s in result.get("Strengths", []):
                            st.write(f"✅ {s}")
                    with col2:
                        st.subheader("⚠️ Weaknesses")
                        for w in result.get("Weaknesses", []):
                            st.write(f"• {w}")

                    # Recommended Improvements
                    st.subheader("🛠️ Recommended Improvements")
                    improvements = result.get("Recommended Improvements", [])
                    if improvements:
                        for improvement in improvements:
                            st.write(f"• {improvement}")
                    else:
                        st.write("No specific improvements recommended.")

                    # Hiring Recommendation
                    st.subheader("🤝 Hiring Recommendation")
                    st.success(result.get("Hiring Recommendation", ""))

                except (json.JSONDecodeError, TypeError):
                    st.error("⚠️ Could not parse structured output. Showing raw response instead:")
                    st.write(response_text)

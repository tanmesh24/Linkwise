import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="LinkWise (OpenAI RAG)", layout="wide")
st.title("🔗 LinkWise: Source-Grounded Q&A over Web Links")

st.sidebar.title("Enter Article URLs (Static Pages Only)")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

INDEX_PATH = "faiss_index"

# ---------------- OpenAI Models ----------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

embeddings = OpenAIEmbeddings()

# ---------------- Process URLs ----------------
if process_url_clicked and urls:
    loader = UnstructuredURLLoader(
        urls=urls,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    data = loader.load()

    if len(data) == 0:
        st.error("❌ No content could be loaded from the URLs.")
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        docs = splitter.split_documents(data)

        st.success(f"✅ Loaded {len(docs)} text chunks")

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_PATH)

        st.success("✅ Vector store created and saved")

# ---------------- Question Answering ----------------
st.subheader("Ask a question")
query = st.text_input("Question")

if query and os.path.exists(INDEX_PATH):
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 🔹 Retrieve docs WITH distance scores
    docs_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=8
    )

    if len(docs_with_scores) == 0:
        st.warning("⚠️ No relevant context retrieved")
    else:
        # ---------------- Build Context ----------------
        context = "\n\n".join(
            doc.page_content for doc, _ in docs_with_scores
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a factual assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
        )

        response = llm.invoke(
            prompt.format(context=context, question=query)
        )

        # ---------------- Answer ----------------
        st.header("📌 Answer")
        st.write(response.content)

        # ---------------- Sources + Cosine Similarity ----------------
        st.subheader("🔗 Sources with Cosine Similarity")

        rows = []
        for i, (doc, distance) in enumerate(docs_with_scores, start=1):
            cosine_similarity = round(1 - distance, 3)

            rows.append({
                "Rank": i,
                "Source": doc.metadata.get("source", "Unknown"),
                "Cosine Similarity": cosine_similarity
            })

        df = pd.DataFrame(rows)

        # ---- Table View ----
        st.dataframe(df, use_container_width=True)

        # ---- Bar Chart ----
        st.subheader("📊 Relevance Visualization")
        st.bar_chart(
            data=df.set_index("Source")["Cosine Similarity"]
        )

        # ---- Optional Context Viewer ----
        with st.expander("🔍 View Retrieved Context"):
            for i, (doc, _) in enumerate(docs_with_scores, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content[:500] + "...")

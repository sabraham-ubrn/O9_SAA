from pathlib import Path
import fitz  # PyMuPDF
import torch
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import CLIPProcessor, CLIPModel
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from typing import List
from ollama_connect import OllamaRESTLLM
from langchain_community.vectorstores import Chroma


import os

# Text embedding model
text_embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Image embedding model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)


def extract_text_chunks(pdf_path: Path) -> List[Document]:
    from langchain.document_loaders import PyMuPDFLoader
    docs = PyMuPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def extract_images(pdf_path: Path, image_dir="images") -> List[Document]:
    os.makedirs(image_dir, exist_ok=True)
    image_docs = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_path = f"{image_dir}/page{page_num}_img{img_index}.png"
            pix.save(img_path)

            # Get image embedding
            image = Image.open(img_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = clip_model.get_image_features(**inputs)
            embedding = embedding.cpu().numpy()[0]

            # Wrap in a Document
            image_docs.append(Document(
                page_content="Image embedding from PDF page",
                metadata={"type": "image", "path": img_path, "vector": embedding.tolist()}
            ))
    return image_docs


def create_or_load_vector_store(pdf_path: Path, persist_directory="db") -> Chroma:
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        return Chroma(persist_directory=persist_directory, embedding_function=text_embedder)

    text_docs = extract_text_chunks(pdf_path)
    image_docs = extract_images(pdf_path)

    # Use text embedder only for text docs — images already contain vector in metadata
    vectordb = Chroma.from_documents(
        documents=text_docs,
        embedding=text_embedder,
        persist_directory=persist_directory
    )

    # Manually insert image embeddings
    for img_doc in image_docs:
        vectordb.collection.add(
            embeddings=[img_doc.metadata["vector"]],
            documents=[img_doc.page_content],
            metadatas=[img_doc.metadata]
        )

    vectordb.persist()
    return vectordb

def get_qa_chain(vectordb):
    llm = OllamaRESTLLM()

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
You are an expert assistant. Use the provided context to answer the user's question.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    return RetrievalQA(
        retriever=vectordb.as_retriever(search_kwargs={"k": 6}),
        combine_documents_chain=combine_documents_chain,
        return_source_documents=True
    )


# def get_qa_chain(vectordb, ollama_model=OLLAMA_MODEL):
#     #llm = Ollama(model=ollama_model)
#     llm = Ollama(
#         model=ollama_model,
#         base_url=OLLAMA_BASE_URL
#     )
#
#     prompt = PromptTemplate(
#         input_variables=["question", "context"],
#         template="""
# You are an expert assistant. Use the provided context to answer the user's question.
#
# Context:
# {context}
#
# Question:
# {question}
#
# Answer:
# """
#     )
#
#     llm_chain = LLMChain(llm=llm, prompt=prompt)
#
#     combine_documents_chain = StuffDocumentsChain(
#         llm_chain=llm_chain,
#         document_variable_name="context"
#     )
#
#     return RetrievalQA(
#         retriever=vectordb.as_retriever(search_kwargs={"k": 6}),
#         combine_documents_chain=combine_documents_chain,
#         return_source_documents=True
#     )

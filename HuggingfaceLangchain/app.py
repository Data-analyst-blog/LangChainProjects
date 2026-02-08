# Databricks notebook source
import streamlit as st
import validators
from langchain_classic.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
import os
from dotenv import load_dotenv

load_dotenv()

## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the HF_API_KEY and URL(YT or Website) to be summarized
with st.sidebar:
    huggingface_api_key = st.text_input("Huggingface API Token", value="",type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model using Huggingface API
repo_id = "google/gemma-2-9b"

# Fix: Added task="conversational" 
llm = HuggingFaceEndpoint(
    repo_id=repo_id, 
    task="conversational",
    max_new_tokens=150, 
    temperature=0.7,
    huggingfacehub_api_token=huggingface_api_key
)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
    

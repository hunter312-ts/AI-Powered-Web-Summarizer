import validators # for vlaidating urls
import streamlit as st
from validators import url as is_url
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain # Summarize chain loader
from langchain_community.document_loaders import YoutubeLoader , UnstructuredURLLoader


# Streamlit Layout Setup
st.set_page_config(page_title="URL → Summary",page_icon="📺")
st.title(" 🌏 → 📑 Text Summarizer ")
st.subheader("Summarize a youtube vedio or any web page")

# Sidebar UI: Instruction , Groq_api_key , And Summary length slider

with st.sidebar:
    st.markdown(
        """
        📜 How to use :
        1. Enter your GROQ API Key.
        2. Adjust summary length if you like.
        3. Paste a Youtube or webpage URL.
        4. Press summarize
        """
    )
    api_key=st.text_input("Enter your GROQ API KEY",type="password")
    summary_length=st.slider("Summary Length (words)",100,500,300, step=50)
# Main input for url
generic_url=st.text_input("URL: ",label_visibility="collapsed")

# Show warning in main body if api key is missing 
if not api_key.strip():
    st.warning("Please eneter your api key",icon="🚨")
else:
    # Only show summarize button if api key is present
    if st.button("📝 Summarize..."):
        if not generic_url.strip() or not api_key.strip():
            st.error("Please enter URL to get started.")
        elif not is_url(generic_url):
            st.error("Please enter a vlaid url (e.g Youtube or webpage)")
        else:
            with st.spinner("Fetching document.."):
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(
                        generic_url,add_video_info=True
                    )
                else:
                    loader=UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36" }
                    )
                docs=loader.load()
            # Initilize the llm with provided api key
            llm=ChatGroq(model="llama3-8b-8192",api_key=api_key)

            prompt=PromptTemplate(
                template=(f"Provide a summary of the following context in {summary_length} words:\n\n{{text}}"),
                input_variables=["text"]

            )
            # load the summarization chain stuff method.
            chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
            # Show spinner while llm laod the summary
            with st.spinner("Summarizing...."):
                summary=chain.run(docs)
            # Display the succes message and Summary
            st.success("Success..")
            st.markdown("## Summary")
            st.write(summary)

st.markdown("---")
st.markdown(
    " Built with 👨🏻‍🎓 Data Science Batch 01 👩🏻‍🎓 using [Streamlit] / [Langchain] / [GroqAPI] "
)
            


    
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from duckduckgo_search import DDGS

# --- Hardcoded Gemini API Key ---
GEMINI_API_KEY = "AIzaSyBncTLXHYefhscCMobtfiu2HsJbNUaKkZU"

# --- Initialize Gemini LLM ---
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)

# --- Create search tool function ---
def ddg_search(query):
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return "\n".join([r["body"] for r in results[:3]]) or "No relevant results found."

# --- Initialize search tool ---
search_tool = Tool(
    name="DuckDuckGo Search",
    func=ddg_search,
    description="Useful for answering questions about current events, recent facts, or the real world."
)

# --- Initialize agent ---
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=False
)

# --- Streamlit App UI ---
st.set_page_config(page_title="🌎 Real-Time Q&A", page_icon="🤖")
st.title("🤖 Ask Anything! 🌍")
st.markdown("Ask real-time questions about **current events** or **facts**. Powered by Gemini + DuckDuckGo!")

question = st.text_input("💬 Your Question:", placeholder="e.g. What's the latest news on AI?")

if st.button("🔍 Get Answer"):
    if not question.strip():
        st.warning("⚠️ Please enter a question!")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                answer = agent.run(question)
                st.success("✅ Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Oops! Something went wrong: {e}")

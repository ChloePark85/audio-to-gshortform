from crewai import Agent
from langchain.chat_models import ChatOpenAI
from tools import extract_highlights_tool, create_highlight_clips_tool

highlight_agent = Agent(
    role='Highlight Extractor',
    goal='Extract the most interesting and relevant highlights from audio content',
    backstory='You are an expert in audio analysis and content curation. You have a keen ear for identifying the most engaging and relevant parts of any audio content.',
    verbose=True,
    allow_delegation=False,
    tools=[extract_highlights_tool, create_highlight_clips_tool],
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
)

from crewai import Agent
from langchain.chat_models import ChatOpenAI
from tools import extract_highlights_tool, create_highlight_clips_tool, generate_subtitle_tool, summarize_text_tool, generate_image_tool, add_header_to_image_tool


highlight_agent = Agent(
    role='Highlight Extractor',
    goal='Extract the most interesting and relevant highlights from audio content',
    backstory='You are an expert in audio analysis and content curation. You have a keen ear for identifying the most engaging and relevant parts of any audio content.',
    verbose=True,
    allow_delegation=False,
    tools=[extract_highlights_tool, create_highlight_clips_tool],
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
)
subtitle_agent = Agent(
    role='Subtitle Generator',
    goal='Generate accurate subtitles for audio clips',
    backstory='You are a skilled transcriptionist with expertise in multiple languages. You can quickly and accurately transcribe speech from various audio sources.',
    verbose=True,
    allow_delegation=False,
    tools=[generate_subtitle_tool],
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
)

summarization_agent = Agent(
    role='Text Summarizer',
    goal='Summarize text content concisely and accurately',
    backstory='You are an expert in condensing complex information into brief, meaningful summaries. You have a knack for identifying key points in any text.',
    verbose=True,
    allow_delegation=False,
    tools=[summarize_text_tool],
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
)

image_generation_agent = Agent(
    role='Image Generator',
    goal='Create visually appealing images based on text descriptions',
    backstory='You are a creative AI artist with a talent for translating text into stunning visual representations. You have a deep understanding of visual aesthetics and design principles.',
    verbose=True,
    allow_delegation=False,
    tools=[generate_image_tool, add_header_to_image_tool],
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
)

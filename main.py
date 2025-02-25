# from slack_sdk import WebClient
# from slack_sdk.errors import SlackApiError

from crewai import Crew
from tasks import (get_highlight_extraction_task, get_subtitle_generation_task,
                  get_veo2_generation_task)

import asyncio
from tools import generate_video_from_image

async def process_video(image_paths: list, prompts: list, audio_segments: list):
    # 각 이미지에 대해 비디오 생성
    video_urls = []
    for image_path, prompt in zip(image_paths, prompts):
        video_url = await generate_video_from_image(image_path, prompt)
        if video_url:
            video_urls.append(video_url)
    
    return video_urls
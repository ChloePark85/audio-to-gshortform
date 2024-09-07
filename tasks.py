from crewai import Task
from agents import highlight_agent, subtitle_agent, summarization_agent, image_generation_agent

def execute_task_with_error_handling(task: Task):
    try:
        result = task.execute()
        if isinstance(result, str) and ("Error" in result or "error" in result.lower()):
            raise Exception(result)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def get_highlight_extraction_task(audio_path: str) -> Task:
    return Task(
        description=f"오디오 파일({audio_path})에서 5개의 하이라이트를 추출합니다. 각 하이라이트의 길이는 25초-30초입니다.",
        agent=highlight_agent,
        expected_output="추출된하이라이트 오디오 파일 mp3"
    )
def get_subtitle_generation_task(audio_clip) -> Task:
    return Task(
        description=f"주어진 오디오 클립에 대한 자막을 생성합니다.",
        agent=subtitle_agent,
        expected_output="생성된 자막 텍스트"
    )

def get_text_summarization_task(text: str, max_length: int = 10) -> Task:
    return Task(
        description=f"주어진 텍스트를 {max_length}자 이내로 요약합니다.",
        agent=summarization_agent,
        expected_output="요약된 텍스트"
    )

def get_image_generation_task(subtitle: str) -> Task:
    prompt = f"""Create a stylish illustration featuring an attractive character based on the following scene: {subtitle}. 
Key points:
- The character should be visually appealing and well-designed
- Use a sophisticated illustration style
- Do not include any speech bubbles or text in the image
- The image should be suitable for a short-form video thumbnail
- Focus on creating a visually striking and engaging scene
"""
    return Task(
        description=f"주어진 프롬프트를 바탕으로 이미지를 생성합니다: {prompt}",
        agent=image_generation_agent,
        expected_output="생성된 이미지 객체"
    )

def get_image_header_addition_task(image, header_text: str) -> Task:
    return Task(
        description=f"주어진 이미지에 헤더 텍스트를 추가합니다: {header_text}",
        agent=image_generation_agent,
        expected_output="헤더가 추가된 이미지 객체"
    )


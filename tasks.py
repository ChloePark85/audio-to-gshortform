from crewai import Task
from agents import highlight_agent

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



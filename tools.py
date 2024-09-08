import librosa
import numpy as np
from pydub import AudioSegment
from langchain.tools import Tool
import tempfile
from openai import OpenAI
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os


client = OpenAI()

def find_sentence_end(y, sr, start_time, min_duration=20, max_duration=30):
    """문장이 끝나는 지점을 찾습니다."""
    start_sample = int(start_time * sr)
    min_samples = int(min_duration * sr)
    max_samples = int(max_duration * sr)
    
    # RMS 에너지 계산
    frame_length = 1024
    hop_length = 512
    rms = librosa.feature.rms(y=y[start_sample:start_sample+max_samples], frame_length=frame_length, hop_length=hop_length)[0]
    
    # 에너지가 낮은 지점 찾기 (문장의 끝으로 간주)
    threshold = np.mean(rms) * 0.5
    potential_ends = np.where(rms < threshold)[0]
    
    for end in potential_ends:
        duration = (end * hop_length) / sr
        if min_duration <= duration <= max_duration:
            return start_time + duration
    
    # 적절한 끝점을 찾지 못한 경우, 최소 길이와 최대 길이  임의의 지점 반환
    return start_time + np.random.uniform(min_duration, max_duration)

def extract_highlights(audio_path: str, num_highlights: int = 5) -> list:
    """
    오디오 파일에서 중복되지 않는 하이라이트를 추출합니다.

    Args:
        audio_path (str): 오디오 파일의 경로
        num_highlights (int): 추출할 하이라이트의 수 (기본값: 5)

    Returns:
        list: 하이라이트 시작 시간과 종료 시간(초)의 리스트
    """
    y, sr = librosa.load(audio_path)
    
    # RMS 에너지 계산
    rms = librosa.feature.rms(y=y)[0]
    
    # 프레임을 시간(초)으로 변환
    times = librosa.times_like(rms)
    
    # RMS 에너지가 높은 구간 찾기
    threshold = np.mean(rms) + np.std(rms)
    peaks = librosa.util.peak_pick(rms, pre_max=30, post_max=30, pre_avg=30, post_avg=30, delta=threshold, wait=30)
    
    # 하이라이트 시작 시간 추출
    highlight_starts = times[peaks]
    
    # 중복되지 않는 하이라이트 선택
    unique_highlights = []
    for start in highlight_starts:
        if not any(start < existing[1] and start > existing[0] for existing in unique_highlights):
            end = find_sentence_end(y, sr, start)
            duration = end - start
            if 20 <= duration <= 30:
                unique_highlights.append((start, end))
                if len(unique_highlights) == num_highlights:
                    break
    
    # 하이라이트 수가 부족한 경우 추가 처리
    while len(unique_highlights) < num_highlights:
        start = np.random.uniform(0, len(y)/sr - 30)
        if not any(start < existing[1] and start > existing[0] for existing in unique_highlights):
            end = find_sentence_end(y, sr, start)
            duration = end - start
            if 20 <= duration <= 30:
                unique_highlights.append((start, end))
    
    return sorted(unique_highlights)

def create_highlight_clips(audio_path: str, highlight_times: dict) -> list:
    """
    오디오 파일에서 하이라이트 클립을 생성합니다.
    """
    audio = AudioSegment.from_file(audio_path)
    highlights = []
    
    for start, end in highlight_times:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        if end_ms > len(audio):
            end_ms = len(audio)
        if start_ms < end_ms:
            highlight = audio[start_ms:end_ms]
            highlights.append(highlight)
    
    return highlights
    
    

def generate_subtitle(audio_segment: AudioSegment) -> str:
    """
    오디오 세그먼트에 대한 자막을 생성합니다.

    Args:
        audio_segment (AudioSegment): 자막을 생성할 오디오 세그먼트

    Returns:
        str: 생성된 자막 텍스트
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
        audio_segment.export(temp_audio_file.name, format="mp3")
        temp_audio_file.close()
        
        with open(temp_audio_file.name, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
    
    os.unlink(temp_audio_file.name)
    return transcription

def generate_image(prompt: str) -> Image.Image:
    """
    DALL-E를 사용하여 주어진 프롬프트에 기반한 이미지를 생성합니다.
    """
    # 프롬프트를 더 일반적이고 중립적으로 수정
    safe_prompt = f"Create a simple, stylized illustration inspired by the following concept: {prompt}. Use abstract shapes and neutral colors."
    
    response = client.images.generate(
            model="dall-e-3",
            prompt=safe_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

    image_url = response.data[0].url
    image_response = requests.get(image_url)
    image = Image.open(BytesIO(image_response.content))
        
    return image

def add_background(image, target_ratio=2/3):
    """이미지에 검은색 배경을 추가하여 2:3 비율로 만듭니다."""
    width, height = image.size
    new_height = int(width / target_ratio)
    result = Image.new('RGB', (width, new_height), color='black')
    paste_y = (new_height - height) // 2
    result.paste(image, (0, paste_y))
    
    return result

def summarize_text(text: str, max_length: int = 10) -> str:
    """
    OpenAI API를 사용하여 텍스트를 요약합니다.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"주어진 텍스를 {max_length}자 이내의 한글로 요약해주세요."},
            {"role": "user", "content": text}
        ],
        max_tokens=30,
        n=1,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def add_header_to_image(image: Image.Image, header_text: str) -> Image.Image:
    """
    이미지의 상단 검은색 부분에 헤더 텍스트를 추가합니다.
    텍스트는 흰색으로, 중앙정렬하여 2줄로 표시되며, 더 두껍고 강조되어 있습니다.양쪽에 10px 마진이 있습니다.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # 마진 설정
    margin = 10
    
    # 폰트 설정 (폰트 파일 경로를 적절히 수정해주세요)
    font_size = int(width / 18)  

    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
        "C:/Windows/Fonts/malgun.ttf",  # Windows
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # 다른 Linux 배포판
    ]
    
    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except IOError:
                continue
    
    if font is None:
        print("Warning: 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    max_width = width - 2 * margin

    # 텍스트 줄바꿈 로직 개선
    lines = []
    for line in textwrap.wrap(header_text, width=int(max_width / (font_size / 2))):
        if font.getbbox(line)[2] <= max_width:
            lines.append(line)
        else:
            # 라인이 너무 길면 더 작은 부분으로 나눕니다
            words = line.split()
            new_line = ""
            for word in words:
                test_line = new_line + " " + word if new_line else word
                if font.getbbox(test_line)[2] <= max_width:
                    new_line = test_line
                else:
                    if new_line:
                        lines.append(new_line)
                    new_line = word
            if new_line:
                lines.append(new_line)
        if len(lines) == 2:
            break

    if len(lines) > 2:
        lines = lines[:2]
    
    bbox = font.getbbox('hg')
    line_height = bbox[3] - bbox[1]
    text_height = line_height * len(lines)
    
    # 시작 y 좌표 계산
    y = (height / 6 - text_height / 2) - 70
    
    # 각 줄 그리기
    for line in lines:
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]
        
        # x 좌표 계산 (중앙 정렬, 마진 강제 적용)
        x = max(margin, (width - line_width) / 2)
        
        # 외곽선 그리기
        outline_color = "black"
        outline_width = 3
        for offset_x in range(-outline_width, outline_width + 1):
            for offset_y in range(-outline_width, outline_width + 1):
                draw.text((x + offset_x, y + offset_y), line, font=font, fill=outline_color)
        
        # 그림자 그리기
        shadow_color = "black"
        shadow_offset = 3
        draw.text((x + shadow_offset, y + shadow_offset), line, font=font, fill=shadow_color)
        
        # 메인 텍스트 그리기
        draw.text((x, y), line, font=font, fill="white")
        
        y += line_height
    
    return image


# Tool 객체로 래핑
extract_highlights_tool = Tool(
    name="Extract Highlights",
    func=extract_highlights,
    description="Extracts highlights from an audio file. Input: {\"audio_path\": \"path/to/audio/file.mp3\"}"
)

create_highlight_clips_tool = Tool(
    name="Create Highlight Clips",
    func=create_highlight_clips,
    description="Creates highlight clips from extracted highlights. Input: {\"audio_path\": \"path/to/audio/file.mp3\", \"highlight_times\": [[start1, end1], [start2, end2], ...]}"
)

generate_subtitle_tool = Tool(
    name="Generate Subtitle",
    func=generate_subtitle,
    description="Generates subtitle for an audio segment. Input: AudioSegment object"
)

generate_image_tool = Tool(
    name="Generate Image",
    func=generate_image,
    description="Generates an image based on a text prompt using DALL-E. Input: {\"prompt\": \"text description\"}"
)
summarize_text_tool = Tool(
    name="Summarize Text",
    func=summarize_text,
    description="Summarizes the given text in Korean within the specified length. Input: {\"text\": \"text to summarize\", \"max_length\": 10}"
)

add_header_to_image_tool = Tool(
    name="Add Header to Image",
    func=add_header_to_image,
    description="Adds a header text to the top black part of the image. Input: {\"image\": PIL Image object, \"header_text\": \"header text\"}"
)
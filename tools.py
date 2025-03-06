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
from dotenv import load_dotenv
import fal_client
import base64
import io
import cv2
import asyncio
import ffmpeg
import whisper
import json
import streamlit as st
import random

load_dotenv()

# FAL API 키 설정 (없으면 건너뛰기)
fal_key = os.getenv('FAL_KEY')
if fal_key:
    os.environ['FAL_KEY'] = fal_key

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

# def extract_highlights(audio_path: str, num_highlights: int = 5) -> list:
#     """
#     오디오 파일에서 중복되지 않는 하이라이트를 추출합니다.

#     Args:
#         audio_path (str): 오디오 파일의 경로
#         num_highlights (int): 추출할 하이라이트의 수 (기본값: 5)

#     Returns:
#         list: 하이라이트 시작 시간과 종료 시간(초)의 리스트
#     """
#     y, sr = librosa.load(audio_path)
    
#     # RMS 에너지 계산
#     rms = librosa.feature.rms(y=y)[0]
    
#     # 프레임을 시간(초)으로 변환
#     times = librosa.times_like(rms)
    
#     # RMS 에너지가 높은 구간 찾기
#     threshold = np.mean(rms) + np.std(rms)
#     peaks = librosa.util.peak_pick(rms, pre_max=30, post_max=30, pre_avg=30, post_avg=30, delta=threshold, wait=30)
    
#     # 하이라이트 시작 시간 추출
#     highlight_starts = times[peaks]
    
#     # 중복되지 않는 하이라이트 선택
#     unique_highlights = []
#     for start in highlight_starts:
#         if not any(start < existing[1] and start > existing[0] for existing in unique_highlights):
#             end = find_sentence_end(y, sr, start)
#             duration = end - start
#             if 20 <= duration <= 30:
#                 unique_highlights.append((start, end))
#                 if len(unique_highlights) == num_highlights:
#                     break
    
#     # 하이라이트 수가 부족한 경우 추가 처리
#     while len(unique_highlights) < num_highlights:
#         start = np.random.uniform(0, len(y)/sr - 30)
#         if not any(start < existing[1] and start > existing[0] for existing in unique_highlights):
#             end = find_sentence_end(y, sr, start)
#             duration = end - start
#             if 20 <= duration <= 30:
#                 unique_highlights.append((start, end))
    
#     return sorted(unique_highlights)

# def create_highlight_clips(audio_path: str, highlight_times: dict) -> list:
#     """
#     오디오 파일에서 하이라이트 클립을 생성합니다.
#     """
#     audio = AudioSegment.from_file(audio_path)
#     highlights = []
    
#     for start, end in highlight_times:
#         start_ms = int(start * 1000)
#         end_ms = int(end * 1000)
#         if end_ms > len(audio):
#             end_ms = len(audio)
#         if start_ms < end_ms:
#             highlight = audio[start_ms:end_ms]
#             highlights.append(highlight)
    
#     return highlights
    
    

# def generate_subtitle(audio_segment: AudioSegment) -> str:
#     """
#     오디오 세그먼트에 대한 자막을 생성합니다.

#     Args:
#         audio_segment (AudioSegment): 자막을 생성할 오디오 세그먼트

#     Returns:
#         str: 생성된 자막 텍스트
#     """
#     with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
#         audio_segment.export(temp_audio_file.name, format="mp3")
#         temp_audio_file.close()
        
#         with open(temp_audio_file.name, "rb") as audio_file:
#             transcription = client.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=audio_file,
#                 response_format="text"
#             )
    
#     os.unlink(temp_audio_file.name)
#     return transcription

async def generate_video_from_image(image_path: str, prompt: str) -> str:
    """
    이미지에 다양한 효과를 적용하여 비디오를 생성합니다. (비동기 버전)
    """
    return await create_video_effect(image_path, effect_type=prompt)

async def create_video_effect(image_path: str, effect_type: str = "zoom_in", duration: int = 5) -> str:
    """
    이미지에 다양한 효과를 적용하여 비디오를 생성합니다. (비동기 버전)
    """
    # CPU 집약적인 작업을 별도의 스레드에서 실행
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _create_video_effect_sync(image_path, effect_type, duration))

def get_audio_duration(audio_path: str) -> float:
    """
    오디오 파일의 길이를 초 단위로 반환합니다.
    """
    try:
        probe = ffmpeg.probe(audio_path)
        audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        raise Exception(f"오디오 길이 확인 중 오류 발생: {str(e)}")

def add_text_overlay(frame, text, frame_height, frame_width):
    """
    비디오 프레임에 텍스트를 오버레이합니다.
    텍스트는 화면 중앙에 표시되며, 그림자 효과로 가독성을 높입니다.
    """
    # 이미지를 PIL Image로 변환
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 한글 폰트 설정
    try:
        font = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 60)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 60)
        except:
            font = ImageFont.load_default()
    
    # 텍스트 줄바꿈 처리
    wrapper = textwrap.TextWrapper(width=20)
    text_lines = wrapper.wrap(text)
    
    # 전체 텍스트 영역의 높이 계산
    line_height = 75
    total_text_height = len(text_lines) * line_height
    
    # 텍스트 시작 y 좌표 (화면 중앙)
    text_y = (frame_height - total_text_height) // 2
    
    for line in text_lines:
        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        
        # 텍스트 중앙 정렬 위치 계산
        x = (frame_width - text_width) // 2
        
        # 그림자 효과 추가 (여러 방향으로 두꺼운 그림자)
        shadow_offset = 3
        shadow_color = (0, 0, 0)
        
        # 8방향 그림자로 더 진한 효과 생성
        for offset_x in [-shadow_offset, 0, shadow_offset]:
            for offset_y in [-shadow_offset, 0, shadow_offset]:
                if offset_x == 0 and offset_y == 0:
                    continue
                draw.text(
                    (x + offset_x, text_y + offset_y),
                    line,
                    font=font,
                    fill=shadow_color
                )
        
        # 더 진한 그림자를 위해 한번 더 그리기
        for offset_x in [-shadow_offset//2, shadow_offset//2]:
            for offset_y in [-shadow_offset//2, shadow_offset//2]:
                draw.text(
                    (x + offset_x, text_y + offset_y),
                    line,
                    font=font,
                    fill=shadow_color
                )
        
        # 메인 텍스트 그리기 (흰색)
        draw.text((x, text_y), line, font=font, fill=(255, 255, 255))
        text_y += line_height
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def add_black_bars_to_square(frame):
    """
    1:1 비율의 프레임을 9:16 비율로 변환하고 위아래에 검은색 바를 추가합니다.
    정사각형 영역을 약간 아래로 이동시켜 배치합니다.
    """
    square_size = frame.shape[0]  # 정사각형 크기
    target_height = int(square_size * 16/9)  # 9:16 비율의 높이
    
    # 위아래 바의 전체 높이
    total_bar_height = target_height - square_size
    
    # 위쪽 바를 더 크게, 아래쪽 바를 더 작게 설정하여 정사각형을 아래로 이동
    top_bar_height = total_bar_height * 0.6  # 60%를 위쪽에
    bottom_bar_height = total_bar_height * 0.4  # 40%를 아래쪽에
    
    # 검은색 바가 포함된 새 프레임 생성
    new_frame = np.zeros((target_height, square_size, 3), dtype=np.uint8)
    
    # 원본 이미지를 아래로 이동하여 배치
    start_y = int(top_bar_height)
    new_frame[start_y:start_y+square_size, :] = frame
    
    return new_frame

def _create_video_effect_sync(image_path: str, effect_type: str = "zoom_in", duration: float = 5.0, text=""):
    """
    실제 비디오 생성을 수행하는 동기 함수
    9:16 비율의 비디오 생성
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("이미지를 불러올 수 없습니다.")
    
    # BGR을 RGB로 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 이미지를 정사각형으로 리사이즈
    size = 1080  # Instagram 권장 크기
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    
    # 9:16 비율의 프레임 크기 계산
    frame_width = size
    frame_height = int(size * 16/9)
    frame_size = (frame_width, frame_height)
    
    # 비디오 설정
    fps = 30
    total_frames = int(duration * fps)
    
    # 임시 비디오 파일 생성
    temp_video_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, frame_size)
    
    try:
        for frame_num in range(total_frames):
            progress = frame_num / total_frames
            frame = img.copy()
            
            # 효과 적용
            if effect_type == "zoom_in":
                scale = 1.0 + (0.5 * progress)
                center_x, center_y = size / 2, size / 2
                M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
                frame = cv2.warpAffine(frame, M, (size, size))
            elif effect_type == "zoom_out":
                scale = 1.5 - (0.5 * progress)
                center_x, center_y = size / 2, size / 2
                M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
                frame = cv2.warpAffine(frame, M, (size, size))
            elif effect_type == "pan_left_to_right":
                offset_x = int(size * 0.2 * progress)
                M = np.float32([[1, 0, offset_x], [0, 1, 0]])
                frame = cv2.warpAffine(frame, M, (size, size))
            elif effect_type == "pan_right_to_left":
                offset_x = int(size * 0.2 * (1 - progress))
                M = np.float32([[1, 0, offset_x], [0, 1, 0]])
                frame = cv2.warpAffine(frame, M, (size, size))
            elif effect_type == "pan_top_to_bottom":
                offset_y = int(size * 0.2 * progress)
                M = np.float32([[1, 0, 0], [0, 1, offset_y]])
                frame = cv2.warpAffine(frame, M, (size, size))
            elif effect_type == "pan_bottom_to_top":
                offset_y = int(size * 0.2 * (1 - progress))
                M = np.float32([[1, 0, 0], [0, 1, offset_y]])
                frame = cv2.warpAffine(frame, M, (size, size))
            elif effect_type == "rotate_clockwise":
                angle = 5 * progress
                M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
                frame = cv2.warpAffine(frame, M, (size, size))
            elif effect_type == "rotate_counterclockwise":
                angle = -5 * progress
                M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
                frame = cv2.warpAffine(frame, M, (size, size))
            elif effect_type == "ken_burns":
                # Ken Burns 효과: 확대 + 이동
                scale = 1.0 + (0.3 * progress)
                offset_x = int(size * 0.1 * progress)
                offset_y = int(size * 0.1 * progress)
                M = cv2.getRotationMatrix2D((size/2 - offset_x, size/2 - offset_y), 0, scale)
                frame = cv2.warpAffine(frame, M, (size, size))
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 텍스트 오버레이 추가
            if text:
                frame = add_text_overlay(frame, text, size, size)
            
            # 9:16 비율로 변환하고 검은색 바 추가
            frame = add_black_bars_to_square(frame)
            
            out.write(frame)
        
        out.release()
        
        # ffmpeg를 사용하여 웹 호환 형식으로 변환
        web_compatible_path = tempfile.mktemp(suffix='.mp4')
        
        stream = ffmpeg.input(temp_video_path)
        stream = ffmpeg.output(stream, web_compatible_path,
                             vcodec='libx264',
                             pix_fmt='yuv420p',
                             acodec='aac',
                             strict='experimental')
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        # 임시 파일 정리
        os.remove(temp_video_path)
        
        return web_compatible_path
        
    except Exception as e:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise Exception(f"비디오 생성 중 오류 발생: {str(e)}")

def create_zoom_effect(image_path: str, effect_type: str = "zoom_in", duration: int = 5) -> str:
    """
    이미지에 줌 인/아웃 효과를 적용하여 비디오를 생성합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        effect_type (str): 'zoom_in' 또는 'zoom_out'
        duration (int): 비디오 길이(초)
    
    Returns:
        str: 생성된 비디오의 파일 경로
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("이미지를 불러올 수 없습니다.")
    
    # 비디오 설정
    fps = 30
    total_frames = duration * fps
    frame_size = (img.shape[1], img.shape[0])
    
    # 임시 비디오 파일 생성
    temp_video_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, frame_size)
    
    try:
        # 줌 효과 생성
        for frame_num in range(total_frames):
            progress = frame_num / total_frames
            if effect_type == "zoom_in":
                scale = 1.0 + (0.5 * progress)  # 1.0에서 1.5까지 확대
            else:  # zoom_out
                scale = 1.5 - (0.5 * progress)  # 1.5에서 1.0까지 축소
            
            # 이미지 중앙에서 확대/축소
            center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            frame = cv2.warpAffine(img, M, frame_size)
            
            out.write(frame)
        
        out.release()
        return temp_video_path
        
    except Exception as e:
        out.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise e

def split_audio(audio_path: str, num_segments: int = 5) -> list:
    """오디오 파일을 5초 길이의 세그먼트로 나눕니다."""
    audio = AudioSegment.from_file(audio_path)
    segment_length = 5000  # 5초 = 5000ms
    segments = []
    
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = audio[start:end]
        
        temp_path = f"temp_segment_{i}.mp3"
        segment.export(temp_path, format="mp3")
        segments.append(temp_path)
    
    return segments

def combine_videos(video_paths: list) -> list:
    """
    여러 비디오의 경로를 리스트로 반환합니다.
    
    Args:
        video_paths (list): 비디오 파일 경로 리스트
    
    Returns:
        list: 비디오 파일 경로 리스트
    """
    return video_paths

# def summarize_text(text: str, max_length: int = 10) -> str:
#     """
#     주어진 텍스트를 지정된 길이 이내로 요약합니다.
    
#     Args:
#         text (str): 요약할 텍스트
#         max_length (int): 요약문의 최대 길이 (기본값: 10)
    
#     Returns:
#         str: 요약된 텍스트
#     """
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "주어진 텍스트를 간단명료하게 요약해주세요."},
#             {"role": "user", "content": f"다음 텍스트를 {max_length}자 이내로 요약해주세요:\n\n{text}"}
#         ]
#     )
    
#     return response.choices[0].message.content.strip()

# Tool 객체로 래핑
# extract_highlights_tool = Tool(
#     name="Extract Highlights",
#     func=extract_highlights,
#     description="Extracts highlights from an audio file. Input: {\"audio_path\": \"path/to/audio/file.mp3\"}"
# )

# create_highlight_clips_tool = Tool(
#     name="Create Highlight Clips",
#     func=create_highlight_clips,
#     description="Creates highlight clips from extracted highlights. Input: {\"audio_path\": \"path/to/audio/file.mp3\", \"highlight_times\": [[start1, end1], [start2, end2], ...]}"
# )

# generate_subtitle_tool = Tool(
#     name="Generate Subtitle",
#     func=generate_subtitle,
#     description="Generates subtitle for an audio segment. Input: AudioSegment object"
# )

# summarize_text_tool = Tool(
#     name="Summarize Text",
#     func=summarize_text,
#     description="Summarizes the given text in Korean within the specified length. Input: {\"text\": \"text to summarize\", \"max_length\": 10}"
# )

generate_video_tool = Tool.from_function(
    func=generate_video_from_image,
    name="generate_video",
    description="Generate video using Kling API based on image and prompt"
)

def combine_video_audio(video_path: str, audio_path: str) -> str:
    """
    비디오와 오디오를 결합합니다.
    """
    try:
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # 오디오 길이 가져오기
        duration = get_audio_duration(audio_path)
        
        # 비디오와 오디오 입력 스트림 생성
        video = ffmpeg.input(video_path)
        audio = ffmpeg.input(audio_path)
        
        # 비디오 길이를 오디오 길이와 동일하게 설정
        stream = ffmpeg.output(
            video,
            audio,
            output_path,
            vcodec='libx264',
            pix_fmt='yuv420p',
            acodec='aac',
            t=str(duration),  # 오디오 길이로 설정
            strict='experimental'
        )
        
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        return output_path
        
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"비디오와 오디오 결합 중 오류 발생: {str(e)}")

def combine_all_videos(video_paths):
    """
    모든 비디오를 순서대로 결합합니다.
    """
    try:
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # 비디오 파일 목록 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for video_path in video_paths:
                f.write(f"file '{video_path}'\n")
        
        # ffmpeg를 사용하여 비디오 결합
        stream = ffmpeg.input(f.name, f='concat', safe=0)
        stream = ffmpeg.output(stream, output_path,
                             vcodec='libx264',
                             pix_fmt='yuv420p',
                             acodec='aac',
                             strict='experimental')
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        os.unlink(f.name)  # 임시 파일 삭제
        return output_path
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"비디오 결합 중 오류 발생: {str(e)}")

def split_long_segments(segments, max_duration=5.0):
    """
    긴 세그먼트를 자연스럽게 분할합니다.
    단어별 타임스탬프를 사용하여 가장 적절한 지점에서 분할합니다.
    """
    new_segments = []
    
    for segment in segments:
        duration = segment["end"] - segment["start"]
        
        # 최대 길이를 초과하는 경우에만 분할
        if duration > max_duration:
            words = segment["words"]
            current_words = []
            current_start = segment["start"]
            current_duration = 0
            
            for word in words:
                word_duration = word["end"] - word["start"]
                
                # 현재 단어를 추가했을 때 최대 길이를 초과하는지 확인
                if current_duration + word_duration > max_duration:
                    # 현재까지의 단어들로 새 세그먼트 생성
                    if current_words:
                        new_segments.append({
                            "text": "".join(current_words).strip(),
                            "start": current_start,
                            "end": word["start"],
                            "words": current_words
                        })
                    
                    # 새로운 세그먼트 시작
                    current_words = [word["word"]]
                    current_start = word["start"]
                    current_duration = word_duration
                else:
                    current_words.append(word["word"])
                    current_duration += word_duration
            
            # 남은 단어들로 마지막 세그먼트 생성
            if current_words:
                new_segments.append({
                    "text": "".join(current_words).strip(),
                    "start": current_start,
                    "end": segment["end"],
                    "words": current_words
                })
        else:
            new_segments.append(segment)
    
    return new_segments

def transcribe_audio(audio_path: str) -> list:
    """
    오디오 파일을 텍스트로 변환하고 문장 단위로 세그먼트를 생성합니다.
    """
    try:
        # Whisper 모델 로드
        model = whisper.load_model("base")
        
        # 음성 인식 실행
        result = model.transcribe(
            audio_path,
            language="ko",
            word_timestamps=True,
            verbose=False
        )
        
        segments = []
        
        # 세그먼트 단위로 처리
        if isinstance(result, dict) and "segments" in result:
            for segment in result["segments"]:
                words = []
                if isinstance(segment, dict):
                    # 단어 정보 추출
                    if "words" in segment and isinstance(segment["words"], list):
                        for word_info in segment["words"]:
                            if isinstance(word_info, dict):
                                words.append({
                                    "word": str(word_info.get("word", "")),
                                    "start": float(word_info.get("start", 0)),
                                    "end": float(word_info.get("end", 0))
                                })
                    
                    # 세그먼트 정보 저장
                    segments.append({
                        "text": str(segment.get("text", "")).strip(),
                        "start": float(segment.get("start", 0)),
                        "end": float(segment.get("end", 0)),
                        "words": words
                    })
        
        return segments
        
    except Exception as e:
        print(f"Transcription error details: {str(e)}")  # 디버깅용 로그
        raise Exception(f"음성 인식 중 오류 발생: {str(e)}")

def extract_interesting_part(segments):
    """
    2분(120초) 길이의 흥미로운 부분을 추출
    """
    if not segments:
        raise ValueError("세그먼트가 비어있습니다.")
    
    # 연속된 120초 구간을 찾기
    best_start = 0
    target_duration = 120  # 2분
    
    for i, segment in enumerate(segments):
        current_start = segment['start']
        current_end = current_start
        current_segments = []
        
        # 120초 구간에 포함되는 세그먼트들 수집
        for next_segment in segments[i:]:
            if next_segment['end'] - current_start <= target_duration:
                current_segments.append(next_segment)
                current_end = next_segment['end']
            else:
                break
        
        if current_segments:
            best_start = i
            break
    
    selected_segments = segments[best_start:best_start + len(current_segments)]
    
    return {
        'title': "2분 하이라이트",
        'reason': "핵심 내용이 포함된 2분 구간입니다.",
        'start': selected_segments[0]['start'],
        'end': selected_segments[-1]['end'],
        'segments': selected_segments
    }

def extract_shorts_segments(interesting_part, audio_path):
    """
    흥미로운 부분에서 문장 단위로 정확하게 세그먼트를 추출합니다.
    """
    segments = interesting_part["segments"]
    shorts_segments = []
    
    # 시간 보정값 (초)
    time_padding = 0.2
    
    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
            
        # 문장의 정확한 시작과 끝 시간 가져오기
        start_time = segment["start"]
        end_time = segment["end"]
        
        # 단어별 타임스탬프가 있는 경우 더 정확한 경계 설정
        if "words" in segment and segment["words"]:
            # 첫 단어의 시작 시간
            first_word = segment["words"][0]
            start_time = first_word["start"]
            
            # 마지막 단어의 끝 시간
            last_word = segment["words"][-1]
            end_time = last_word["end"]
        
        # 자연스러운 시작과 끝을 위한 최소한의 패딩 추가
        adjusted_start = max(0, start_time - time_padding)
        adjusted_end = end_time + time_padding
        
        try:
            # 오디오 세그먼트 추출
            audio_segment_path = extract_audio_segment(
                audio_path,
                adjusted_start,
                adjusted_end,
                padding=time_padding
            )
            
            # 오디오 길이 확인
            audio_duration = get_audio_duration(audio_segment_path)
            
            # 최소 길이 체크 (0.5초)
            if audio_duration < 0.5:
                continue
            
            # 세그먼트 정보 저장
            shorts_segments.append({
                "scene_description": text,
                "start": adjusted_start,
                "end": adjusted_end,
                "audio_duration": audio_duration,
                "audio_file": audio_segment_path,
                "segments": [{
                    "text": text,
                    "start": adjusted_start,
                    "end": adjusted_end
                }]
            })
            
        except Exception as e:
            print(f"세그먼트 처리 중 오류: {str(e)}")
            continue
    
    # 최대 10개의 세그먼트 선택 (균등한 간격으로)
    if len(shorts_segments) > 10:
        step = len(shorts_segments) / 10
        indices = [int(i * step) for i in range(10)]
        selected_segments = [shorts_segments[i] for i in indices]
    else:
        selected_segments = shorts_segments
    
    # 시간순 정렬
    selected_segments.sort(key=lambda x: x["start"])
    
    return selected_segments

def extract_audio_segment_with_padding(audio_path: str, start: float, end: float) -> str:
    """
    오디오 파일에서 특정 구간을 추출하고 앞뒤에 여유 시간을 추가합니다.
    텍스트와 오디오 싱크 문제 해결을 위한 함수
    """
    try:
        # 시작/종료 시간 검증
        if start < 0:
            start = 0
        
        # 오디오 로드
        audio = AudioSegment.from_file(audio_path)
        
        # 최소 1초 길이 보장
        if end <= start or end - start < 1.0:
            end = start + 1.0
        
        # 자르는 시간 앞/뒤로 여유 추가 (0.5초)
        padding = 500  # 밀리초 단위
        start_ms = max(0, int((start - 0.5) * 1000))  # 시작 시간 0.5초 앞당김
        end_ms = min(int((end + 0.5) * 1000), len(audio))  # 종료 시간 0.5초 연장
        
        # 오디오 구간 추출
        segment = audio[start_ms:end_ms]
        
        # 음량 정규화 (오디오 품질 향상)
        normalized_segment = match_target_amplitude(segment, -16.0)
        
        # 임시 파일 생성
        output_path = tempfile.mktemp(suffix='.mp3')
        
        # 고품질 MP3로 내보내기
        normalized_segment.export(
            output_path,
            format="mp3",
            bitrate="192k",
            parameters=["-ac", "2", "-ar", "44100", "-q:a", "2"]
        )
        
        return output_path
    except Exception as e:
        raise Exception(f"오디오 세그먼트 추출 중 오류 발생: {str(e)}")

def extract_audio_segment(audio_path: str, start: float, end: float, padding: float = 0.5) -> str:
    """
    오디오 파일에서 특정 구간을 추출합니다.
    오디오 품질과 일치도를 개선했습니다.
    
    Args:
        audio_path (str): 원본 오디오 파일 경로
        start (float): 시작 시간(초)
        end (float): 종료 시간(초)
        padding (float): 앞뒤로 추가할 여유 시간(초), 기본값 0.5초
        
    Returns:
        str: 추출된 오디오 파일 경로
    """
    try:
        # 시작/종료 시간 검증
        if start < 0:
            start = 0
        
        # 오디오 로드
        audio = AudioSegment.from_file(audio_path)
        
        # 최소 1초 길이 보장
        if end <= start or end - start < 1.0:
            end = start + 1.0
        
        # 자르는 시간 앞/뒤로 여유 추가
        start_ms = max(0, int((start - padding) * 1000))  # 시작 시간 padding초 앞당김
        end_ms = min(int((end + padding) * 1000), len(audio))  # 종료 시간 padding초 연장
        
        # 오디오 구간 추출
        segment = audio[start_ms:end_ms]
        
        # 음량 정규화 (오디오 품질 향상)
        normalized_segment = match_target_amplitude(segment, -16.0)
        
        # 임시 파일 생성
        output_path = tempfile.mktemp(suffix='.mp3')
        
        # 고품질 MP3로 내보내기
        normalized_segment.export(
            output_path,
            format="mp3",
            bitrate="192k",
            parameters=["-ac", "2", "-ar", "44100", "-q:a", "2"]
        )
        
        return output_path
    except Exception as e:
        raise Exception(f"오디오 세그먼트 추출 중 오류 발생: {str(e)}")

# extract_shorts_segments 함수 내에서 호출 코드 예시
# audio_segment = extract_audio_segment(audio_path, segment['start'], segment['end'])

# 다른 함수 내 호출 시 패딩값 조정 가능
# audio_segment = extract_audio_segment(audio_path, segment['start'], segment['end'], padding=0.3)
    
def match_target_amplitude(sound, target_dBFS):
    """
    오디오 음량을 목표 데시벨로 정규화합니다.
    """
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def process_shorts_with_verification(audio_path, interesting_part):
    """
    숏츠 세그먼트를 생성하고 오디오와 텍스트 일치 여부를 검증합니다.
    """
    # 세그먼트 추출
    segments = extract_shorts_segments(interesting_part, audio_path)
    
    # 결과 검증 및 필터링
    verified_segments = []
    
    for segment in segments:
        # 오디오 파일이 있고 길이가 적절한지 확인
        if 'audio_file' in segment and segment.get('audio_duration', 0) >= 1.0:
            # 추출된 오디오를 다시 음성 인식하여 텍스트 검증 (선택 사항)
            # 이 부분은 리소스 소모가 큼
            verified_segments.append(segment)
    
    return verified_segments


def process_shorts_audio_segments(audio_path: str, shorts_segments: list) -> list:
    """
    숏츠 세그먼트에 대한 오디오 파일을 추출하고 목록에 추가합니다.
    
    Args:
        audio_path (str): 원본 오디오 파일 경로
        shorts_segments (list): extract_shorts_segments 함수에서 반환된 세그먼트 목록
        
    Returns:
        list: 오디오 파일 경로가 추가된 세그먼트 목록
    """
    processed_segments = []
    
    for segment in shorts_segments:
        try:
            # 오디오 시간 정보 가져오기
            start_time = segment['start']
            end_time = segment['end']
            
            # 오디오 세그먼트 추출
            audio_file = extract_audio_segment(audio_path, start_time, end_time)
            
            # 세그먼트에 오디오 파일 정보 추가
            segment_copy = segment.copy()
            segment_copy['audio_file'] = audio_file
            
            processed_segments.append(segment_copy)
            
        except Exception as e:
            print(f"세그먼트 오디오 처리 중 오류 발생: {str(e)}")
    
    return processed_segments

def generate_midjourney_prompt(scene_description: str) -> str:
    """
    Generate Midjourney prompt from scene description
    """
    try:
        prompt = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                Create a detailed Midjourney prompt in English based on the scene description.
                Include these elements:
                - Character details and emotions
                - Actions or gestures
                - Background setting
                - Composition and camera angle
                - Lighting and mood
                
                Format:
                [character details], [action/expression], [background], [composition], [style], cinematic lighting, 8k, hyperrealistic
                """},
                {"role": "user", "content": f"Create a Midjourney prompt for this scene: {scene_description}"}
            ]
        )
        return prompt.choices[0].message.content
    except Exception as e:
        return "Error generating prompt"
    
def main_shorts_generation(audio_path):
    """
    메인 숏츠 생성 프로세스
    """
    try:
        # 1. 오디오 전체 음성 인식
        segments = transcribe_audio(audio_path)
        
        # 2. 흥미로운 부분 추출
        interesting_part = extract_interesting_part(segments)
        
        # 3. 숏츠 세그먼트 추출 (audio_path 파라미터 추가)
        shorts_segments = extract_shorts_segments(interesting_part, audio_path)
        
        # 또는 검증 포함 버전 사용
        # shorts_segments = process_shorts_with_verification(audio_path, interesting_part)
        
        return shorts_segments
        
    except Exception as e:
        print(f"숏츠 생성 중 오류: {str(e)}")
        return []
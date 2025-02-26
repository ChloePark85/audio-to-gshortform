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
import random
import streamlit as st

load_dotenv()

# FAL API 키 설정
os.environ['FAL_KEY'] = os.getenv('FAL_KEY')

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
    """
    # 이미지를 PIL Image로 변환
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 한글 폰트 설정 (시스템에 설치된 폰트 경로 사용)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 40)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 40)
        except:
            font = ImageFont.load_default()
    
    # 텍스트 줄바꿈 처리
    wrapper = textwrap.TextWrapper(width=30)
    text_lines = wrapper.wrap(text)
    
    # 텍스트 배경 설정
    line_height = 50
    total_text_height = len(text_lines) * line_height
    text_y = frame_height - total_text_height - 50  # 화면 하단에서 50픽셀 위
    
    for line in text_lines:
        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        
        # 텍스트 중앙 정렬 위치 계산
        x = (frame_width - text_width) // 2
        
        # 텍스트 배경 그리기
        padding = 10
        background_bbox = [
            x - padding,
            text_y - padding,
            x + text_width + padding,
            text_y + line_height
        ]
        draw.rectangle(background_bbox, fill=(0, 0, 0, 180))
        
        # 텍스트 그리기
        draw.text((x, text_y), line, font=font, fill=(255, 255, 255))
        text_y += line_height
    
    # PIL Image를 다시 OpenCV 형식으로 변환
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def _create_video_effect_sync(image_path: str, effect_type: str = "zoom_in", duration: float = 5.0, text=""):
    """
    실제 비디오 생성을 수행하는 동기 함수
    
    Args:
        image_path (str): 이미지 파일 경로
        effect_type (str): 효과 종류
        duration (float): 비디오 길이(초) - 오디오 길이에 맞춤
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("이미지를 불러올 수 없습니다.")
    
    # BGR을 RGB로 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 이미지 크기를 2배로 확대
    height, width = img.shape[:2]
    new_width = width * 2
    new_height = height * 2
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # 비디오 설정
    fps = 30
    total_frames = int(duration * fps)
    frame_size = (new_width, new_height)
    
    # 임시 비디오 파일 생성
    temp_video_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, frame_size)
    
    try:
        for frame_num in range(total_frames):
            progress = frame_num / total_frames
            frame = img.copy()
            
            if effect_type == "zoom_in":
                scale = 1.0 + (0.5 * progress)
                center_x, center_y = new_width / 2, new_height / 2
                M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
                frame = cv2.warpAffine(img, M, frame_size)
            else:  # zoom_out
                scale = 1.5 - (0.5 * progress)  # 1.5에서 1.0까지 축소
            
            # 이미지 중앙에서 확대/축소
            center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            frame = cv2.warpAffine(img, M, frame_size)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 텍스트 오버레이 추가
            if text:
                frame = add_text_overlay(frame, text, new_height, new_width)
            
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
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        return web_compatible_path
        
    except Exception as e:
        out.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise e

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

def transcribe_audio(audio_path: str) -> list:
    """
    Whisper API를 사용하여 오디오를 텍스트로 변환하고 타임스탬프를 반환합니다.
    보다 세밀한 세그먼트 분할을 위한 옵션 추가
    """
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                language="ko",
                # 보다 정확한 타임스탬프를 위한 옵션
                timestamp_granularities=["segment", "word"]
            )
        
        # 세그먼트 파싱
        segments = []
        if hasattr(transcription, 'segments'):
            for segment in transcription.segments:
                segments.append({
                    'text': segment['text'],
                    'start': float(segment['start']),
                    'end': float(segment['end']),
                    'words': segment.get('words', [])  # 단어 단위 타임스탬프가 있다면 저장
                })
        else:
            segments.append({
                'text': transcription.text,
                'start': 0.0,
                'end': get_audio_duration(audio_path)
            })
        
        return segments
    except Exception as e:
        raise Exception(f"음성 인식 중 오류 발생: {str(e)}")

def extract_interesting_part(segments):
    """
    2분(120초) 길이의 흥미로운 부분을 추출
    매번 다른 구간을 선택하기 위해 세션 상태를 활용
    """
    if not segments:
        raise ValueError("세그먼트가 비어있습니다.")
    
    # 이전에 선택된 구간들을 세션 상태에서 관리
    if 'selected_segments_history' not in st.session_state:
        st.session_state.selected_segments_history = []
    
    # 가능한 모든 2분 구간 찾기
    target_duration = 120  # 2분
    possible_segments = []
    
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
        
        if current_segments and (current_end - current_start) >= 60:  # 최소 1분 이상
            segment_info = {
                'start_idx': i,
                'segments': current_segments,
                'duration': current_end - current_start,
                'text_length': sum(len(seg['text']) for seg in current_segments)
            }
            possible_segments.append(segment_info)
    
    if not possible_segments:
        raise ValueError("적절한 길이의 구간을 찾을 수 없습니다.")
    
    # 이전에 선택되지 않은 구간 중에서 선택
    unselected_segments = [
        seg for seg in possible_segments 
        if seg['start_idx'] not in st.session_state.selected_segments_history
    ]
    
    if not unselected_segments:
        # 모든 구간이 선택된 경우 히스토리 초기화
        st.session_state.selected_segments_history = []
        unselected_segments = possible_segments
    
    # 텍스트 길이가 긴 순서로 정렬하고 상위 3개 중에서 랜덤 선택
    sorted_segments = sorted(unselected_segments, key=lambda x: x['text_length'], reverse=True)
    selected = random.choice(sorted_segments[:min(3, len(sorted_segments))])
    
    # 선택된 구간 인덱스를 히스토리에 추가
    st.session_state.selected_segments_history.append(selected['start_idx'])
    
    # 결과 생성
    result = {
        'title': f"2분 하이라이트 #{len(st.session_state.selected_segments_history)}",
        'reason': "텍스트가 풍부하고 흥미로운 2분 구간입니다.",
        'start': selected['segments'][0]['start'],
        'end': selected['segments'][-1]['end'],
        'segments': selected['segments']
    }
    
    return result

def extract_shorts_segments(interesting_part, audio_path):
    """
    음성 인식 결과에서 문장별로 정확하게 세그먼트를 추출하고
    각 문장에 맞는 오디오 파일을 생성합니다.
    텍스트와 오디오 싱크를 맞추기 위한 추가 보정 적용
    
    Args:
        interesting_part (dict): 흥미로운 부분에 대한 정보
        audio_path (str): 원본 오디오 파일 경로
        
    Returns:
        list: 오디오 파일이 포함된 문장 세그먼트 목록
    """
    segments = interesting_part['segments']
    
    # 10초 이후 세그먼트만 필터링
    filtered_segments = [seg for seg in segments if seg['start'] >= 10.0]
    if not filtered_segments and segments:
        filtered_segments = segments
    
    # 단일 문장 세그먼트 목록
    single_sentence_segments = []
    
    # 각 세그먼트를 문장 단위로 나누기
    for segment in filtered_segments:
        text = segment['text'].strip()
        if not text:
            continue
        
        # 문장 단위로 분리 (., !, ? 기준)
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in ['.', '!', '?']:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # 남은 텍스트가 있다면 처리 (마침표 추가)
        if current_sentence.strip():
            sentences.append(current_sentence.strip() + ".")
        
        # 문장이 없으면 다음 세그먼트로
        if not sentences:
            continue
        
        # 시간 보정값 (초)
        time_padding = 0.3
        
        # 여러 문장이 있는 경우 시간 비율에 따라 세그먼트 분할
        segment_duration = segment['end'] - segment['start']
        
        if len(sentences) == 1:
            # 단일 문장인 경우 원본 세그먼트 시간 사용
            start_time = segment['start'] - time_padding  # 시작 시간 앞당김
            if start_time < 0:
                start_time = 0
                
            single_sentence_segments.append({
                'scene_description': sentences[0],
                'start': start_time,
                'end': segment['end'] + time_padding,  # 종료 시간 연장
                'segments': [{
                    'text': sentences[0],
                    'start': start_time,
                    'end': segment['end'] + time_padding
                }]
            })
        else:
            # 여러 문장이 있는 경우 개별 문장의 시간 계산
            total_chars = sum(len(s) for s in sentences)
            start_time = segment['start']
            
            for idx, sentence in enumerate(sentences):
                char_ratio = len(sentence) / total_chars
                sentence_duration = segment_duration * char_ratio
                end_time = start_time + sentence_duration
                
                # 첫 문장은 시작을 앞당기고, 마지막 문장은 끝을 연장
                adjusted_start = start_time
                adjusted_end = end_time
                
                if idx == 0:
                    adjusted_start = max(0, adjusted_start - time_padding)
                
                if idx == len(sentences) - 1:
                    adjusted_end = adjusted_end + time_padding
                
                single_sentence_segments.append({
                    'scene_description': sentence,
                    'start': adjusted_start,
                    'end': adjusted_end,
                    'segments': [{
                        'text': sentence,
                        'start': adjusted_start,
                        'end': adjusted_end
                    }]
                })
                
                start_time = end_time
    
    # 시간순으로 정렬
    single_sentence_segments.sort(key=lambda x: x['start'])
    
    # 각 문장에 오디오 파일 생성하고 검증
    validated_segments = []
    
    for segment in single_sentence_segments:
        try:
            # 오디오 세그먼트 추출 (여유 있게 추출)
            audio_segment = extract_audio_segment_with_padding(audio_path, segment['start'], segment['end'])
            
            # 추출된 오디오 길이 확인
            audio_duration = get_audio_duration(audio_segment)
            
            # 너무 짧은 오디오는 제외 (0.5초 미만)
            if audio_duration < 0.5:
                continue
                
            # 오디오 파일 정보 추가
            segment['audio_file'] = audio_segment
            segment['audio_duration'] = audio_duration
            
            validated_segments.append(segment)
            
        except Exception as e:
            print(f"오디오 세그먼트 처리 오류: {str(e)}")
    
    # 10개의 세그먼트 선택 (균등한 간격으로)
    if len(validated_segments) >= 10:
        step = len(validated_segments) / 10
        indices = [int(i * step) for i in range(10)]
        results = [validated_segments[i] for i in indices]
    else:
        # 세그먼트가 10개 미만인 경우
        results = validated_segments[:]
        
        # 부족한 세그먼트 채우기
        while len(results) < 10:
            if results:
                # 마지막 세그먼트 복제
                last = results[-1].copy()
                last_duration = last['end'] - last['start']
                last['start'] = last['end'] + 0.1
                last['end'] = last['start'] + last_duration
                last['segments'][0]['start'] = last['start']
                last['segments'][0]['end'] = last['end']
                
                # 새 오디오 파일 생성
                try:
                    last['audio_file'] = extract_audio_segment_with_padding(audio_path, last['start'], last['end'])
                    last['audio_duration'] = get_audio_duration(last['audio_file'])
                    results.append(last)
                except:
                    # 오디오 추출 실패시 무한 루프 방지를 위해 건너뜀
                    break
            else:
                # 세그먼트가 하나도 없는 경우
                start_time = interesting_part['start'] + 10.0
                end_time = start_time + 5.0
                
                try:
                    dummy_audio = extract_audio_segment_with_padding(audio_path, start_time, end_time)
                    dummy = {
                        'scene_description': "내용이 없습니다.",
                        'start': start_time,
                        'end': end_time,
                        'audio_file': dummy_audio,
                        'audio_duration': get_audio_duration(dummy_audio),
                        'segments': [{
                            'text': "내용이 없습니다.",
                            'start': start_time,
                            'end': end_time
                        }]
                    }
                    results.append(dummy)
                except:
                    break
    
    return results

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
    Generate Midjourney prompt for Korean webtoon style scenes
    Maintains consistent character appearance across all scenes
    Uses 1:1 aspect ratio
    """
    try:
        # 캐릭터 기본 설정을 세션 상태로 관리
        if 'character_base_prompt' not in st.session_state:
            # 첫 실행시 캐릭터 기본 설정 생성
            character_prompt = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """
                    Create a detailed character description for a Korean webtoon.
                    Include:
                    - Age and gender
                    - Facial features
                    - Hair style and color
                    - Body type
                    - Clothing style
                    
                    Format your response as a concise prompt string.
                    """},
                    {"role": "user", "content": "Create a base character description"}
                ]
            )
            st.session_state.character_base_prompt = character_prompt.choices[0].message.content

        prompt = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"""
                Create a Midjourney prompt for a Korean webtoon style scene.
                Use this consistent character description for all scenes:
                {st.session_state.character_base_prompt}
                
                Required elements:
                - Character's emotion and action specific to the scene
                - Background setting
                - Camera angle and composition
                - Lighting and mood
                
                Required style elements:
                - Korean webtoon style, manhwa art
                - Clean lines and vibrant colors
                - 1:1 aspect ratio
                - High detail and quality
                
                Format:
                [character with emotion/action], [background], [composition], Korean webtoon style, manhwa art, clean lines, vibrant colors, square aspect ratio, highly detailed, 8k
                """},
                {"role": "user", "content": f"Create a Midjourney prompt for this scene: {scene_description}"}
            ]
        )
        
        # 프롬프트 끝에 비율 강제 지정
        prompt_text = prompt.choices[0].message.content.strip()
        if not prompt_text.endswith("--ar 1:1"):
            prompt_text += " --ar 1:1"
            
        return prompt_text
    except Exception as e:
        return "Error generating prompt --ar 1:1"
    
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
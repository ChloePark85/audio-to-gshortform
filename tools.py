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

def _create_video_effect_sync(image_path: str, effect_type: str = "zoom_in", duration: float = 5.0) -> str:
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
    """
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                language="ko"
            )
        
        # 응답 형식이 다르므로 파싱 방식 수정
        segments = []
        if hasattr(transcription, 'segments'):
            for segment in transcription.segments:
                segments.append({
                    'text': segment['text'],
                    'start': float(segment['start']),
                    'end': float(segment['end'])
                })
        else:
            # 전체 텍스트만 있는 경우
            segments.append({
                'text': transcription.text,
                'start': 0.0,
                'end': None  # 끝 시간을 알 수 없음
            })
        
        return segments
    except Exception as e:
        raise Exception(f"음성 인식 중 오류 발생: {str(e)}\n상세: {transcription if 'transcription' in locals() else 'No response'}")

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

def extract_shorts_segments(interesting_part):
    """
    문장 단위로 정확히 10개의 세그먼트를 추출
    각 세그먼트는 정확히 하나의 문장만 포함
    첫 10초 이후의 문장부터 선택
    """
    segments = interesting_part['segments']
    results = []
    
    # 10초 이후의 세그먼트부터 시작
    filtered_segments = [seg for seg in segments if seg['start'] >= 10.0]
    
    for segment in filtered_segments:
        # 세그먼트의 텍스트를 문장 단위로 분리
        text = segment['text'].strip()
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            # 문장의 길이를 기준으로 시간 비율 계산
            segment_duration = segment['end'] - segment['start']
            sentence_ratio = len(sentence) / len(text)
            sentence_duration = segment_duration * sentence_ratio
            
            sentence_start = segment['start']
            sentence_end = sentence_start + sentence_duration
            
            results.append({
                'scene_description': sentence,
                'start': sentence_start,
                'end': sentence_end,
                'segments': [{
                    'text': sentence,
                    'start': sentence_start,
                    'end': sentence_end
                }]
            })
    
    # 결과가 10개보다 적으면 마지막 세그먼트를 복제하여 채움
    while len(results) < 10:
        if results:  # 결과가 하나라도 있는 경우
            results.append(results[-1].copy())
        else:  # 결과가 하나도 없는 경우
            # 기본 세그먼트 생성
            default_segment = {
                'scene_description': "기본 세그먼트",
                'start': 0.0,
                'end': 1.0,
                'segments': [{
                    'text': "기본 세그먼트",
                    'start': 0.0,
                    'end': 1.0
                }]
            }
            results.append(default_segment)
    
    # 정확히 10개의 세그먼트 반환
    return results[:10]

def extract_audio_segment(audio_path: str, start: float, end: float) -> str:
    """
    오디오 파일에서 특정 구간을 추출합니다.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        segment = audio[start*1000:end*1000]  # milliseconds
        output_path = tempfile.mktemp(suffix='.mp3')
        segment.export(output_path, format="mp3")
        return output_path
    except Exception as e:
        raise Exception(f"오디오 세그먼트 추출 중 오류 발생: {str(e)}")
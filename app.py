import streamlit as st
from tools import _create_video_effect_sync, combine_video_audio, get_audio_duration, transcribe_audio, extract_interesting_part, extract_shorts_segments, extract_audio_segment, generate_midjourney_prompt
import tempfile
import os
from dotenv import load_dotenv
import ffmpeg
import random

load_dotenv()

def combine_all_videos(video_paths):
    """모든 비디오를 순서대로 결합"""
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
        st.error(f"비디오 결합 중 오류 발생: {str(e)}")
        return None

# Streamlit app setup
st.set_page_config(page_title="이미지 투 비디오 생성기", page_icon="🎥")
st.title("스크립트 분석 및 숏폼 생성기")

# 세션 상태 초기화 - 항상 10개로 고정
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {
        'images': [None] * 10,
        'audios': [None] * 10
    }

# 효과 리스트
effects = [
    "zoom_in", "zoom_out", 
    "pan_left_to_right", "pan_right_to_left",
    "pan_top_to_bottom", "pan_bottom_to_top",
    "rotate_clockwise", "rotate_counterclockwise",
    "ken_burns"
]

# 스크립트 분석 섹션
st.header("1. 스크립트 분석")
col1, col2 = st.columns(2)

with col1:
    script_file = st.file_uploader("스크립트 파일 업로드 (.txt)", type=['txt'])
    if script_file:
        script_content = script_file.getvalue().decode('utf-8')
        st.text_area("스크립트 내용", script_content, height=200)

with col2:
    audio_file = st.file_uploader("오디오 파일 업로드 (.mp3, .wav)", type=['mp3', 'wav'])
    if audio_file:
        st.audio(audio_file)
        # 오디오 임시 저장
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_audio.write(audio_file.getvalue())
        st.session_state['original_audio_path'] = temp_audio.name

# 스크립트와 오디오가 모두 업로드된 경우
if script_file and audio_file:
    if st.button("음성 인식 및 싱크 분석"):
        with st.spinner("음성을 텍스트로 변환하고 싱크를 분석중..."):
            try:
                # 오디오 파일 임시 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                    temp_audio.write(audio_file.getvalue())
                    audio_path = temp_audio.name
                
                # Whisper API 호출
                segments = transcribe_audio(audio_path)
                st.session_state['transcript_segments'] = segments
                
                # 임시 파일 정리
                os.unlink(audio_path)
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
                if 'audio_path' in locals():
                    try:
                        os.unlink(audio_path)
                    except:
                        pass

# 음성 인식 결과를 항상 표시
if 'transcript_segments' in st.session_state:
    st.subheader("🎙️ 음성 인식 결과")
    for segment in st.session_state['transcript_segments']:
        start_time = segment.get('start', 0)
        end_time = segment.get('end', '')
        text = segment.get('text', '')
        
        time_info = f"{start_time:.2f}s"
        if end_time:
            time_info += f" - {end_time:.2f}s"
        
        st.write(f"{time_info}: {text}")

# 흥미로운 부분 선택 섹션
col1, col2 = st.columns(2)

with col1:
    if st.button("흥미로운 부분 선택"):
        with st.spinner("흥미로운 부분을 분석중..."):
            try:
                result = extract_interesting_part(st.session_state['transcript_segments'])
                st.session_state['interesting_part'] = result
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

    # 흥미로운 부분 결과를 항상 표시
    if 'interesting_part' in st.session_state:
        result = st.session_state['interesting_part']
        st.subheader(f"📌 {result['title']}")
        st.info(f"선택 이유: {result['reason']}")
        st.write(f"전체 구간: {result['start']:.2f}초 ~ {result['end']:.2f}초")
        
        # 선택된 세그먼트들 표시
        with st.expander("전체 스크립트", expanded=True):
            for segment in result['segments']:
                st.write(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
        
        # 오디오 추출 및 재생
        if 'original_audio_path' in st.session_state:
            audio_path = extract_audio_segment(
                st.session_state['original_audio_path'],
                result['start'],
                result['end']
            )
            st.audio(audio_path)

with col2:
    if st.button("숏츠에 사용될 부분 선택"):
        if 'interesting_part' not in st.session_state:
            st.warning("먼저 '흥미로운 부분 선택'을 실행해주세요.")
        else:
            with st.spinner("숏츠용 세그먼트를 분석중..."):
                try:
                    results = extract_shorts_segments(
                        st.session_state['interesting_part'],
                        st.session_state['original_audio_path']
                    )
                    st.session_state['shorts_segments'] = results
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

    # 숏츠 시퀀스 결과를 항상 표시
    if 'shorts_segments' in st.session_state:
        st.subheader("🎬 숏츠 시퀀스")
        
        for i, result in enumerate(st.session_state['shorts_segments'], 1):
            with st.expander(f"Scene {i}: {result['scene_description']}", expanded=True):
                # 세그먼트 내용 표시
                for segment in result['segments']:
                    st.write(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
                
                # 오디오 추출 및 재생
                if 'original_audio_path' in st.session_state:
                    audio_path = extract_audio_segment(
                        st.session_state['original_audio_path'],
                        result['start'],
                        result['end']
                    )
                    st.audio(audio_path)
        
        # 적용 버튼 추가
        if st.button("숏츠 시퀀스 적용"):
            with st.spinner("숏츠 시퀀스 적용 중..."):
                # 결과가 10개 미만인 경우를 처리
                num_results = len(st.session_state['shorts_segments'])
                if num_results < 10:
                    st.warning(f"추출된 세그먼트가 {num_results}개입니다. 나머지 섹션은 비워둡니다.")
                
                # 미리 모든 프롬프트 생성
                prompts = []
                for segment in st.session_state['shorts_segments']:
                    prompt = generate_midjourney_prompt(segment['scene_description'])
                    prompts.append(prompt)
                st.session_state['midjourney_prompts'] = prompts
                
                # 한 번에 모든 오디오 세그먼트 추출
                for i in range(min(num_results, 10)):
                    audio_segment = extract_audio_segment(
                        st.session_state['original_audio_path'],
                        st.session_state['shorts_segments'][i]['start'],
                        st.session_state['shorts_segments'][i]['end']
                    )
                    st.session_state.uploaded_files['audios'][i] = audio_segment
                
                # 나머지 섹션은 None으로 초기화
                for i in range(num_results, 10):
                    st.session_state.uploaded_files['audios'][i] = None
                
                st.success(f"{num_results}개의 숏츠 시퀀스가 적용되었습니다!")

st.markdown("---")

# 10개의 섹션 생성
for i in range(10):
    st.subheader(f"{i+1}번 섹션")
    col1, col2 = st.columns(2)
    
    with col1:
        # 오디오 표시 및 장면 설명
        if st.session_state.uploaded_files['audios'][i] is not None:
            if 'shorts_segments' in st.session_state and i < len(st.session_state['shorts_segments']):
                scene = st.session_state['shorts_segments'][i]
                st.markdown(f"**🎬 장면 {i+1}**")
                st.write(f"시간: {scene['start']:.2f}s - {scene['end']:.2f}s")
                st.info(f"📝 {scene['scene_description']}")
            st.audio(st.session_state.uploaded_files['audios'][i])
        else:
            uploaded_audio = st.file_uploader(f"{i+1}번 오디오", type=['mp3', 'wav'], key=f"audio_{i}")
            if uploaded_audio:
                st.audio(uploaded_audio)
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_audio.write(uploaded_audio.getvalue())
                st.session_state.uploaded_files['audios'][i] = temp_audio.name
            
    with col2:
        # 이미지 업로드 섹션
        st.markdown("### 🖼️ 이미지 업로드")
        
        # 이미지 생성 프롬프트 표시
        if 'midjourney_prompts' in st.session_state and i < len(st.session_state['midjourney_prompts']):
            with st.expander("🎨 Image Generation Prompt", expanded=True):
                st.code(st.session_state['midjourney_prompts'][i], language='text')
        
        uploaded_image = st.file_uploader(f"{i+1}번 이미지", type=['png', 'jpg', 'jpeg'], key=f"image_{i}")
        if uploaded_image:
            st.image(uploaded_image, caption=f"{i+1}번 이미지")
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_image.write(uploaded_image.getvalue())
            st.session_state.uploaded_files['images'][i] = temp_image.name
    
    st.divider()

# 모든 파일이 업로드되었는지 확인
all_uploaded = all(st.session_state.uploaded_files['images']) and all(st.session_state.uploaded_files['audios'])

if all_uploaded:
    st.markdown("---")
    st.subheader("🎬 최종 비디오 생성")
    
    if st.button("최종 비디오 생성"):
        generated_videos = []
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # 각 비디오당 8%씩 할당 (80%)
            # 최종 결합에 20% 할당
            for i in range(10):
                progress_text.text(f"{i+1}번 비디오 생성 중...")
                progress_bar.progress(i * 8)  # 0, 8, 16, ..., 72
                
                # 오디오 길이 가져오기
                audio_duration = get_audio_duration(st.session_state.uploaded_files['audios'][i])
                
                # 랜덤 효과 선택
                random_effect = random.choice(effects)
                
                # 비디오 효과 생성
                video_path = _create_video_effect_sync(
                    st.session_state.uploaded_files['images'][i],
                    effect_type=random_effect,
                    duration=audio_duration,
                    text=st.session_state['shorts_segments'][i]['scene_description']
                )
                
                # 비디오와 오디오 결합
                final_video_path = combine_video_audio(
                    video_path,
                    st.session_state.uploaded_files['audios'][i]
                )
                
                generated_videos.append(final_video_path)
                progress_bar.progress(min(80, (i + 1) * 8))  # 최대 80%까지
            
            progress_text.text("최종 비디오 결합 중...")
            progress_bar.progress(90)  # 최종 결합 시작
            
            final_video = combine_all_videos(generated_videos)
            
            if final_video:
                progress_text.text("완료!")
                progress_bar.progress(100)  # 정확히 100%로 설정
                
                st.success("최종 비디오 생성 완료!")
                st.video(final_video)
                
                # 다운로드 버튼
                with open(final_video, 'rb') as f:
                    st.download_button(
                        label="최종 비디오 다운로드",
                        data=f,
                        file_name="final_video.mp4",
                        mime="video/mp4"
                    )
                
                # 임시 파일 정리
                for video in generated_videos:
                    if os.path.exists(video):
                        os.remove(video)
                
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            
else:
    st.info("모든 이미지와 오디오를 업로드한 후 최종 비디오를 생성할 수 있습니다.")
import streamlit as st
from crewai import Agent, Task, Crew
from tasks import get_highlight_extraction_task, execute_task_with_error_handling
from tools import extract_highlights, create_highlight_clips, generate_subtitle, summarize_text, generate_image, add_background, add_header_to_image
from agents import highlight_agent
import tempfile
import os
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

# Streamlit 앱 설정
st.set_page_config(page_title="그래픽 숏폼 generator", page_icon="🎵")
st.title("그래픽 숏폼 생성기")

# 파일 업로더 위젯
uploaded_file = st.file_uploader("오디오 파일을 선택하세요", type=['mp3', 'wav'])
tmp_file_path = None

if uploaded_file is not None:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success(f"파일이 성공적으로 업로드되었습니다: {uploaded_file.name}")
    st.audio(tmp_file_path)

    # 하이라이트 추출
    highlight_times = extract_highlights(tmp_file_path)

    if not highlight_times:
        st.error("하이라이트를 추출하지 못했습니다.")
    else:
        # 하이라이트 클립 생성
        highlight_clips = create_highlight_clips(tmp_file_path, highlight_times)
        
        for i, clip in enumerate(highlight_clips):
            start, end = highlight_times[i]
            st.write(f"하이라이트 {i+1}: {start:.2f}초 - {end:.2f}초 (길이: {(end-start):.2f}초)")
            
            # 임시 파일로 저장하고 재생
            clip_path = f"temp_clip_{i}.mp3"
            clip.export(clip_path, format="mp3")
            st.audio(clip_path)
            
            # 자막 생성
            subtitle = generate_subtitle(clip)
            st.write("자막:")
            st.write(subtitle)
            
            # 자막 요약
            summary = summarize_text(subtitle)
            st.write("요약:")
            st.write(summary)
            
            # 이미지 생성
            image_prompt = f"""Create a stylish illustration featuring an attractive character based on the following scene: {subtitle}. 
Key points:
- The character should be visually appealing and well-designed
- Use a sophisticated illustration style
- Do not include any speech bubbles or text in the image
- The image should be suitable for a short-form video thumbnail
- Focus on creating a visually striking and engaging scene
"""
            image = generate_image(image_prompt)

            # 이미지에 검은색 배경 추가 (2:3 비율로)
            image_with_background = add_background(image)

            # 이미지에 헤더 추가
            image_with_header = add_header_to_image(image_with_background, summary)

            # 이미지 표시
            st.image(image_with_header, caption=f"하이라이트 {i+1} 이미지")

            # 임시 파일 삭제
            os.remove(clip_path)

            st.write("---")

    # 임시 파일 삭제
    if tmp_file_path:
        os.unlink(tmp_file_path)
        tmp_file_path = None

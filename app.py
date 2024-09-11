import io
import streamlit as st
from crewai import Crew
from agents import highlight_agent, subtitle_agent, summarization_agent, image_generation_agent
from tasks import (
    get_highlight_extraction_task, 
    get_subtitle_generation_task, 
    get_text_summarization_task, 
    get_image_generation_task, 
    get_image_header_addition_task
)
from tools import extract_highlights, create_highlight_clips, add_background, generate_subtitle, summarize_text, generate_image, add_header_to_image
import tempfile
import os
from PIL import Image
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip

from dotenv import load_dotenv
load_dotenv()

# Streamlit app setup
st.set_page_config(page_title="그래픽 숏폼 generator", page_icon="🎵")
st.title("그래픽 숏폼 생성기")

# File uploader widget
uploaded_file = st.file_uploader("오디오 파일을 선택하세요", type=['mp3', 'wav'])
tmp_file_path = None

if uploaded_file is not None:
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success(f"파일이 성공적으로 업로드되었습니다: {uploaded_file.name}")
    st.audio(tmp_file_path)

    # Extract highlights
    with st.spinner('하이라이트 추출 중...'):
        highlight_times = extract_highlights(tmp_file_path)

    if not highlight_times:
        st.error("하이라이트를 추출하지 못했습니다.")
    else:
        # Create highlight clips
        with st.spinner('하이라이트 클립 생성 중...'):
            highlight_clips = create_highlight_clips(tmp_file_path, highlight_times)
        
        for i, clip in enumerate(highlight_clips):
            start, end = highlight_times[i]
            st.write(f"하이라이트 {i+1}: {start:.2f}초 - {end:.2f}초 (길이: {(end-start):.2f}초)")
            
            # Save and play temporary file
            clip_path = f"temp_clip_{i}.mp3"
            clip.export(clip_path, format="mp3")
            st.audio(clip_path)
            
            # Generate subtitle
            with st.spinner('자막 생성 중...'):
                subtitle = generate_subtitle(clip)
            st.write("자막:")
            st.write(subtitle)
            
            # Summarize subtitle
            with st.spinner('요약 생성 중...'):
                summary = summarize_text(subtitle)
            st.write("요약:")
            st.write(summary)
            
            # Generate image
            with st.spinner('이미지 생성 중...'):
                image = generate_image(subtitle)
            if isinstance(image, Image.Image):
                # Add black background to image (2:3 ratio)
                image_with_background = add_background(image)

                # Add header to image
                image_with_header = add_header_to_image(image_with_background, summary)

                # Display image
                st.image(image_with_header, caption=f"하이라이트 {i+1} 이미지")

                # Create video
                with st.spinner('비디오 생성 중...'):
                    image_path = f"temp_image_{i}.png"
                    image_with_header.save(image_path)
                    
                    video = ImageClip(image_path).set_duration(end-start)
                    audio = AudioFileClip(clip_path)
                    final_clip = CompositeVideoClip([video]).set_audio(audio)
                    
                    video_path = f"temp_video_{i}.mp4"
                    final_clip.write_videofile(video_path, fps=24)

                # Display video
                st.video(video_path)

                with open(video_path, "rb") as file:
                    btn = st.download_button(
                        label=f"하이라이트 {i+1} 비디오 다운로드",
                        data=file,
                        file_name=f"highlight_{i+1}.mp4",
                        mime="video/mp4"
                    )

                # Delete temporary files
                os.remove(image_path)
                os.remove(video_path)
            else:
                st.error("이미지 생성에 실패했습니다.")

            # Delete temporary audio file
            os.remove(clip_path)

            st.write("---")

    # Delete temporary file
    if tmp_file_path:
        os.unlink(tmp_file_path)
        tmp_file_path = None

st.write("모든 작업이 완료되었습니다.")
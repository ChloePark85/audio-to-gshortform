import os
from dotenv import load_dotenv
import pandas as pd

# .env 파일에서 환경 변수 로드
load_dotenv()

# 필요한 환경 변수 설정
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
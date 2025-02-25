Kling 1.6 Image to Video API는 이미지를 입력으로 받아 비디오 클립을 생성하는 기능을 제공합니다. 아래는 해당 API의 사용법을 마크다운 형식으로 정리한 내용입니다.

Kling 1.6 Image to Video API 문서

목차
	1.	API 호출
	•	클라이언트 설치
	•	API 키 설정
	•	요청 제출
	2.	인증
	•	API 키
	3.	큐
	•	요청 제출
	•	요청 상태 확인
	•	결과 가져오기
	4.	파일
	•	Data URI (base64)
	•	호스팅된 파일(URL)
	•	파일 업로드
	5.	스키마
	•	입력
	•	출력
	•	기타

API 호출

클라이언트 설치

클라이언트는 모델 API와 상호 작용하기 위한 편리한 방법을 제공합니다.

npm install --save @fal-ai/client

@fal-ai/client로 마이그레이션

기존의 @fal-ai/serverless-client 패키지는 더 이상 사용되지 않으며, @fal-ai/client로 마이그레이션해야 합니다. 자세한 내용은 마이그레이션 가이드에서 확인할 수 있습니다.

API 키 설정

런타임 환경에서 FAL_KEY를 환경 변수로 설정해야 합니다.

export FAL_KEY="YOUR_API_KEY"

요청 제출

클라이언트 API는 요청 상태 업데이트를 처리하고, 요청이 완료되면 결과를 반환합니다.

import { fal } from "@fal-ai/client";

const result = await fal.subscribe("fal-ai/kling-video/v1.6/pro/image-to-video", {
  input: {
    prompt: "도쿄 거리에서 따뜻한 네온과 애니메이션 도시 간판으로 가득한 길을 걷는 스타일리시한 여성. 그녀는 검은 가죽 재킷, 긴 빨간 드레스, 검은 부츠를 착용하고, 검은 가방을 들고 있습니다.",
    image_url: "https://fal.media/files/panda/TuXlMwArpQcdYNCLAEM8K.webp"
  },
  logs: true,
  onQueueUpdate: (update) => {
    if (update.status === "IN_PROGRESS") {
      update.logs.map((log) => log.message).forEach(console.log);
    }
  },
});
console.log(result.data);
console.log(result.requestId);

인증

API는 API 키를 사용하여 인증을 수행합니다. 가능한 경우 런타임 환경에서 FAL_KEY 환경 변수를 설정하는 것이 좋습니다.

API 키

환경 변수 설정이 불가능한 경우, 클라이언트 설정에서 API 키를 수동으로 지정할 수 있습니다.

import { fal } from "@fal-ai/client";

fal.config({
  credentials: "YOUR_FAL_KEY"
});

API 키 보호

클라이언트 측(예: 브라우저, 모바일 앱, GUI 애플리케이션)에서 코드를 실행할 때는 FAL_KEY를 노출하지 않도록 주의해야 합니다. 대신 서버 측 프록시를 사용하여 API에 요청을 보내는 것이 좋습니다. 자세한 내용은 서버 측 통합 가이드에서 확인할 수 있습니다.

큐

장시간 실행되는 요청

학습 작업이나 느린 추론 시간을 가진 모델과 같은 장시간 실행되는 요청의 경우, 큐 상태를 확인하고 웹훅을 사용하는 것이 좋습니다.

요청 제출

클라이언트 API는 모델에 대한 요청을 제출하는 편리한 방법을 제공합니다.

import { fal } from "@fal-ai/client";

const { request_id } = await fal.queue.submit("fal-ai/kling-video/v1.6/pro/image-to-video", {
  input: {
    prompt: "도쿄 거리에서 따뜻한 네온과 애니메이션 도시 간판으로 가득한 길을 걷는 스타일리시한 여성. 그녀는 검은 가죽 재킷, 긴 빨간 드레스, 검은 부츠를 착용하고, 검은 가방을 들고 있습니다.",
    image_url: "https://fal.media/files/panda/TuXlMwArpQcdYNCLAEM8K.webp"
  },
  webhookUrl: "https://optional.webhook.url/for/results",
});

요청 상태 확인

요청이 완료되었는지 또는 진행 중인지 확인하기 위해 요청의 상태를 가져올 수 있습니다.

import { fal } from "@fal-ai/client";

const status = await fal.queue.status("fal-ai/kling-video/v1.6/pro/image-to-video", {
  requestId: "764cabcf-b745-4b3e-ae38-1200304cf45b",
  logs: true,
});

결과 가져오기

요청이 완료되면 결과를 가져올 수 있습니다. 예상되는 결과 형식은 출력 스키마를 참조하세요.

import { fal } from "@fal-ai/client";

const result = await fal.queue.result("fal-ai/kling-video/v1.6/pro/image-to-video", {
  requestId: "764cabcf-b745-4b3e-ae38-1200304cf45b"
});
console.log(result.data);
console.log(result.requestId);

파일

일부 API 속성은 파일 URL을 입력으로 받습니다. 이 경우 직접 URL을 제공하거나 Base64 데이터 URI를 사용할 수 있습니다.

Data URI (base64)

파일 입력으로 Base64 데이터 URI를 전달할 수 있으며, API는 파일 디코딩을 처리합니다.

 
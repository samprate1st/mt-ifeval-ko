# MT-IFEval-Ko

Multi-turn Instruction Following 평가를 위한 도구입니다. 
Langfuse와 LangChain을 활용하여 평가결과를 모니터링 합니다.
본 과제는 한국어 데이터셋을 개발하기 편리한 환경 구축을 목적으로 합니다.
개발된 한국어 데이터셋은 추후 공개 예정입니다.

## 주요 기능

1. Langfuse 서버에서 데이터셋 다운로드
2. LangChain/LangGraph 기반 멀티턴 대화 생성
3. IFEval을 사용하여 strict/loose 평가 수행
4. 평가 결과를 Langfuse 서버에 기록

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/samprat1st/mt-ifeval-ko.git
cd mt-ifeval-ko
```

2. 가상 환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

3. 의존성 설치
```bash
pip install -e .
```

4. 환경 변수 설정
`.env` 파일을 수정하여 필요한 API 키와 환경 설정을 입력하세요.

## 프로젝트 구조

```
mt-ifeval-ko/
├── LICENSE.txt                    # 라이선스 파일
├── README.md                      # 프로젝트 문서
├── pyproject.toml                 # Python 프로젝트 설정
├── requirements.txt               # Python 의존성 목록
├── run_eval.sh                    # 평가 실행 스크립트
├── upload_dataset.sh              # 데이터셋 업로드 스크립트
├── data/                          # 데이터 파일 디렉토리
│   └── multiIF_20241018_English.jsonl
├── doc/                           # 문서 디렉토리
├── gradio/                        # Gradio 웹 인터페이스
│   ├── app.py                     # Gradio 앱
│   └── requirements.txt           # Gradio 의존성
├── ifeval/                        # 핵심 평가 라이브러리
│   ├── __init__.py
│   ├── evaluation_lib.py          # 평가 라이브러리
│   ├── evaluation_main.py         # 메인 평가 모듈
│   ├── instructions_registry.py   # 지시사항 레지스트리
│   ├── instruction_utils/         # 지시사항 유틸리티
│   │   ├── en_instructions_util.py
│   │   └── ko_instructions_util.py
│   └── instructions/              # 지시사항 정의
│       ├── en_instructions.py
│       └── ko_instructions.py
├── output/                        # 평가 결과 출력 디렉토리
├── scripts/                       # 실행 스크립트
│   ├── ifeval_langchain_with_langfuse.py
│   ├── ifeval_langgraph_with_langfuse.py
│   └── langfuse_dataset_upload.py
└── temp/                          # 임시 파일 디렉토리
```

## 사용 방법

### 평가 실행

```bash
./run_eval.sh
```

`run_eval.sh` 파일을 수정하여 평가할 데이터셋, 모델, 온도 등의 설정을 변경할 수 있습니다.

### 주요 설정 옵션

- `DATASET`: 평가할 데이터셋 이름
- `MODEL`: 사용할 LLM 모델 이름
- `TEMPERATURE`: 생성 온도
- `LIMIT`: 평가할 최대 아이템 수
- `PARALLEL`: 병렬 처리 사용 여부
- `WORKERS`: 병렬 처리 작업자 수

### Gradio 웹 인터페이스

```bash
cd gradio
pip install -r requirements.txt
python app.py
```

### 데이터셋 업로드

```bash
./upload_dataset.sh
```
현재는 영어 데이터셋 (by Meta)만 공유되어 있습니다.
한글 버전은 작업 중입니다.

## 라이선스

MIT

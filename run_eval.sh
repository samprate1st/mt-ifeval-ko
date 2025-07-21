#!/bin/bash

# Langfuse와 LangChain을 활용한 멀티턴 Instruction Following 평가 실행 스크립트


# 데이터셋 이름 (필수)
#DATASET="Telco_MultiIF_Korean"
#DATASET="General_MultiIF_Korean"
DATASET="General_MultiIF_English"

# LLM 모델 이름
#MODEL="gpt-4o"
#MODEL="gpt-4.1"
#MODEL="gpt-4.1-mini"
MODEL="gpt-4.1-nano"

# safe_model_name 은 "/" 와 ":" 를 "_"로 변경한 문자열이다.
safe_model_name=$(echo "$MODEL" | sed 's/\//_/g' | sed 's/:/_/g')


# 실행 이름 (기본값: 타임스탬프 기반)
RUN_NAME="${safe_model_name}-${DATASET}-$(date +%H%M%S)"


# 생성 온도
TEMPERATURE="0.6"

# 평가할 최대 아이템 수 (비워두면 전체 데이터셋 평가)
LIMIT="1000"

# 상세 로그 출력 여부 (true/false)
VERBOSE="false"

# 병렬 처리 사용 여부 (true/false)
# 이제 ThreadPoolExecutor와 tqdm을 사용한 병렬 처리가 구현되었습니다.
PARALLEL="true"

# 병렬 처리 작업자 수
WORKERS="4"

# 자동 실행 모드 (true/false)
# true로 설정하면 확인 없이 바로 실행됩니다.
AUTO_RUN="true"

# ===== 실행 영역 (수정 불필요) =====

# 스크립트 디렉토리 설정
WORK_DIR="$(pwd)"
SCRIPT_DIR="$WORK_DIR/scripts"

# 출력 디렉토리 설정
OUTPUT_DIR="$WORK_DIR/output/langfuse/$safe_model_name/$DATASET/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"
OUTPUT="$OUTPUT_DIR/${DATASET}_results.json"

# 명령어 구성
CMD="python $SCRIPT_DIR/ifeval_langchain_with_langfuse.py --dataset $DATASET --run-name $RUN_NAME --output $OUTPUT"

# 모델 설정
if [ ! -z "$MODEL" ]; then
    CMD="$CMD --model $MODEL"
fi

# 온도 설정
if [ ! -z "$TEMPERATURE" ]; then
    CMD="$CMD --temperature $TEMPERATURE"
fi

# 아이템 수 제한
if [ ! -z "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# 상세 로그 설정
if [ "$VERBOSE" = "true" ]; then
    CMD="$CMD --verbose"
fi

# 병렬 처리 설정
if [ "$PARALLEL" = "true" ]; then
    CMD="$CMD --parallel"
    if [ ! -z "$WORKERS" ]; then
        CMD="$CMD --workers $WORKERS"
    fi
fi

# 설정 정보 출력
echo "===== 평가 설정 ====="
echo "데이터셋: $DATASET"
echo "실행 이름: $RUN_NAME"
echo "모델: $MODEL"
echo "온도: $TEMPERATURE"
echo "아이템 제한: $LIMIT"
echo "상세 로그: $VERBOSE"
echo "병렬 처리: $PARALLEL"
if [ "$PARALLEL" = "true" ]; then
    echo "작업자 수: $WORKERS"
fi
echo "결과 저장 위치: $OUTPUT"
echo "===================="
echo ""

# 실행 확인 (자동 실행 모드가 아닌 경우)
if [ "$AUTO_RUN" != "true" ]; then
    echo "위 설정으로 평가를 실행하시겠습니까? (y/n)"
    read -p "> " confirm

    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "평가가 취소되었습니다."
        exit 0
    fi
fi

# 명령어 출력
echo ""
echo "실행 명령어:"
echo "$CMD"
echo ""

# 명령어 실행
eval $CMD

# 결과 확인
if [ $? -eq 0 ]; then
    echo -e "\n✅ 평가 완료. 결과 저장 위치: $OUTPUT"
else
    echo -e "\n❌ 평가 실행 중 오류가 발생했습니다."
    exit 1
fi 

# 데이터 셋 준비

## 데이텃 업로드
DATASET_NAME="General_MultiIF_English"
DATASET_FILE="multiIF_20241018_English.jsonl"
python scripts/langfuse_dataset_upload.py -n $DATASET_NAME -f $DATASET_FILE


## 데이텃 조회
#python scripts/langfuse_dataset_upload.py -n $DATASET_NAME -s


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Langfuse 데이터셋 업로드 스크립트

이 스크립트는 다음 기능을 수행합니다:
1. Langfuse 서버 설정 및 연결
2. 데이터셋 생성 및 업로드
3. 데이터셋 중복 항목 제거
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pprint import pprint
from dotenv import load_dotenv
import uuid

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
print(f"SCRIPT_DIR: {SCRIPT_DIR}")
WORK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
print(f"WORK_DIR: {WORK_DIR}")
ENV_FILE = os.path.join(WORK_DIR, ".env.params")
print(f"ENV_FILE: {ENV_FILE}")
load_dotenv(ENV_FILE)

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.abspath('..'))

# Langfuse 관련 라이브러리 가져오기
try:
    from langfuse import get_client, Langfuse
    from langfuse.langchain import CallbackHandler
except ImportError:
    print("Error: Langfuse 라이브러리를 설치해주세요. (pip install langfuse)")
    sys.exit(1)

def setup_langfuse():
    """Langfuse 서버 설정 및 클라이언트 초기화"""
    # 환경 변수 로드
    env_file = os.getenv("ENV_FILE", "../.env.params")
    load_dotenv(env_file)
    
    # 필수 환경 변수 확인
    required_vars = ["LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: 다음 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        print("환경 변수를 .env.params 파일에 설정하거나 직접 환경에 설정해주세요.")
        sys.exit(1)
    
    # Langfuse 클라이언트 초기화
    try:
        langfuse_client = get_client()
        auth_status = langfuse_client.auth_check()
        print(f"Langfuse 인증 상태: {auth_status}")
        
        if not auth_status:
            print("Error: Langfuse 인증에 실패했습니다. API 키를 확인해주세요.")
            sys.exit(1)
            
        return langfuse_client
    except Exception as e:
        print(f"Error: Langfuse 클라이언트 초기화 중 오류가 발생했습니다: {str(e)}")
        sys.exit(1)

def load_dataset(file_path: str) -> pd.DataFrame:
    """CSV 또는 JSONL 파일에서 데이터셋 로드"""
    print(f"데이터셋 파일 로드 중: {file_path}")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.jsonl'):
        # JSONL 파일 처리
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. CSV 또는 JSONL 파일만 지원합니다.")
    
    print(f"데이터셋 로드 완료: {len(df)} 행, {len(df.columns)} 열")
    print(f"데이터셋 컬럼: {df.columns.tolist()}")
    
    # 데이터 정제 수행
    print("데이터셋 정제 시작...")
    refined_df = refine_dataset_in_dataframe(df)
    
    return refined_df

def create_benchmark_dataset(langfuse_client, name: str, data: List[Dict]) -> Any:
    """평가용 데이터셋 생성 및 업로드"""
    print(f"데이터셋 '{name}' 생성 중...")
    
    # 만일 동일한 이름의 데이터셋이 이미 존재한다면 skip 하고 본 함수를 종료
    # 만일 동일한 이름의 데이터셋이 없다면, 새로운 데이터셋을 생성
    try:
        dataset = langfuse_client.get_dataset(name=name)
        print(f"데이터셋 '{name}'이(가) 이미 존재합니다. 기존 데이터셋을 사용합니다.")
        return dataset
    except Exception as e:
        if "Dataset not found" in str(e):
            try:
                dataset = langfuse_client.create_dataset(name=name)
                print(f"새 데이터셋 '{name}' 생성 완료")
            except Exception as create_error:
                print(f"Error: 데이터셋 생성 중 오류가 발생했습니다: {str(create_error)}")
                sys.exit(1)
        else:
            print(f"Error: 데이터셋 확인 중 오류가 발생했습니다: {str(e)}")
            sys.exit(1)
    
    # 데이터셋 아이템 추가
    print(f"데이터셋에 {len(data)} 개의 항목 추가 중...")
    success_count = 0
    error_count = 0
    
    for idx, item in enumerate(data, 1):
        try:
            langfuse_client.create_dataset_item(
                dataset_name=name,
                input=item.get("input", ""),
                expected_output=item.get("output", ""),
                metadata=item.get("metadata", ""),
            )
            success_count += 1
            
            if idx % 50 == 0:
                print(f"진행 중: {idx}/{len(data)} 항목 처리됨")
                
        except Exception as e:
            error_count += 1
            print(f"Warning: 항목 추가 중 오류 발생 (항목 {idx}): {str(e)}")
    
    print(f"데이터셋 업로드 완료: 성공 {success_count}개, 실패 {error_count}개")
    return dataset

def remove_dataset_duplicates(langfuse_client, dataset_name: str) -> Any:
    """데이터셋 내 중복 아이템 제거"""
    print(f"데이터셋 '{dataset_name}'의 중복 항목 제거 중...")
    
    try:
        dataset = langfuse_client.get_dataset(name=dataset_name)
        print(f"데이터셋 크기: {len(dataset.items)} 항목")
        
        # 데이터셋 내 아이템 중복 삭제
        # 중복 여부는 metadata.key 기준으로 확인
        dataset_items = dataset.items
        
        # metadata의 "key" 값이 중복인 경우, 첫 번째 항목만 남기고 나머지는 삭제
        key_to_items = defaultdict(list)
        for item in dataset_items:
            key = item.metadata.get("key")
            key_to_items[key].append(item)
        
        # 중복 key를 가진 아이템 식별 및 삭제
        deleted_count = 0
        for key, items in key_to_items.items():
            if len(items) > 1:
                # 첫 번째만 남기고 나머지 삭제
                for dup_item in items[1:]:
                    print(f"중복 key 발견 및 삭제: {key}")
                    dup_item.delete()  # 아이템 자체의 delete() 메서드 사용
                    deleted_count += 1
        
        # 중복 제거 후 남은 아이템 개수 확인
        dataset = langfuse_client.get_dataset(name=dataset_name)
        print(f"중복 제거 완료: {deleted_count}개 삭제됨")
        print(f"중복 제거 후 데이터셋 크기: {len(dataset.items)} 항목")
        
        return dataset
    except Exception as e:
        print(f"Error: 중복 제거 중 오류가 발생했습니다: {str(e)}")
        return None

def delete_benchmark_dataset(langfuse_client, dataset_name: str) -> bool:
    """지정된 이름의 데이터셋을 Langfuse 서버에서 삭제"""
    print(f"데이터셋 '{dataset_name}' 삭제 시도 중...")
    
    try:
        # 데이터셋 존재 확인
        dataset = langfuse_client.get_dataset(name=dataset_name)
        print(f"데이터셋 '{dataset_name}'을(를) 찾았습니다. 항목 수: {len(dataset.items)}")
        
        # 사용자 확인 요청
        confirm = input(f"정말로 데이터셋 '{dataset_name}'을(를) 삭제하시겠습니까? (y/n): ")
        if confirm.lower() != 'y':
            print("삭제 작업이 취소되었습니다.")
            return False
        
        # 데이터셋 내 모든 아이템 삭제
        print(f"데이터셋 '{dataset_name}'의 모든 항목 삭제 중...")
        items_count = len(dataset.items)
        for idx, item in enumerate(dataset.items, 1):
            try:
                item.delete()
                if idx % 50 == 0 or idx == items_count:
                    print(f"진행 중: {idx}/{items_count} 항목 삭제됨")
            except Exception as e:
                print(f"Warning: 항목 삭제 중 오류 발생 (항목 ID: {item.id}): {str(e)}")
        
        # 데이터셋 자체 삭제
        try:
            langfuse_client.delete_dataset(name=dataset_name)
            print(f"데이터셋 '{dataset_name}' 삭제 완료")
            return True
        except Exception as e:
            print(f"Error: 데이터셋 삭제 중 오류가 발생했습니다: {str(e)}")
            return False
            
    except Exception as e:
        if "Dataset not found" in str(e):
            print(f"Error: 데이터셋 '{dataset_name}'을(를) 찾을 수 없습니다.")
        else:
            print(f"Error: 데이터셋 확인 중 오류가 발생했습니다: {str(e)}")
        return False


def refine_dataset_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임의 각 row 별 데이터 상태를 확인하고 정제하여 업데이트 하고 리턴"""
    
    def clean_string_with_json(value_str: str) -> str:
        """문자열을 JSON 파싱을 통해 정제"""
        if not isinstance(value_str, str) or not value_str.strip():
            return value_str
        
        # 정제 시도 단계
        attempts = [
            # 1. 원본 그대로 파싱 시도
            lambda s: s,
            # 2. 연속된 백슬래시 줄이기 시도
            lambda s: s.replace('\\\\', '\\'),
            # 3. 이스케이프된 따옴표 처리 시도
            lambda s: s.replace('\\"', '"'),
            # 4. 백슬래시와 따옴표 모두 처리 시도
            lambda s: s.replace('\\\\', '\\').replace('\\"', '"'),
            # 5. 역슬래시 제거 후 시도
            lambda s: s.replace('\\', '')
        ]
        
        original_str = value_str
        
        for attempt_func in attempts:
            try:
                # 정제 함수 적용
                cleaned_str = attempt_func(value_str)
                
                # JSON 파싱 시도
                parsed = json.loads(cleaned_str)
                
                # 성공하면 다시 문자열로 변환하여 반환 (한글 유지)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
        
        # 모든 시도 실패 시 원본 반환
        print(f"Warning: JSON 정제 실패: {original_str[:50]}...")
        return original_str
    
    def clean_prompt(prompt_str: str) -> str:
        """프롬프트 문자열 정제"""
        if not isinstance(prompt_str, str):
            return prompt_str
        
        # JSON 파싱 시도
        try:
            return clean_string_with_json(prompt_str)
        except Exception:
            # JSON 파싱 실패 시 원본 반환
            return prompt_str
    
    def clean_instruction_id_list(id_list_str: str) -> str:
        """instruction_id_list 문자열 정제"""
        if not isinstance(id_list_str, str):
            return id_list_str
        
        # JSON 파싱 시도
        try:
            return clean_string_with_json(id_list_str)
        except Exception:
            # JSON 파싱 실패 시 원본 반환
            return id_list_str
    
    def clean_kwargs(kwargs_str: str) -> str:
        """kwargs 문자열 정제"""
        if not isinstance(kwargs_str, str):
            return kwargs_str
        
        # JSON 파싱 시도
        try:
            return clean_string_with_json(kwargs_str)
        except Exception:
            # JSON 파싱 실패 시 원본 반환
            return kwargs_str
    
    # 데이터프레임 복사본 생성
    refined_df = df.copy()
    
    # 각 열에 대해 정제 함수 적용
    for column in refined_df.columns:
        if column in ['turns', 'responses', 'language', 'key', 'turn_index']:
            # 정제가 필요 없는 항목
            continue
        elif any(column.startswith(f'turn_{i}_prompt') for i in range(10)):
            # 정제 케이스 1: turn_{digit}_prompt
            refined_df[column] = refined_df[column].apply(clean_prompt)
        elif any(column.startswith(f'turn_{i}_instruction_id_list') for i in range(10)):
            # 정제 케이스 2: turn_{digit}_instruction_id_list
            refined_df[column] = refined_df[column].apply(clean_instruction_id_list)
        elif any(column.startswith(f'turn_{i}_kwargs') for i in range(10)):
            # 정제 케이스 3: turn_{digit}_kwargs
            refined_df[column] = refined_df[column].apply(clean_kwargs)
    
    print(f"데이터프레임 정제 완료: {len(refined_df)} 행")
    return refined_df



def process_dataframe_to_langfuse_format(df: pd.DataFrame, language: str = "Korean") -> List[Dict]:
    """데이터프레임을 Langfuse 데이터셋 형식으로 변환"""
    data = []
    
    for _, row in df.iterrows():
        # 각 행을 처리하여 Langfuse 데이터셋 형식으로 변환
        item = {}
        
        # 메타데이터 설정
        metadata = {}
        metadata["key"] = row["key"]            
        metadata["language"] = row["language"]
            
        # 입력 데이터 구성
        input_data = {}
        
        max_turns = 3  # 기본값
        for turn_idx in range(1, max_turns + 1):
        
            turn_key = f"turn_{turn_idx}"
            turn_data = {}
            
            # 프롬프트
            prompt_key = f"{turn_key}_prompt"
            if "" != row[prompt_key]:
                turn_data["prompt"] = row[prompt_key]
            
            inst_key = f"{turn_key}_instruction_id_list"
            if "" != row[inst_key]:
                turn_data["instruction_id_list"] = row[inst_key]
            
            # kwargs
            kwargs_key = f"{turn_key}_kwargs"
            if "" != row[kwargs_key]:
                turn_data["kwargs"] = row[kwargs_key]
            
            # 턴 데이터가 있으면 입력에 추가
            if turn_data:
                input_data[turn_key] = turn_data
        
        
        # 항목 구성
        item["input"] = input_data
        item["expected_output"] = None
        item["metadata"] = metadata
        
        data.append(item)
    
    # 변환된 데이터 출력 (디버깅용)
    if data:
        print("\n샘플 항목 정보:")
        sample = data[0]
        print(f"입력:\n{json.dumps(sample['input'], ensure_ascii=False, indent=2)}")
        print(f"기대 출력: {sample.get('expected_output', "")}")
        print(f"메타데이터:\n{json.dumps(sample['metadata'], ensure_ascii=False, indent=2)}")
    
    return data

def show_dataset_stats(langfuse_client, dataset_name: str) -> None:
    """지정된 이름의 데이터셋 통계 정보와 샘플을 출력"""
    print(f"데이터셋 '{dataset_name}' 정보 조회 중...")
    
    try:
        # 데이터셋 존재 확인
        dataset = langfuse_client.get_dataset(name=dataset_name)
        items = dataset.items
        items_count = len(items)
        
        if items_count == 0:
            print(f"데이터셋 '{dataset_name}'에 항목이 없습니다.")
            return
        
        print(f"\n===== 데이터셋 '{dataset_name}' 통계 정보 =====")
        print(f"총 항목 수: {items_count}")
        
        # 메타데이터 키 분석
        metadata_keys = set()
        input_types = defaultdict(int)
        
        for item in items:
            if item.metadata:
                metadata_keys.update(item.metadata.keys())
            
            # 입력 타입 분석
            input_type = "unknown"
            if isinstance(item.input, dict):
                input_type = "dict"
            elif isinstance(item.input, str):
                input_type = "string"
            elif isinstance(item.input, list):
                input_type = "list"
            input_types[input_type] += 1
        
        print(f"메타데이터 키: {', '.join(sorted(metadata_keys)) if metadata_keys else '없음'}")
        print(f"입력 데이터 타입 분포: {dict(input_types)}")
        
        # 샘플 데이터 출력 (최대 2개)
        print("\n===== 샘플 데이터 =====")
        for i, item in enumerate(items[:2]):
            print(f"\n[샘플 {i+1}]")
            print(f"ID: {item.id}")
            
            # 입력 데이터 출력 (한글 가독성 향상)
            if isinstance(item.input, dict):
                # 딕셔너리를 JSON 문자열로 변환 (한글 유지)
                input_str = json.dumps(item.input, ensure_ascii=False, indent=2)
                print(f"입력:\n{input_str}")
            elif isinstance(item.input, str) and len(item.input) > 200:
                print(f"입력: {item.input[:200]}...")
            else:
                print(f"입력: {item.input}")
                
            if item.metadata:
                # 메타데이터도 한글 유지하여 출력
                metadata_str = json.dumps(item.metadata, ensure_ascii=False, indent=2)
                print(f"메타데이터:\n{metadata_str}")
        
    except Exception as e:
        print(f"오류: 데이터셋 정보를 가져오는 중 문제가 발생했습니다: {str(e)}")
        return

def list_datasets(langfuse_client):
    """사용 가능한 데이터셋 목록 표시"""
    print("사용 가능한 데이터셋 목록 가져오는 중...")
    
    try:
        # API 엔드포인트 구성
        langfuse_host = os.getenv("LANGFUSE_HOST").rstrip('/')
        api_url = f"{langfuse_host}/api/v1/datasets"
        
        # 인증 헤더 설정
        auth_str = f"{os.getenv('LANGFUSE_PUBLIC_KEY')}:{os.getenv('LANGFUSE_SECRET_KEY')}"
        import base64
        auth_bytes = auth_str.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth_b64}"
        }
        
        # API 호출
        import requests
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            datasets = data.get("datasets", [])
            
            if not datasets:
                print("사용 가능한 데이터셋이 없습니다.")
                return
            
            print(f"\n총 {len(datasets)}개의 데이터셋 발견:")
            print("-" * 80)
            print(f"{'이름':<30} {'항목 수':<10} {'생성 일시':<20} {'ID':<20}")
            print("-" * 80)
            
            for dataset in datasets:
                # 데이터셋 정보 출력
                name = dataset.get("name", "N/A")
                item_count = dataset.get("itemCount", "N/A")
                created_at = dataset.get("createdAt", "N/A")
                dataset_id = dataset.get("id", "N/A")
                
                print(f"{name:<30} {item_count:<10} {created_at:<20} {dataset_id}")
            
            print("-" * 80)
        else:
            print(f"API 호출 실패: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"데이터셋 목록을 가져오는 중 오류가 발생했습니다: {str(e)}")
        return None

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Langfuse 데이터셋 업로드 스크립트")
    parser.add_argument("--file", "-f", help="업로드할 데이터셋 파일 경로 (CSV 또는 JSONL)")
    parser.add_argument("--name", "-n", help="Langfuse에 생성할/삭제할/조회할 데이터셋 이름")
    parser.add_argument("--language", "-l", default="Korean", help="데이터셋 언어 (기본값: Korean)")
    parser.add_argument("--remove-duplicates", "-r", action="store_true", help="중복 항목 제거 여부")
    parser.add_argument("--delete", "-d", action="store_true", help="지정된 데이터셋 삭제")
    parser.add_argument("--show", "-s", action="store_true", help="지정된 데이터셋의 통계 정보와 샘플 출력")
    parser.add_argument("--list", action="store_true", help="사용 가능한 데이터셋 목록 표시")
    args = parser.parse_args()
    
    # Langfuse 설정
    langfuse_client = setup_langfuse()
    
    # 데이터셋 목록 표시 모드
    if args.list:
        list_datasets(langfuse_client)
        return
    
    # 데이터셋 이름 필수 체크 (목록 표시 모드 제외)
    if not args.name:
        print("오류: 데이터셋 이름(-n/--name)을 지정해야 합니다.")
        parser.print_help()
        return
    
    # 데이터셋 삭제 모드
    if args.delete:
        delete_benchmark_dataset(langfuse_client, args.name)
        return
    
    # 데이터셋 정보 조회 모드
    if args.show:
        show_dataset_stats(langfuse_client, args.name)
        return
    
    # 업로드 모드에서는 파일 필수
    if not args.file:
        print("오류: 업로드 모드에서는 파일(-f/--file)을 지정해야 합니다.")
        parser.print_help()
        return
    
    # 데이터셋 로드
    df = load_dataset(args.file)
    
    print(f"데이터셋 크기: {len(df)} 행")
    print(f"데이터셋 컬럼: {df.columns.tolist()}")
    print(f"데이터셋 샘플: {df.head()}")

    # 데이터셋 정제
    #refined_df = refine_dataset_in_dataframe(df)

    # 데이터 변환
    data = process_dataframe_to_langfuse_format(df, args.language)
    #data = process_dataframe_to_langfuse_format(refined_df, args.language)
    
    # 데이터셋 생성 및 업로드
    dataset = create_benchmark_dataset(langfuse_client, args.name, data)
    
    # 중복 제거 (옵션)
    if args.remove_duplicates:
        dataset = remove_dataset_duplicates(langfuse_client, args.name)
    
    print(f"데이터셋 '{args.name}' 처리 완료")
    
    # 데이터셋 확인
    try:
        dataset = langfuse_client.get_dataset(name=args.name)
        print(f"최종 데이터셋 크기: {len(dataset.items)} 항목")
        
        # 샘플 항목 출력
        if len(dataset.items) > 0:
            print("\n샘플 항목 정보:")
            item = dataset.items[0]
            print(f"ID: {item.id}")
            print(f"입력: {item.input}")
            print(f"기대 출력: {item.expected_output}")
            print(f"메타데이터: {item.metadata}")
    except Exception as e:
        print(f"Warning: 데이터셋 확인 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Langfuse와 LangChain을 활용한 멀티턴 Instruction Following 평가 스크립트

이 스크립트는 다음 기능을 수행합니다:
1. Langfuse 서버에서 데이터셋 다운로드
2. LangChain을 사용하여 멀티턴 대화 생성
3. IFEval을 사용하여 strict/loose 평가 수행
4. 평가 결과를 Langfuse 서버에 기록
"""

import os
import sys
import json
import time
import uuid
import argparse
from glob import glob
from pprint import pprint
import pandas as pd
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 경고 무시
warnings.filterwarnings("ignore")

# 환경변수 설정 및 로드
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
WORK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ENV_FILE = os.path.join(WORK_DIR, ".env.params")
print(f"ENV_FILE: {ENV_FILE}")
from dotenv import load_dotenv
load_dotenv(ENV_FILE)

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(WORK_DIR)

# Langfuse 관련 라이브러리
try:
    from langfuse import get_client, Langfuse
    from langfuse.langchain import CallbackHandler
except ImportError:
    print("Error: Langfuse 라이브러리를 설치해주세요. (pip install langfuse)")
    sys.exit(1)

# LangChain 관련 라이브러리
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# IFEval 관련 라이브러리
try:
    # 상대 경로로 ifeval 모듈 임포트
    sys.path.append(os.path.join(WORK_DIR, "ifeval"))
    from ifeval.evaluation_main import test_instruction_following_strict, test_instruction_following_loose
    from ifeval.evaluation_main import InputExample, OutputExample
except ImportError:
    print("Warning: ifeval 모듈을 가져올 수 없습니다. 평가 기능이 제한될 수 있습니다.")
    sys.exit(1)

# 평가 결과를 담는 데이터 클래스들
@dataclass
class EvaluationScore:
    """평가 점수를 담는 데이터 클래스"""
    overall_score: float
    strict_prompt_score: float
    strict_inst_score: float
    loose_prompt_score: float
    loose_inst_score: float

@dataclass
class EvaluationDetail:
    """평가 세부 정보를 담는 데이터 클래스"""
    follow_all_instructions: bool
    follow_instruction_list: list[bool]
    is_evaluated_list: list[bool]

@dataclass
class EvaluationResult:
    """평가 결과를 담는 데이터 클래스"""
    item_id: str
    key: str
    turn_index: int
    input: Any
    output: str
    score: EvaluationScore
    result_strict: EvaluationDetail
    result_loose: EvaluationDetail
    trace_id: Optional[str] = None
    error: Optional[str] = None

class LangfuseEvaluator:
    """Langfuse를 사용한 평가 클래스"""
    
    def __init__(self, model_name: str = None, temperature: float = 0.0, verbose: bool = False):
        """
        평가기 초기화
        
        Args:
            model_name: 사용할 LLM 모델 이름
            temperature: 생성 온도
            verbose: 상세 로그 출력 여부
        """
        self.verbose = verbose
        
        # Langfuse 클라이언트 초기화
        self.langfuse_client = get_client()
        auth_status = self.langfuse_client.auth_check()
        print(f"Langfuse 인증 상태: {auth_status}")
        
        if not auth_status:
            print("Error: Langfuse 인증에 실패했습니다. API 키를 확인해주세요.")
            sys.exit(1)
        
        # LLM 모델 초기화
        if model_name is None:
            if os.getenv("OPENAI_BASE_URL") == "https://api.openai.com/v1":
                model_name = "gpt-4.1-mini"
            else:
                model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
        
        self.model_name = model_name
        self.temperature = temperature
        
        print(f"LLM 모델: {self.model_name} (temperature={self.temperature})")
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        
        # Langfuse 콜백 핸들러 생성
        self.langfuse_handler = CallbackHandler()
        
        # 체인 생성
        self.ifgen_chain = self._create_ifgen_chain()
    
    def _create_ifgen_chain(self):
        """IFGen 체인 생성"""
        template = """{prompt}"""
        prompt = ChatPromptTemplate.from_template(template)

        ifgen_chain = (
            {"prompt": RunnableLambda(self._check_prompt)} 
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return ifgen_chain
    
    def _check_prompt(self, prompt):
        """프롬프트 처리 함수"""
        # 문자열이 아닌 경우에만 JSON 파싱을 시도
        if isinstance(prompt, str):
            try:
                # JSON 형식인 경우 파싱 시도
                item = json.loads(prompt)
                if isinstance(item, str):
                    return item
                elif isinstance(item, dict):
                    return item.get("content")
                else:
                    raise ValueError("Invalid prompt type")
            except json.JSONDecodeError:
                # JSON 형식이 아닌 일반 문자열인 경우 그대로 반환
                return prompt
        else:
            # 문자열이 아닌 경우 (이미 파싱된 객체인 경우)
            if isinstance(prompt, dict):
                return prompt.get("content", prompt)
            return prompt
    
    def invoke_with_tracing(self, prompt: str, **kwargs):
        """추적이 포함된 체인 실행"""
        return self.ifgen_chain.invoke(
            prompt, 
            config={
                "callbacks": [self.langfuse_handler],
                "metadata": {
                    "langfuse_tags": ["evaluation", "ifeval"],
                    "evaluation_run": True
                },
            },
            **kwargs
        )
    
    def gen_response_with_max_retry(self, messages: list[BaseMessage], max_retry: int = 3) -> BaseMessage:
        """최대 재시도 횟수로 응답 생성"""
        if len(messages) == 0:
            raise ValueError("messages is empty")
        
        for attempt in range(max_retry, 0, -1):
            try:
                # 메시지 체인 실행
                prompt = messages[-1].content if messages else ""
                ai_content = self.invoke_with_tracing(prompt)
                return AIMessage(content=ai_content)
            except Exception as e:
                print(f"   ❌ 실패: {str(e)}")
                time.sleep(1)
        return AIMessage(content=f'[MAX_RETRY={max_retry}] Failed.')
    
    def get_dataset(self, dataset_name: str):
        """Langfuse에서 데이터셋 가져오기"""
        try:
            print(f"🔍 Langfuse에서 데이터셋 '{dataset_name}' 가져오는 중...")
            dataset = self.langfuse_client.get_dataset(name=dataset_name)
            print(f"✅ Langfuse 데이터셋 가져오기 성공: {len(dataset.items)}개 아이템")
            return dataset
        except Exception as e:
            print(f"❌ Langfuse 데이터셋 가져오기 실패: {e}")
            sys.exit(1)
    
    def get_prompt(self, item: Any, turn_index: int) -> str:
        """아이템에서 프롬프트 추출"""
        try:
            prompt = item.input.get(f"turn_{turn_index}", {}).get("prompt", "")
            if prompt:
                return self._check_prompt(prompt)
            return ""
        except Exception as e:
            print(f"Warning: 프롬프트 추출 실패 - {e}")
            return ""
    
    def get_instruction_id_list(self, item: Any, turn_index: int) -> list[str]:
        """아이템에서 instruction_id_list 추출"""
        try:
            instruction_id_list = item.input.get(f"turn_{turn_index}", {}).get("instruction_id_list", "[]")
            
            # 문자열인 경우 JSON으로 파싱
            if isinstance(instruction_id_list, str):
                instruction_id_list = json.loads(instruction_id_list)
            
            # 언어 접두사 추가
            prefix = "en:"
            if item.metadata.get("language") == "Korean" or str(item.metadata.get("key", "")).endswith("ko"):
                prefix = "ko:"
            
            return [prefix + str(instruction_id) for instruction_id in instruction_id_list]
        except Exception as e:
            print(f"Warning: instruction_id_list 추출 실패 - {e}")
            return []
    
    def get_kwargs(self, item: Any, turn_index: int) -> list[dict]:
        """아이템에서 kwargs 추출"""
        try:
            kwargs = item.input.get(f"turn_{turn_index}", {}).get("kwargs", "[]")
            
            # 문자열인 경우 JSON으로 파싱
            if isinstance(kwargs, str):
                kwargs = json.loads(kwargs)
            
            # 리스트가 아닌 경우 리스트로 변환
            if not isinstance(kwargs, list):
                kwargs = [kwargs]
            
            # 각 항목 처리
            kwargs_list = []
            for kwarg in kwargs:
                if isinstance(kwarg, dict):
                    kwargs_list.append(kwarg)
                elif isinstance(kwarg, str):
                    try:
                        parsed_kwarg = json.loads(kwarg)
                        kwargs_list.append(parsed_kwarg)
                    except json.JSONDecodeError:
                        print(f"Warning: kwargs 파싱 실패: {kwarg}")
                        kwargs_list.append({})
                else:
                    kwargs_list.append({})
            
            return kwargs_list
        except Exception as e:
            print(f"Warning: kwargs 추출 실패 - {e}")
            return [{}]
    
    def create_input_example(self, item: Any, turn_index: int) -> InputExample:
        """평가용 입력 예제 생성"""
        return InputExample(
            key=item.metadata.get("key", ""),
            prompt=self.get_prompt(item, turn_index),
            instruction_id_list=self.get_instruction_id_list(item, turn_index),
            kwargs=self.get_kwargs(item, turn_index)
        )
    
    def create_input_to_response_dict(self, item: Any, turn_index: int, response: str) -> Dict[str, str]:
        """입력-응답 딕셔너리 생성"""
        prompt = self.get_prompt(item, turn_index)
        return {prompt: response}
    
    def calc_score_example(self, output_strict: OutputExample, output_loose: OutputExample) -> EvaluationScore:
        """평가 점수 계산"""
        strict_prompt_score = 1.0 if output_strict.follow_all_instructions else 0.0
        loose_prompt_score = 1.0 if output_loose.follow_all_instructions else 0.0
        
        # 지시 목록이 비어있는 경우 처리
        if len(output_strict.follow_instruction_list) > 0:
            strict_inst_score = sum(output_strict.follow_instruction_list) / len(output_strict.follow_instruction_list)
        else:
            strict_inst_score = 0.0
            
        if len(output_loose.follow_instruction_list) > 0:
            loose_inst_score = sum(output_loose.follow_instruction_list) / len(output_loose.follow_instruction_list)
        else:
            loose_inst_score = 0.0
        
        overall_score = (strict_prompt_score + loose_prompt_score + strict_inst_score + loose_inst_score) / 4
        return EvaluationScore(
            overall_score=overall_score,
            strict_prompt_score=strict_prompt_score,
            strict_inst_score=strict_inst_score,
            loose_prompt_score=loose_prompt_score,
            loose_inst_score=loose_inst_score
        )
    
    def merge_result_example(self, item, turn_index: int, output_strict: OutputExample, output_loose: OutputExample, trace_id: str) -> EvaluationResult:
        """평가 결과 병합"""
        input_example = self.create_input_example(item, turn_index)
        output = output_strict.response
        
        result_strict = EvaluationDetail(
            follow_all_instructions=output_strict.follow_all_instructions,
            follow_instruction_list=output_strict.follow_instruction_list,
            is_evaluated_list=output_strict.is_evaluated_list
        )
        
        result_loose = EvaluationDetail(
            follow_all_instructions=output_loose.follow_all_instructions,
            follow_instruction_list=output_loose.follow_instruction_list,
            is_evaluated_list=output_loose.is_evaluated_list
        )
        
        score = self.calc_score_example(output_strict, output_loose)
        
        return EvaluationResult(
            item_id=item.id,
            key=item.metadata.get("key", ""),
            turn_index=turn_index,
            input=input_example,
            output=output,
            trace_id=trace_id,
            score=score,
            result_strict=result_strict,
            result_loose=result_loose,
            error=None
        )
    
    def eval_process_by_turn(self, item, turn_index: int, run_name: str, messages: list[BaseMessage]) -> Tuple[EvaluationResult, list[BaseMessage]]:
        """턴별 평가 처리"""
        with item.run(run_name=run_name+f"_turn_{turn_index}") as root_span:
            
            # 프롬프트 가져오기
            prompt = self.get_prompt(item, turn_index)
            if not prompt:
                print(f"   ⚠️ 턴 {turn_index}의 프롬프트가 비어 있습니다.")
                result = EvaluationResult(
                    item_id=item.id,
                    key=item.metadata.get("key", ""),
                    turn_index=turn_index,
                    input=None,
                    output="",
                    score=EvaluationScore(0, 0, 0, 0, 0),
                    result_strict=None,
                    result_loose=None,
                    trace_id=getattr(root_span, 'trace_id', None),
                    error=f"Empty prompt for turn {turn_index}"
                )
                return result, messages
            
            # 메시지 추가 및 응답 생성
            messages.append(HumanMessage(content=prompt))
            print(f"   🔄 턴 {turn_index} 처리 중...")
            
            if self.verbose:
                print("   입력 메시지:")
                pprint(messages)
            
            ai_message = self.gen_response_with_max_retry(messages)
            messages.append(ai_message)
            response = ai_message.content
        
            if self.verbose:
                print("   출력 메시지:")
                pprint(messages)
                    
            if "Failed" in response:
                result = EvaluationResult(
                    item_id=item.id,
                    key=item.metadata.get("key", ""),
                    turn_index=turn_index,
                    input=self.create_input_example(item, turn_index),
                    output=response,
                    score=EvaluationScore(0, 0, 0, 0, 0),
                    result_strict=None,
                    result_loose=None,
                    trace_id=getattr(root_span, 'trace_id', None),
                    error=f"Failed to generate response: {response}"
                )
                return result, messages
            
            # 평가 수행
            input_example = self.create_input_example(item, turn_index)
            prompt_to_response = self.create_input_to_response_dict(item, turn_index, response)
            
            # 1. strict instruction following evaluation
            output_strict = test_instruction_following_strict(input_example, prompt_to_response)
            
            # 2. loose instruction following evaluation
            output_loose = test_instruction_following_loose(input_example, prompt_to_response)
            
            # 전체 점수 계산
            score = self.calc_score_example(output_strict, output_loose)
            print(f"   ✅ 턴 {turn_index} 평가 완료 (overall score: {score.overall_score:.2f})")
            
            # 3. 평가 결과 기록
            # 각 instruction 의 True/False 결과를 comment에 저장한다. 또한, evaluated True/False 결과를 저장한다.
            # follow_all_instructions 는 절대로 comment에 저장하지 않는다.
            root_span.score(name="overall", value=score.overall_score, comment=str(item.metadata.get("key", "")))
            root_span.score(name="strict_prompt_score", value=score.strict_prompt_score, comment="result: "+str(output_strict.follow_instruction_list)+", tested: "+str(output_strict.is_evaluated_list))
            root_span.score(name="loose_prompt_score", value=score.loose_prompt_score, comment="result: "+str(output_loose.follow_instruction_list)+", tested: "+str(output_loose.is_evaluated_list))
            root_span.score(name="strict_inst_score", value=score.strict_inst_score, comment="result: "+str(output_strict.follow_instruction_list)+", tested: "+str(output_strict.is_evaluated_list))
            root_span.score(name="loose_inst_score", value=score.loose_inst_score, comment="result: "+str(output_loose.follow_instruction_list)+", tested: "+str(output_loose.is_evaluated_list))

            # 메타데이터 업데이트
            # Langfuse API 변경: update_metadata -> 객체 업데이트 시 metadata 필드 사용
            root_span.update(metadata={
                "key": item.metadata.get("key", ""),
                "instruction_id_list": input_example.instruction_id_list,
                "kwargs": input_example.kwargs,
                "strict_IF_result": output_strict.follow_instruction_list,
                "strict_IF_tested": output_strict.is_evaluated_list,
                "loose_IF_result": output_loose.follow_instruction_list,
                "loose_IF_tested": output_loose.is_evaluated_list
            })
            
            # 결과 저장
            result = self.merge_result_example(item, turn_index, output_strict, output_loose, getattr(root_span, 'trace_id', None))
            return result, messages
    
    def run_dataset_evaluation(self, dataset_name: str, run_name: str, limit: int = None, parallel: bool = False, max_workers: int = 4) -> List[Dict]:
        """데이터셋 전체에 대한 평가 실행"""
        # 데이터셋 가져오기
        dataset = self.get_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"데이터셋 '{dataset_name}'이(가) 존재하지 않습니다.")
        
        items = dataset.items
        if limit and limit > 0:
            items = items[:limit]
        
        print(f"📊 multi-if 평가 시작: {dataset_name} ({len(items)}개 항목)")
        
        if parallel:
            print(f"🔄 병렬 처리 모드 (작업자: {max_workers}개)")
            return self._run_parallel_evaluation(items, run_name, max_workers)
        else:
            print(f"🔄 순차 처리 모드")
            return self._run_sequential_evaluation(items, run_name)
    
    def _run_sequential_evaluation(self, items: List, run_name: str) -> List[Dict]:
        """순차 평가 실행"""
        results = []
        successful = 0
        failed = 0
        
        # 아이템 순회
        for idx, item in enumerate(items, 1):
            print(f"\n🔄 아이템 {idx}/{len(items)} 처리 중...")
            
            # 메타데이터 출력
            print(f"   📋 키: {item.metadata.get('key', 'N/A')}")
            print(f"   📋 언어: {item.metadata.get('language', 'N/A')}")
            
            try:
                item_results = self._evaluate_single_item(item, idx, run_name)
                
                # 성공/실패 확인
                if all(result["result"].error is None for result in item_results):
                    successful += 1
                else:
                    failed += 1
                    print(f"   ❌ {item.metadata.get('key', 'N/A')} 오류 발생")
                
                results.extend(item_results)
                
            except Exception as e:
                failed += 1
                print(f"   ❌ {item.metadata.get('key', 'N/A')} 예외 발생: {e}")
                # 오류 항목도 결과에 포함
                results.append({
                    "item_index": idx,
                    "turn_index": 0,
                    "result": EvaluationResult(
                        item_id=item.id,
                        key=item.metadata.get("key", ""),
                        turn_index=0,
                        input=None,
                        output="",
                        score=EvaluationScore(0, 0, 0, 0, 0),
                        result_strict=None,
                        result_loose=None,
                        error=str(e)
                    )
                })

        # 결과 요약
        print(f"\n📋 평가 완료: 성공 {successful}개 / 실패 {failed}개 / 총 {len(items)}개")
        
        return results
    
    def _run_parallel_evaluation(self, items: List, run_name: str, max_workers: int) -> List[Dict]:
        """병렬 평가 실행"""
        def evaluate_item_wrapper(item_data):
            item, idx = item_data
            try:
                return self._evaluate_single_item(item, idx, run_name)
            except Exception as e:
                print(f"   ❌ 아이템 {idx} 예외 발생: {e}")
                return [{
                    "item_index": idx,
                    "turn_index": 0,
                    "result": EvaluationResult(
                        item_id=item.id,
                        key=item.metadata.get("key", ""),
                        turn_index=0,
                        input=None,
                        output="",
                        score=EvaluationScore(0, 0, 0, 0, 0),
                        result_strict=None,
                        result_loose=None,
                        error=str(e)
                    )
                }]
        
        # 병렬 처리를 위한 데이터 준비
        item_data_list = [(item, idx) for idx, item in enumerate(items, 1)]
        
        # ThreadPoolExecutor를 사용한 병렬 처리
        results = []
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # tqdm으로 진행률 표시
            future_to_item = {executor.submit(evaluate_item_wrapper, item_data): item_data for item_data in item_data_list}
            
            for future in tqdm(future_to_item, desc="평가 진행", unit="item"):
                try:
                    item_results = future.result()
                    
                    # 성공/실패 확인
                    if all(result["result"].error is None for result in item_results):
                        successful += 1
                    else:
                        failed += 1
                    
                    results.extend(item_results)
                    
                except Exception as e:
                    failed += 1
                    item_data = future_to_item[future]
                    item, idx = item_data
                    print(f"   ❌ 아이템 {idx} 처리 실패: {e}")
                    
                    # 오류 항목도 결과에 포함
                    results.append({
                        "item_index": idx,
                        "turn_index": 0,
                        "result": EvaluationResult(
                            item_id=item.id,
                            key=item.metadata.get("key", ""),
                            turn_index=0,
                            input=None,
                            output="",
                            score=EvaluationScore(0, 0, 0, 0, 0),
                            result_strict=None,
                            result_loose=None,
                            error=str(e)
                        )
                    })

        # 결과 요약
        print(f"\n📋 병렬 평가 완료: 성공 {successful}개 / 실패 {failed}개 / 총 {len(items)}개")
        
        return results
    
    def _evaluate_single_item(self, item, idx: int, run_name: str) -> List[Dict]:
        """단일 아이템 평가"""
        if not self.verbose:
            print(f"🔄 아이템 {idx} 처리 중... (키: {item.metadata.get('key', 'N/A')})")
        else:
            print(f"\n🔄 아이템 {idx} 처리 중...")
            print(f"   📋 키: {item.metadata.get('key', 'N/A')}")
            print(f"   📋 언어: {item.metadata.get('language', 'N/A')}")
        
        messages = []
        item_results = []
        
        # 턴 수 확인
        max_turn = 0
        for key in item.input:
            if key.startswith("turn_") and key[5:].isdigit():
                turn_num = int(key[5:])
                max_turn = max(max_turn, turn_num)
        
        if max_turn == 0:
            if self.verbose:
                print(f"   ⚠️ 아이템에 턴 정보가 없습니다.")
            return []
        
        # 각 턴별로 평가 수행
        with item.run(run_name=run_name) as root_span_all:

            success_turns = []
            for turn_index in range(1, max_turn + 1):
                result, messages = self.eval_process_by_turn(item, turn_index, run_name, messages)

                if result.error is None:
                    success_turns.append(True)
                else:
                    success_turns.append(False)
                    if self.verbose:
                        print(f"   ❌ {item.metadata.get('key', 'N/A')} 오류: {result.error}")

                item_results.append({
                    "item_index": idx,
                    "turn_index": turn_index,
                    "result": result
                })

            # 평가 결과 기록
            score = EvaluationScore(0, 0, 0, 0, 0)
            for result in item_results:
                score.overall_score += result["result"].score.overall_score
                score.strict_prompt_score += result["result"].score.strict_prompt_score
                score.loose_prompt_score += result["result"].score.loose_prompt_score
                score.strict_inst_score += result["result"].score.strict_inst_score
                score.loose_inst_score += result["result"].score.loose_inst_score

            if item_results:
                score.overall_score /= len(item_results)
                score.strict_prompt_score /= len(item_results)
                score.loose_prompt_score /= len(item_results)
                score.strict_inst_score /= len(item_results)
                score.loose_inst_score /= len(item_results)

            root_span_all.score(name="overall", value=score.overall_score, comment=f"평균 점수: {len(item_results)}개 턴")
            root_span_all.score(name="strict_prompt_score", value=score.strict_prompt_score, comment=f"평균 점수: {len(item_results)}개 턴")
            root_span_all.score(name="loose_prompt_score", value=score.loose_prompt_score, comment=f"평균 점수: {len(item_results)}개 턴")
            root_span_all.score(name="strict_inst_score", value=score.strict_inst_score, comment=f"평균 점수: {len(item_results)}개 턴")
            root_span_all.score(name="loose_inst_score", value=score.loose_inst_score, comment=f"평균 점수: {len(item_results)}개 턴")

            # 전체 아이템에 대한 메타데이터 추가
            all_metadata = {
                "key": item.metadata.get("key", ""),
                "language": item.metadata.get("language", ""),
                "total_turns": len(item_results),
                "turn_results": [
                    {
                        "turn_index": result["turn_index"],
                        "overall_score": result["result"].score.overall_score,
                        "strict_prompt_score": result["result"].score.strict_prompt_score,
                        "loose_prompt_score": result["result"].score.loose_prompt_score,
                        "strict_inst_score": result["result"].score.strict_inst_score,
                        "loose_inst_score": result["result"].score.loose_inst_score,
                        
                        "strict_IF_result": result["result"].result_strict.follow_instruction_list,
                        "strict_IF_tested": result["result"].result_strict.is_evaluated_list,
                        "loose_IF_result": result["result"].result_loose.follow_instruction_list,
                        "loose_IF_tested": result["result"].result_loose.is_evaluated_list
                    } for result in item_results
                ]
            }
            root_span_all.update(metadata=all_metadata)
            
            return item_results
    
    def save_results_to_file(self, results: List[Dict], output_path: str):
        """평가 결과를 파일로 저장"""
        try:
            # 결과를 직렬화 가능한 형태로 변환
            serializable_results = []
            for result_item in results:
                item_dict = {
                    "item_index": result_item["item_index"],
                    "turn_index": result_item["turn_index"],
                }
                
                result = result_item["result"]
                result_dict = {
                    "item_id": result.item_id,
                    "key": result.key,
                    "turn_index": result.turn_index,
                    "input": result.input.prompt,
                    "output": result.output,
                    "trace_id": result.trace_id,
                    "error": result.error
                }
                
                # 점수 정보
                if result.score:
                    result_dict["score"] = {
                        "overall_score": result.score.overall_score,
                        "strict_prompt_score": result.score.strict_prompt_score,
                        "strict_inst_score": result.score.strict_inst_score,
                        "loose_prompt_score": result.score.loose_prompt_score,
                        "loose_inst_score": result.score.loose_inst_score
                    }
                
                # strict 평가 결과
                if result.result_strict:
                    result_dict["result_strict"] = {
                        "follow_all_instructions": result.result_strict.follow_all_instructions,
                        "follow_instruction_list": result.result_strict.follow_instruction_list,
                        "is_evaluated_list": result.result_strict.is_evaluated_list
                    }
                
                # loose 평가 결과
                if result.result_loose:
                    result_dict["result_loose"] = {
                        "follow_all_instructions": result.result_loose.follow_all_instructions,
                        "follow_instruction_list": result.result_loose.follow_instruction_list,
                        "is_evaluated_list": result.result_loose.is_evaluated_list
                    }
                
                item_dict["result"] = result_dict
                serializable_results.append(item_dict)
            
            # 결과 저장
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 결과 저장 완료: {output_path}")
            
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Langfuse와 LangChain을 활용한 멀티턴 Instruction Following 평가")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="평가할 Langfuse 데이터셋 이름")
    parser.add_argument("--run-name", "-r", type=str, default=f"ifeval_{int(time.time())}", help="평가 실행 이름")
    parser.add_argument("--model", "-m", type=str, help="사용할 LLM 모델 이름")
    parser.add_argument("--temperature", "-t", type=float, default=0.6, help="생성 온도")
    parser.add_argument("--limit", "-l", type=int, help="평가할 최대 아이템 수")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    parser.add_argument("--output", "-o", type=str, help="결과를 저장할 파일 경로")
    parser.add_argument("--parallel", "-p", action="store_true", help="병렬 처리 사용")
    parser.add_argument("--workers", "-w", type=int, default=4, help="병렬 처리 작업자 수")
    
    args = parser.parse_args()
    
    # 평가기 초기화
    evaluator = LangfuseEvaluator(
        model_name=args.model,
        temperature=args.temperature,
        verbose=args.verbose
    )
    
    # 평가 실행
    results = evaluator.run_dataset_evaluation(
        dataset_name=args.dataset,
        run_name=args.run_name,
        limit=args.limit,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    # 결과 저장
    if args.output:
        evaluator.save_results_to_file(results, args.output)
    
    print("✅ 평가 완료")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Langfuse와 LangGraph를 활용한 멀티턴 Instruction Following 평가 스크립트

이 스크립트는 다음 기능을 수행합니다:
1. Langfuse 서버에서 데이터셋 다운로드
2. LangGraph를 사용하여 멀티턴 대화 생성 및 평가 워크플로우 구축
3. IFEval을 사용하여 strict/loose 평가 수행
4. 평가 결과를 Langfuse 서버에 기록

LangGraph 최적화 특징:
- 상태 관리를 통한 멀티턴 대화 처리
- 조건부 노드를 통한 효율적인 워크플로우 제어
- 병렬 처리 지원
- 에러 처리 및 재시도 메커니즘
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
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional, TypedDict, Annotated, Literal
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import asyncio
from functools import partial

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

# LangGraph 관련 라이브러리
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph.message import add_messages
except ImportError:
    print("Error: LangGraph 라이브러리를 설치해주세요. (pip install langgraph)")
    sys.exit(1)

# LangChain 관련 라이브러리
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# IFEval 관련 라이브러리
try:
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

# LangGraph 상태 정의
class GraphState(TypedDict):
    """LangGraph 상태를 정의하는 클래스"""
    # 데이터셋 및 아이템 정보
    dataset_item: Any
    item_index: int
    
    # 대화 상태
    messages: Annotated[List[BaseMessage], add_messages]
    current_turn: int
    max_turns: int
    
    # 평가 관련
    evaluation_results: List[EvaluationResult]
    current_prompt: str
    current_response: str
    
    # 메타데이터
    run_name: str
    trace_id: Optional[str]
    
    # 에러 처리
    error: Optional[str]
    retry_count: int
    max_retries: int

class LangGraphEvaluator:
    """LangGraph를 사용한 평가 클래스"""
    
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
                model_name = "gpt-4o-mini"
            else:
                model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        self.model_name = model_name
        self.temperature = temperature
        
        print(f"LLM 모델: {self.model_name} (temperature={self.temperature})")
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        
        # Langfuse 콜백 핸들러 생성
        self.langfuse_handler = CallbackHandler()
        
        # LangGraph 워크플로우 생성
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(GraphState)
        
        # 노드 추가
        workflow.add_node("initialize_evaluation", self._initialize_evaluation)
        workflow.add_node("prepare_turn", self._prepare_turn)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("evaluate_response", self._evaluate_response)
        workflow.add_node("record_results", self._record_results)
        workflow.add_node("check_completion", self._check_completion)
        workflow.add_node("finalize_evaluation", self._finalize_evaluation)
        
        # 엣지 추가
        workflow.add_edge(START, "initialize_evaluation")
        workflow.add_edge("initialize_evaluation", "prepare_turn")
        workflow.add_edge("prepare_turn", "generate_response")
        workflow.add_edge("generate_response", "evaluate_response")
        workflow.add_edge("evaluate_response", "record_results")
        workflow.add_edge("record_results", "check_completion")
        
        # 조건부 엣지 추가
        workflow.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "prepare_turn",
                "finalize": "finalize_evaluation"
            }
        )
        
        workflow.add_edge("finalize_evaluation", END)
        
        # 메모리 체크포인트 설정
        memory = MemorySaver()
        
        return workflow.compile(checkpointer=memory)
    
    def _initialize_evaluation(self, state: GraphState) -> GraphState:
        """평가 초기화"""
        item = state["dataset_item"]
        
        # 턴 수 확인
        max_turns = 0
        for key in item.input:
            if key.startswith("turn_") and key[5:].isdigit():
                turn_num = int(key[5:])
                max_turns = max(max_turns, turn_num)
        
        state.update({
            "current_turn": 1,
            "max_turns": max_turns,
            "messages": [],
            "evaluation_results": [],
            "retry_count": 0,
            "max_retries": 3,
            "error": None
        })
        
        if self.verbose:
            print(f"   🔄 평가 초기화: 최대 {max_turns}턴")
        
        return state
    
    def _prepare_turn(self, state: GraphState) -> GraphState:
        """턴 준비"""
        item = state["dataset_item"]
        turn_index = state["current_turn"]
        
        # 프롬프트 가져오기
        prompt = self.get_prompt(item, turn_index)
        
        if not prompt:
            state["error"] = f"Empty prompt for turn {turn_index}"
            return state
        
        state["current_prompt"] = prompt
        
        if self.verbose:
            print(f"   🔄 턴 {turn_index} 준비 완료")
        
        return state
    
    def _generate_response(self, state: GraphState) -> GraphState:
        """응답 생성"""
        try:
            prompt = state["current_prompt"]
            messages = state["messages"].copy()
            
            # 새 메시지 추가
            messages.append(HumanMessage(content=prompt))
            
            # LLM 호출
            response = self.llm.invoke(
                messages,
                config={
                    "callbacks": [self.langfuse_handler],
                    "metadata": {
                        "langfuse_tags": ["evaluation", "ifeval", "langgraph"],
                        "evaluation_run": True,
                        "turn_index": state["current_turn"]
                    }
                }
            )
            
            ai_message = AIMessage(content=response.content)
            messages.append(ai_message)
            
            state.update({
                "messages": messages,
                "current_response": response.content,
                "retry_count": 0
            })
            
            if self.verbose:
                print(f"   ✅ 응답 생성 완료 (턴 {state['current_turn']})")
            
        except Exception as e:
            state["retry_count"] += 1
            if state["retry_count"] < state["max_retries"]:
                print(f"   ⚠️ 응답 생성 실패, 재시도 ({state['retry_count']}/{state['max_retries']}): {e}")
                time.sleep(1)
            else:
                state["error"] = f"Failed to generate response after {state['max_retries']} attempts: {e}"
                state["current_response"] = f"[MAX_RETRY={state['max_retries']}] Failed."
        
        return state
    
    def _evaluate_response(self, state: GraphState) -> GraphState:
        """응답 평가"""
        if state.get("error"):
            return state
        
        item = state["dataset_item"]
        turn_index = state["current_turn"]
        response = state["current_response"]
        
        try:
            # 입력 예제 생성
            input_example = self.create_input_example(item, turn_index)
            prompt_to_response = self.create_input_to_response_dict(item, turn_index, response)
            
            # 평가 수행
            output_strict = test_instruction_following_strict(input_example, prompt_to_response)
            output_loose = test_instruction_following_loose(input_example, prompt_to_response)
            
            # 점수 계산
            score = self.calc_score_example(output_strict, output_loose)
            
            # 결과 생성
            result = self.merge_result_example(item, turn_index, output_strict, output_loose, state.get("trace_id"))
            
            # 결과 추가
            state["evaluation_results"].append(result)
            
            if self.verbose:
                print(f"   ✅ 평가 완료 (턴 {turn_index}, 점수: {score.overall_score:.2f})")
            
        except Exception as e:
            state["error"] = f"Evaluation failed for turn {turn_index}: {e}"
            print(f"   ❌ 평가 실패: {e}")
        
        return state
    
    def _record_results(self, state: GraphState) -> GraphState:
        """결과 기록"""
        if state.get("error") or not state["evaluation_results"]:
            return state
        
        item = state["dataset_item"]
        turn_index = state["current_turn"]
        
        try:
            # 현재 턴의 결과 가져오기
            current_result = state["evaluation_results"][-1]
            
            # Langfuse에 결과 기록 - 턴별 트레이스 생성
            with item.run(run_name=f"{state['run_name']}_turn_{turn_index}") as span:
                score = current_result.score
                
                # 점수 기록 - LangChain 버전과 동일한 형식
                span.score(name="overall", value=score.overall_score, comment=str(item.metadata.get("key", "")))
                span.score(name="strict_prompt_score", value=score.strict_prompt_score, 
                          comment="result: "+str(current_result.result_strict.follow_instruction_list)+", tested: "+str(current_result.result_strict.is_evaluated_list))
                span.score(name="loose_prompt_score", value=score.loose_prompt_score,
                          comment="result: "+str(current_result.result_loose.follow_instruction_list)+", tested: "+str(current_result.result_loose.is_evaluated_list))
                span.score(name="strict_inst_score", value=score.strict_inst_score,
                          comment="result: "+str(current_result.result_strict.follow_instruction_list)+", tested: "+str(current_result.result_strict.is_evaluated_list))
                span.score(name="loose_inst_score", value=score.loose_inst_score,
                          comment="result: "+str(current_result.result_loose.follow_instruction_list)+", tested: "+str(current_result.result_loose.is_evaluated_list))
                
                # 메타데이터 업데이트 - LangChain 버전과 동일한 형식
                span.update(metadata={
                    "key": item.metadata.get("key", ""),
                    "instruction_id_list": current_result.input.instruction_id_list,
                    "kwargs": current_result.input.kwargs,
                    "strict_IF_result": current_result.result_strict.follow_instruction_list,
                    "strict_IF_tested": current_result.result_strict.is_evaluated_list,
                    "loose_IF_result": current_result.result_loose.follow_instruction_list,
                    "loose_IF_tested": current_result.result_loose.is_evaluated_list
                })
                
                # trace_id 업데이트
                if hasattr(span, 'trace_id'):
                    current_result.trace_id = span.trace_id
                    state["trace_id"] = span.trace_id
            
            if self.verbose:
                print(f"   📊 결과 기록 완료 (턴 {turn_index})")
                
        except Exception as e:
            print(f"   ⚠️ 결과 기록 실패: {e}")
        
        return state
    
    def _check_completion(self, state: GraphState) -> GraphState:
        """완료 확인"""
        state["current_turn"] += 1
        return state
    
    def _should_continue(self, state: GraphState) -> Literal["continue", "finalize"]:
        """계속 진행할지 결정"""
        if state.get("error"):
            return "finalize"
        
        if state["current_turn"] <= state["max_turns"]:
            return "continue"
        else:
            return "finalize"
    
    def _finalize_evaluation(self, state: GraphState) -> GraphState:
        """평가 완료"""
        item = state["dataset_item"]
        results = state["evaluation_results"]
        
        try:
            # 전체 결과 요약 - 전체 아이템에 대한 메인 트레이스 생성
            with item.run(run_name=state["run_name"]) as root_span:
                if results:
                    # 평균 점수 계산
                    avg_score = EvaluationScore(0, 0, 0, 0, 0)
                    for result in results:
                        avg_score.overall_score += result.score.overall_score
                        avg_score.strict_prompt_score += result.score.strict_prompt_score
                        avg_score.loose_prompt_score += result.score.loose_prompt_score
                        avg_score.strict_inst_score += result.score.strict_inst_score
                        avg_score.loose_inst_score += result.score.loose_inst_score
                    
                    count = len(results)
                    avg_score.overall_score /= count
                    avg_score.strict_prompt_score /= count
                    avg_score.loose_prompt_score /= count
                    avg_score.strict_inst_score /= count
                    avg_score.loose_inst_score /= count
                    
                    # 전체 점수 기록 - LangChain 버전과 동일한 형식
                    root_span.score(name="overall", value=avg_score.overall_score, comment=f"평균 점수: {count}개 턴")
                    root_span.score(name="strict_prompt_score", value=avg_score.strict_prompt_score, comment=f"평균 점수: {count}개 턴")
                    root_span.score(name="loose_prompt_score", value=avg_score.loose_prompt_score, comment=f"평균 점수: {count}개 턴")
                    root_span.score(name="strict_inst_score", value=avg_score.strict_inst_score, comment=f"평균 점수: {count}개 턴")
                    root_span.score(name="loose_inst_score", value=avg_score.loose_inst_score, comment=f"평균 점수: {count}개 턴")
                    
                    # 전체 메타데이터 업데이트 - LangChain 버전과 동일한 형식
                    all_metadata = {
                        "key": item.metadata.get("key", ""),
                        "language": item.metadata.get("language", ""),
                        "total_turns": count,
                        "turn_results": [
                            {
                                "turn_index": result.turn_index,
                                "overall_score": result.score.overall_score,
                                "strict_prompt_score": result.score.strict_prompt_score,
                                "loose_prompt_score": result.score.loose_prompt_score,
                                "strict_inst_score": result.score.strict_inst_score,
                                "loose_inst_score": result.score.loose_inst_score,
                                "strict_IF_result": result.result_strict.follow_instruction_list,
                                "strict_IF_tested": result.result_strict.is_evaluated_list,
                                "loose_IF_result": result.result_loose.follow_instruction_list,
                                "loose_IF_tested": result.result_loose.is_evaluated_list
                            } for result in results
                        ]
                    }
                    root_span.update(metadata=all_metadata)
                    
                    if self.verbose:
                        print(f"   📊 전체 평가 완료 (평균 점수: {avg_score.overall_score:.2f})")
                
        except Exception as e:
            print(f"   ⚠️ 최종 결과 기록 실패: {e}")
        
        return state
    
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
    
    def _check_prompt(self, prompt):
        """프롬프트 처리 함수"""
        if isinstance(prompt, str):
            try:
                item = json.loads(prompt)
                if isinstance(item, str):
                    return item
                elif isinstance(item, dict):
                    return item.get("content")
                else:
                    raise ValueError("Invalid prompt type")
            except json.JSONDecodeError:
                return prompt
        else:
            if isinstance(prompt, dict):
                return prompt.get("content", prompt)
            return prompt
    
    def get_instruction_id_list(self, item: Any, turn_index: int) -> list[str]:
        """아이템에서 instruction_id_list 추출"""
        try:
            instruction_id_list = item.input.get(f"turn_{turn_index}", {}).get("instruction_id_list", "[]")
            
            if isinstance(instruction_id_list, str):
                instruction_id_list = json.loads(instruction_id_list)
            
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
            
            if isinstance(kwargs, str):
                kwargs = json.loads(kwargs)
            
            if not isinstance(kwargs, list):
                kwargs = [kwargs]
            
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
    
    def run_dataset_evaluation(self, dataset_name: str, run_name: str, limit: int = None, parallel: bool = False, max_workers: int = 4) -> List[Dict]:
        """데이터셋 전체에 대한 평가 실행"""
        # 데이터셋 가져오기
        dataset = self.get_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"데이터셋 '{dataset_name}'이(가) 존재하지 않습니다.")
        
        items = dataset.items
        if limit and limit > 0:
            items = items[:limit]
        
        print(f"📊 LangGraph multi-if 평가 시작: {dataset_name} ({len(items)}개 항목)")
        
        results = []
        successful = 0
        failed = 0
        
        # 병렬 처리 또는 순차 처리
        if parallel:
            results = self._run_parallel_evaluation(items, run_name, max_workers)
        else:
            results = self._run_sequential_evaluation(items, run_name)
        
        # 성공/실패 통계 계산
        for result in results:
            if result.get("error"):
                failed += 1
            else:
                successful += 1
        
        print(f"\n📋 평가 완료: 성공 {successful}개 / 실패 {failed}개 / 총 {len(items)}개")
        
        return results
    
    def _run_sequential_evaluation(self, items: List, run_name: str) -> List[Dict]:
        """순차 평가 실행"""
        results = []
        
        for idx, item in enumerate(items, 1):
            print(f"\n🔄 아이템 {idx}/{len(items)} 처리 중...")
            print(f"   📋 키: {item.metadata.get('key', 'N/A')}")
            print(f"   📋 언어: {item.metadata.get('language', 'N/A')}")
            
            try:
                # 초기 상태 설정
                initial_state = {
                    "dataset_item": item,
                    "item_index": idx,
                    "run_name": run_name,
                    "messages": [],
                    "evaluation_results": [],
                    "current_turn": 1,
                    "max_turns": 0,
                    "trace_id": None,
                    "error": None,
                    "retry_count": 0,
                    "max_retries": 3
                }
                
                # 워크플로우 실행
                config = {"configurable": {"thread_id": f"eval_{idx}"}}
                final_state = self.workflow.invoke(initial_state, config)
                
                # 결과 처리
                if final_state.get("error"):
                    print(f"   ❌ 오류: {final_state['error']}")
                    results.append({
                        "item_index": idx,
                        "error": final_state["error"],
                        "results": []
                    })
                else:
                    print(f"   ✅ 완료: {len(final_state['evaluation_results'])}개 턴")
                    results.append({
                        "item_index": idx,
                        "error": None,
                        "results": final_state["evaluation_results"]
                    })
                    
            except Exception as e:
                print(f"   ❌ 예외 발생: {e}")
                results.append({
                    "item_index": idx,
                    "error": str(e),
                    "results": []
                })
        
        return results
    
    def _run_parallel_evaluation(self, items: List, run_name: str, max_workers: int) -> List[Dict]:
        """병렬 평가 실행"""
        def evaluate_item(item_data):
            item, idx = item_data
            try:
                initial_state = {
                    "dataset_item": item,
                    "item_index": idx,
                    "run_name": run_name,
                    "messages": [],
                    "evaluation_results": [],
                    "current_turn": 1,
                    "max_turns": 0,
                    "trace_id": None,
                    "error": None,
                    "retry_count": 0,
                    "max_retries": 3
                }
                
                config = {"configurable": {"thread_id": f"eval_{idx}"}}
                final_state = self.workflow.invoke(initial_state, config)
                
                if final_state.get("error"):
                    return {
                        "item_index": idx,
                        "error": final_state["error"],
                        "results": []
                    }
                else:
                    return {
                        "item_index": idx,
                        "error": None,
                        "results": final_state["evaluation_results"]
                    }
                    
            except Exception as e:
                return {
                    "item_index": idx,
                    "error": str(e),
                    "results": []
                }
        
        # 병렬 처리 실행
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            item_data = [(item, idx) for idx, item in enumerate(items, 1)]
            results = list(tqdm(
                executor.map(evaluate_item, item_data),
                total=len(items),
                desc="평가 진행"
            ))
        
        return results
    
    def save_results_to_file(self, results: List[Dict], output_path: str):
        """평가 결과를 파일로 저장"""
        try:
            serializable_results = []
            for result_item in results:
                if result_item.get("error"):
                    serializable_results.append({
                        "item_index": result_item["item_index"],
                        "error": result_item["error"],
                        "results": []
                    })
                else:
                    item_dict = {
                        "item_index": result_item["item_index"],
                        "error": None,
                        "results": []
                    }
                    
                    for result in result_item["results"]:
                        result_dict = {
                            "item_id": result.item_id,
                            "key": result.key,
                            "turn_index": result.turn_index,
                            "input": result.input.prompt,
                            "output": result.output,
                            "trace_id": result.trace_id,
                            "error": result.error
                        }
                        
                        if result.score:
                            result_dict["score"] = asdict(result.score)
                        
                        if result.result_strict:
                            result_dict["result_strict"] = asdict(result.result_strict)
                        
                        if result.result_loose:
                            result_dict["result_loose"] = asdict(result.result_loose)
                        
                        item_dict["results"].append(result_dict)
                    
                    serializable_results.append(item_dict)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 결과 저장 완료: {output_path}")
            
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Langfuse와 LangGraph를 활용한 멀티턴 Instruction Following 평가")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="평가할 Langfuse 데이터셋 이름")
    parser.add_argument("--run-name", "-r", type=str, default=f"langgraph_ifeval_{int(time.time())}", help="평가 실행 이름")
    parser.add_argument("--model", "-m", type=str, help="사용할 LLM 모델 이름")
    parser.add_argument("--temperature", "-t", type=float, default=0.6, help="생성 온도")
    parser.add_argument("--limit", "-l", type=int, help="평가할 최대 아이템 수")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    parser.add_argument("--output", "-o", type=str, help="결과를 저장할 파일 경로")
    parser.add_argument("--parallel", "-p", action="store_true", help="병렬 처리 사용")
    parser.add_argument("--workers", "-w", type=int, default=4, help="병렬 처리 작업자 수")
    
    args = parser.parse_args()
    
    # 평가기 초기화
    evaluator = LangGraphEvaluator(
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
    
    print("✅ LangGraph 평가 완료")

if __name__ == "__main__":
    main()
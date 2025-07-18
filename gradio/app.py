#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gradio 데모 앱: Langfuse 데이터셋 평가 인터페이스

이 데모는 다음 기능을 제공합니다:
1. Langfuse 서버에서 데이터셋 리스트 조회
2. 데이터셋 선택 시 아이템 리스트 표시
3. 특정 아이템 선택 시 input과 metadata 내용 표시
4. run 실행 시 turn별 평가 수행 및 결과 표시
"""

import os
import sys
import json
import time
import uuid
import gradio as gr
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# 환경변수 설정
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
WORK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ENV_FILE = os.path.join(WORK_DIR, ".env.params")

from dotenv import load_dotenv
load_dotenv(ENV_FILE)

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(WORK_DIR)

# Langfuse 관련 라이브러리
try:
    from langfuse import get_client, Langfuse
    from langfuse.langchain import CallbackHandler
except ImportError:
    raise ImportError("Error: Langfuse 라이브러리를 설치해주세요. (pip install langfuse)")

# LangChain 관련 라이브러리
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# IFEval 관련 라이브러리
try:
    sys.path.append(os.path.join(WORK_DIR, "ifeval"))
    from ifeval.evaluation_main import test_instruction_following_strict, test_instruction_following_loose
    from ifeval.evaluation_main import InputExample, OutputExample
except ImportError:
    raise ImportError("Warning: ifeval 모듈을 가져올 수 없습니다.")

# 기존 평가 클래스 임포트
sys.path.append(os.path.join(WORK_DIR, "scripts"))
from ifeval_langchain_with_langfuse import LangfuseEvaluator, EvaluationResult, EvaluationScore

class GradioEvaluationApp:
    """Gradio 평가 앱 클래스"""
    
    def __init__(self):
        """앱 초기화"""
        self.langfuse_client = None
        self.evaluator = None
        self.current_dataset = None
        self.current_item = None
        self.current_messages = []
        self.current_scores = []
        self.current_model = "gpt-4o-mini"
        
        # 사용 가능한 모델 리스트
        self.available_models = [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-3.5-turbo"
        ]
        
        # Langfuse 클라이언트 초기화
        self._initialize_langfuse()
        
        # 평가기 초기화 (기본 모델로)
        self._initialize_evaluator()
    
    def _initialize_langfuse(self):
        """Langfuse 클라이언트 초기화"""
        try:
            self.langfuse_client = get_client()
            auth_status = self.langfuse_client.auth_check()
            if not auth_status:
                raise Exception("Langfuse 인증 실패")
            print("✅ Langfuse 클라이언트 초기화 완료")
        except Exception as e:
            print(f"❌ Langfuse 초기화 실패: {e}")
            raise
    
    def _initialize_evaluator(self, model_name: str = None):
        """평가기 초기화"""
        try:
            if model_name:
                self.current_model = model_name
            
            self.evaluator = LangfuseEvaluator(
                model_name=self.current_model,
                temperature=0.6,
                verbose=False
            )
            print(f"✅ 평가기 초기화 완료 (모델: {self.current_model})")
        except Exception as e:
            print(f"❌ 평가기 초기화 실패: {e}")
            raise
    
    def get_dataset_list(self) -> List[str]:
        """데이터셋 리스트 가져오기"""
        try:
            # Langfuse API로 데이터셋 리스트 가져오기
            datasets = self.langfuse_client.api.datasets.list()
            return [dataset.name for dataset in datasets]
        except Exception as e:
            print(f"❌ 데이터셋 리스트 가져오기 실패: {e}")
            # 기본 데이터셋 리스트 반환
            return ["multi-if-ko", "General_MultiIF_English", "Telco_MultiIF_Korean"]
    
    def load_dataset(self, dataset_name: str) -> Tuple[List[str], str]:
        """데이터셋 로드"""
        if not dataset_name:
            return [], "데이터셋을 선택해주세요."
        
        try:
            self.current_dataset = self.evaluator.get_dataset(dataset_name)
            item_list = []
            for idx, item in enumerate(self.current_dataset.items):
                key = item.metadata.get("key", f"item_{idx}")
                language = item.metadata.get("language", "Unknown")
                item_list.append(f"{key} ({language})")
            
            return item_list, f"✅ 데이터셋 '{dataset_name}' 로드 완료 ({len(item_list)}개 아이템)"
        except Exception as e:
            return [], f"❌ 데이터셋 로드 실패: {e}"
    
    def select_item(self, item_selection: str) -> Tuple[str, str]:
        """아이템 선택"""
        if not item_selection or not self.current_dataset:
            return "", ""
        
        try:
            # 아이템 인덱스 추출
            item_idx = 0
            for idx, item in enumerate(self.current_dataset.items):
                key = item.metadata.get("key", f"item_{idx}")
                language = item.metadata.get("language", "Unknown")
                if f"{key} ({language})" == item_selection:
                    item_idx = idx
                    break
            
            self.current_item = self.current_dataset.items[item_idx]
            
            # input 내용 표시
            input_content = "### Input 내용:\n"
            for key, value in self.current_item.input.items():
                input_content += f"**{key}:**\n```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n\n"
            
            # metadata 내용 표시
            metadata_content = "### Metadata 내용:\n"
            metadata_content += f"```json\n{json.dumps(self.current_item.metadata, indent=2, ensure_ascii=False)}\n```"
            
            return input_content, metadata_content
        except Exception as e:
            return f"❌ 아이템 선택 실패: {e}", ""
    
    def change_model(self, model_name: str) -> str:
        """모델 변경"""
        try:
            self._initialize_evaluator(model_name)
            return f"✅ 모델을 '{model_name}'으로 변경했습니다."
        except Exception as e:
            return f"❌ 모델 변경 실패: {e}"
    
    def run_evaluation(self, progress=gr.Progress()) -> Tuple[List[Tuple[str, str]], str]:
        """평가 실행"""
        if not self.current_item:
            return [], "아이템을 선택해주세요."
        
        try:
            # 대화 기록 초기화
            self.current_messages = []
            self.current_scores = []
            chat_history = []
            
            # 턴 수 확인
            max_turn = 0
            for key in self.current_item.input:
                if key.startswith("turn_") and key[5:].isdigit():
                    turn_num = int(key[5:])
                    max_turn = max(max_turn, turn_num)
            
            if max_turn == 0:
                return [], "❌ 이 아이템에는 턴 정보가 없습니다."
            
            # 각 턴별로 평가 수행
            run_name = f"gradio_eval_{self.current_model}_{int(time.time())}"
            messages = []
            
            progress_step = 1.0 / max_turn
            
            for turn_index in range(1, max_turn + 1):
                progress(progress_step * turn_index, desc=f"Turn {turn_index}/{max_turn} 평가 중... (모델: {self.current_model})")
                
                # 턴별 평가 수행
                result, messages = self.evaluator.eval_process_by_turn(
                    self.current_item, 
                    turn_index, 
                    run_name, 
                    messages
                )
                
                # 채팅 기록 추가
                if len(messages) >= 2:
                    user_message = str(messages[-2].content)
                    ai_message = str(messages[-1].content)
                    
                    chat_history.append((user_message, ai_message))
                
                # 점수 기록
                self.current_scores.append({
                    "turn": turn_index,
                    "score": result.score,
                    "error": result.error
                })
            
            # 점수 표시 문자열 생성
            scores_display = self._format_scores()
            
            return chat_history, scores_display
            
        except Exception as e:
            return [], f"❌ 평가 실행 실패: {e}"
    
    def _format_scores(self) -> str:
        """점수 포맷팅"""
        if not self.current_scores:
            return "점수 정보가 없습니다."
        
        scores_text = f"### 턴별 점수 (모델: {self.current_model}):\n\n"
        
        for score_info in self.current_scores:
            turn = score_info["turn"]
            score = score_info["score"]
            error = score_info["error"]
            
            scores_text += f"**Turn {turn}:**\n"
            
            if error:
                scores_text += f"❌ 오류: {error}\n\n"
            else:
                scores_text += f"- Overall Score: {score.overall_score:.3f}\n"
                scores_text += f"- Strict Prompt Score: {score.strict_prompt_score:.3f}\n"
                scores_text += f"- Strict Instruction Score: {score.strict_inst_score:.3f}\n"
                scores_text += f"- Loose Prompt Score: {score.loose_prompt_score:.3f}\n"
                scores_text += f"- Loose Instruction Score: {score.loose_inst_score:.3f}\n\n"
        
        # 평균 점수 계산
        if self.current_scores:
            valid_scores = [s for s in self.current_scores if s["error"] is None]
            if valid_scores:
                avg_overall = sum(s["score"].overall_score for s in valid_scores) / len(valid_scores)
                avg_strict_prompt = sum(s["score"].strict_prompt_score for s in valid_scores) / len(valid_scores)
                avg_strict_inst = sum(s["score"].strict_inst_score for s in valid_scores) / len(valid_scores)
                avg_loose_prompt = sum(s["score"].loose_prompt_score for s in valid_scores) / len(valid_scores)
                avg_loose_inst = sum(s["score"].loose_inst_score for s in valid_scores) / len(valid_scores)
                
                scores_text += "### 평균 점수:\n"
                scores_text += f"- Overall Score: {avg_overall:.3f}\n"
                scores_text += f"- Strict Prompt Score: {avg_strict_prompt:.3f}\n"
                scores_text += f"- Strict Instruction Score: {avg_strict_inst:.3f}\n"
                scores_text += f"- Loose Prompt Score: {avg_loose_prompt:.3f}\n"
                scores_text += f"- Loose Instruction Score: {avg_loose_inst:.3f}\n"
        
        return scores_text

def create_gradio_app() -> gr.Blocks:
    """Gradio 앱 생성"""
    
    # 앱 인스턴스 생성
    app = GradioEvaluationApp()
    
    # 데이터셋 리스트 가져오기
    dataset_list = app.get_dataset_list()
    
    with gr.Blocks(title="MT-IF-Eval: Langfuse 데이터셋 평가 데모", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 MT-IF-Eval: Langfuse 데이터셋 평가 데모")
        gr.Markdown("이 데모는 Langfuse에서 mt-if-eval 데이터셋을 가져와 멀티턴 평가를 수행합니다.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 모델 선택
                model_dropdown = gr.Dropdown(
                    choices=app.available_models,
                    label="🤖 LLM 모델 선택",
                    value=app.current_model
                )
                
                model_status = gr.Textbox(
                    label="모델 상태",
                    value=f"현재 모델: {app.current_model}",
                    interactive=False
                )
                
                # 데이터셋 선택
                dataset_dropdown = gr.Dropdown(
                    choices=dataset_list,
                    label="📊 데이터셋 선택",
                    value=dataset_list[0] if dataset_list else None
                )
                
                dataset_status = gr.Textbox(
                    label="데이터셋 상태",
                    value="데이터셋을 선택해주세요.",
                    interactive=False
                )
                
                # 아이템 선택
                item_dropdown = gr.Dropdown(
                    choices=[],
                    label="📄 아이템 선택",
                    value=None
                )
                
                # 평가 실행 버튼
                run_button = gr.Button("🚀 평가 실행", variant="primary")
                
            with gr.Column(scale=2):
                # 아이템 정보 표시
                with gr.Tab("Input 내용"):
                    input_display = gr.Markdown("아이템을 선택하면 input 내용이 표시됩니다.")
                
                with gr.Tab("Metadata 내용"):
                    metadata_display = gr.Markdown("아이템을 선택하면 metadata 내용이 표시됩니다.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # 채팅 인터페이스
                chat_interface = gr.Chatbot(
                    label="💬 대화 기록",
                    height=800,
                    value=[]
                )
                
            with gr.Column(scale=1):
                # 점수 표시
                scores_display = gr.Markdown(
                    "### 📊 점수 정보\n평가를 실행하면 턴별 점수가 표시됩니다.",
                    height=400
                )
        
        # 이벤트 핸들러
        def on_model_change(model_name):
            status = app.change_model(model_name)
            return status
        
        def on_dataset_change(dataset_name):
            item_list, status = app.load_dataset(dataset_name)
            return gr.update(choices=item_list, value=None), status
        
        def on_item_change(item_selection):
            input_content, metadata_content = app.select_item(item_selection)
            return input_content, metadata_content
        
        def on_run_evaluation(progress=gr.Progress()):
            chat_history, scores = app.run_evaluation(progress)
            return chat_history, scores
        
        # 이벤트 바인딩
        model_dropdown.change(
            on_model_change,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        dataset_dropdown.change(
            on_dataset_change,
            inputs=[dataset_dropdown],
            outputs=[item_dropdown, dataset_status]
        )
        
        item_dropdown.change(
            on_item_change,
            inputs=[item_dropdown],
            outputs=[input_display, metadata_display]
        )
        
        run_button.click(
            on_run_evaluation,
            inputs=[],
            outputs=[chat_interface, scores_display]
        )
    
    return demo

if __name__ == "__main__":
    # 앱 실행
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )
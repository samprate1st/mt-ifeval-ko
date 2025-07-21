#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gradio Demo App: Langfuse Dataset Evaluation Interface

This demo provides the following features:
1. Retrieve dataset list from Langfuse server
2. Display item list when dataset is selected
3. Show input and metadata content when specific item is selected
4. Perform turn-by-turn evaluation and display results when run is executed
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

# Ignore warnings
warnings.filterwarnings("ignore")

# Environment variable setup
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
WORK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ENV_FILE = os.path.join(WORK_DIR, ".env.params")

from dotenv import load_dotenv
load_dotenv(ENV_FILE)

# Add project root directory to Python path
sys.path.append(WORK_DIR)

# Langfuse related libraries
try:
    from langfuse import get_client, Langfuse
    from langfuse.langchain import CallbackHandler
except ImportError:
    raise ImportError("Error: Please install Langfuse library. (pip install langfuse)")

# LangChain related libraries
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Define required data classes directly (instead of importing from evaluation_main.py)
@dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Any]]]

@dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]
    is_evaluated_list: list[bool] = None

# Import evaluation classes
try:
    # Add script directory with absolute path
    scripts_path = os.path.join(WORK_DIR, "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    
    # Import evaluation classes
    from scripts.ifeval_langchain_with_langfuse import LangfuseEvaluator, EvaluationResult, EvaluationScore
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}")

class GradioEvaluationApp:
    """Gradio evaluation app class"""
    
    def __init__(self):
        """Initialize app"""
        self.langfuse_client = None
        self.evaluator = None
        self.current_dataset = None
        self.current_item = None
        self.current_messages = []
        self.current_scores = []
        self.current_message_logs = []  # Store detailed message logs for each turn
        self.current_model = "gpt-4o-mini"
        
        # List of available models
        self.available_models = [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano"
        ]
        
        # Initialize Langfuse client
        self._initialize_langfuse()
        
        # Initialize evaluator (with default model)
        self._initialize_evaluator()
    
    def _initialize_langfuse(self):
        """Initialize Langfuse client"""
        try:
            self.langfuse_client = get_client()
            auth_status = self.langfuse_client.auth_check()
            if not auth_status:
                raise Exception("Langfuse authentication failed")
            print("✅ Langfuse client initialization completed")
        except Exception as e:
            print(f"❌ Langfuse initialization failed: {e}")
            raise
    
    def _initialize_evaluator(self, model_name: str = None):
        """Initialize evaluator"""
        try:
            if model_name:
                self.current_model = model_name
            
            self.evaluator = LangfuseEvaluator(
                model_name=self.current_model,
                temperature=0.6,
                verbose=False
            )
            print(f"✅ Evaluator initialization completed (model: {self.current_model})")
        except Exception as e:
            print(f"❌ Evaluator initialization failed: {e}")
            raise
    
    def get_dataset_list(self) -> List[str]:
        """Get dataset list"""
        try:
            # Get dataset list using Langfuse API
            datasets = self.langfuse_client.api.datasets.list()
            
            # PaginatedDatasets 형식 처리
            if hasattr(datasets, 'data') and isinstance(datasets.data, list):
                return [dataset.name for dataset in datasets.data if hasattr(dataset, 'name') and dataset.name not in ["RAG_Evaluation_Dataset"]]
            
            # 다른 형식인 경우 디버깅 정보 출력
            print(f"Dataset format not properly handled. Type: {type(datasets)}")
            
            # 기본값 반환
            return ["Google_IFEval_Korean", "META_General_MultiIF_English", "SKT_General_MultiIF_Korean", "SKT_Telco_MultiIF_Korean"]
        except Exception as e:
            print(f"❌ Failed to get dataset list: {e}")
            # Return default dataset list
            return ["Google_IFEval_Korean", "META_General_MultiIF_English", "SKT_General_MultiIF_Korean", "SKT_Telco_MultiIF_Korean"]
    
    def load_dataset(self, dataset_name: str) -> Tuple[List[str], str]:
        """Load dataset"""
        if not dataset_name:
            return [], "Please select a dataset."
        
        try:
            self.current_dataset = self.evaluator.get_dataset(dataset_name)
            item_list = []
            for idx, item in enumerate(self.current_dataset.items):
                key = item.metadata.get("key", f"item_{idx}")
                language = item.metadata.get("language", "Unknown")
                item_list.append(f"{key} ({language})")
            
            return item_list, f"✅ Dataset '{dataset_name}' loaded successfully ({len(item_list)} items)"
        except Exception as e:
            return [], f"❌ Dataset loading failed: {e}"
    
    def select_item(self, item_selection: str) -> Tuple[str, str, str, str, str, str, str, str, str, str]:
        """Select item"""
        if not item_selection or not self.current_dataset:
            return "", "", "", "", "", "", "", "", "", ""
        
        try:
            # Extract item index
            item_idx = 0
            for idx, item in enumerate(self.current_dataset.items):
                key = item.metadata.get("key", f"item_{idx}")
                language = item.metadata.get("language", "Unknown")
                if f"{key} ({language})" == item_selection:
                    item_idx = idx
                    break
            
            self.current_item = self.current_dataset.items[item_idx]
            
            # Get individual field contents for each turn
            def get_field_content(turn_data, field_name):
                if not turn_data or field_name not in turn_data:
                    return "" if field_name == "prompt" else "[]" if field_name == "instruction_id_list" else "{}"
                
                value = turn_data[field_name]
                if field_name == "prompt":
                    return str(value)
                else:
                    return json.dumps(value, indent=2, ensure_ascii=False)
            
            # Turn 1 fields
            turn1_data = self.current_item.input.get("turn_1", {})
            turn1_prompt = get_field_content(turn1_data, "prompt")
            turn1_instruction_ids = get_field_content(turn1_data, "instruction_id_list")
            turn1_kwargs = get_field_content(turn1_data, "kwargs")
            
            # Turn 2 fields
            turn2_data = self.current_item.input.get("turn_2", {})
            turn2_prompt = get_field_content(turn2_data, "prompt")
            turn2_instruction_ids = get_field_content(turn2_data, "instruction_id_list")
            turn2_kwargs = get_field_content(turn2_data, "kwargs")
            
            # Turn 3 fields
            turn3_data = self.current_item.input.get("turn_3", {})
            turn3_prompt = get_field_content(turn3_data, "prompt")
            turn3_instruction_ids = get_field_content(turn3_data, "instruction_id_list")
            turn3_kwargs = get_field_content(turn3_data, "kwargs")
            
            # Metadata
            metadata_content = json.dumps(self.current_item.metadata, indent=2, ensure_ascii=False)
            
            return (turn1_prompt, turn1_instruction_ids, turn1_kwargs,
                   turn2_prompt, turn2_instruction_ids, turn2_kwargs,
                   turn3_prompt, turn3_instruction_ids, turn3_kwargs,
                   metadata_content)
        except Exception as e:
            return f"❌ Item selection failed: {e}", "", "", "", "", "", "", "", "", ""
    
    def update_dataset_modification(self, 
                                    turn1_prompt: str, turn1_instruction_ids: str, turn1_kwargs: str,
                                    turn2_prompt: str, turn2_instruction_ids: str, turn2_kwargs: str,
                                    turn3_prompt: str, turn3_instruction_ids: str, turn3_kwargs: str,
                                    metadata_text: str) -> str:
        """Update dataset with modifications"""
        if not self.current_item:
            return "❌ No item selected. Please select an item first."
        
        try:
            # Helper function to build turn data
            def build_turn_data(prompt: str, instruction_ids: str, kwargs: str):
                turn_data = {}
                
                # Add prompt (as string)
                if prompt.strip():
                    turn_data["prompt"] = prompt.strip()
                
                # Add instruction_id_list (parse JSON)
                if instruction_ids.strip():
                    try:
                        turn_data["instruction_id_list"] = json.loads(instruction_ids.strip())
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON format for instruction_id_list: {instruction_ids}")
                
                # Add kwargs (parse JSON)
                if kwargs.strip():
                    try:
                        turn_data["kwargs"] = json.loads(kwargs.strip())
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON format for kwargs: {kwargs}")
                
                return turn_data
            
            # Build turn data
            turn1_data = build_turn_data(turn1_prompt, turn1_instruction_ids, turn1_kwargs)
            turn2_data = build_turn_data(turn2_prompt, turn2_instruction_ids, turn2_kwargs)
            turn3_data = build_turn_data(turn3_prompt, turn3_instruction_ids, turn3_kwargs)
            
            # Parse metadata
            metadata_data = json.loads(metadata_text) if metadata_text.strip() else {}
            
            # Update current item in memory
            self.current_item.input["turn_1"] = turn1_data
            self.current_item.input["turn_2"] = turn2_data
            self.current_item.input["turn_3"] = turn3_data
            self.current_item.metadata = metadata_data
            
            # Update in Langfuse dataset
            # Note: This requires implementing the Langfuse dataset update functionality
            # For now, we'll update the local dataset item
            for idx, item in enumerate(self.current_dataset.items):
                if item == self.current_item:
                    self.current_dataset.items[idx] = self.current_item
                    break
            
            return "✅ Dataset modification updated successfully in memory. Note: Langfuse server update not yet implemented."
            
        except ValueError as e:
            return f"❌ {e}"
        except json.JSONDecodeError as e:
            return f"❌ JSON parsing error: {e}. Please check your JSON format."
        except Exception as e:
            return f"❌ Update failed: {e}"
    
    def update_server_dataset(self) -> str:
        """Update dataset on Langfuse server"""
        if not self.current_item or not self.current_dataset:
            return "❌ No item or dataset selected. Please select an item first."
        
        try:
            # Get the dataset name
            dataset_name = None
            if hasattr(self.current_dataset, 'name'):
                dataset_name = self.current_dataset.name
            else:
                return "❌ Unable to determine dataset name"
            
            # Find the current item in the dataset to get its ID
            item_id = None
            if hasattr(self.current_item, 'id'):
                item_id = self.current_item.id
            elif hasattr(self.current_item, 'metadata'):
                item_id = self.current_item.metadata.get('id')
            
            if not item_id:
                return "❌ Unable to determine item ID for server update"
            
            # Try different approaches to update the dataset item
            try:
                # Method 1: Try using create_dataset_item (which might work for updates too)
                response = self.langfuse_client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=self.current_item.input,
                    metadata=self.current_item.metadata,
                    id=item_id  # This might update existing item if ID exists
                )
                return f"✅ Dataset item updated successfully on Langfuse server (Method 1, ID: {item_id})"
                
            except Exception as e1:
                # Method 2: Try direct API access using environment variables
                try:
                    import requests
                    import os
                    
                    # Get credentials from environment variables
                    public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
                    secret_key = os.getenv('LANGFUSE_SECRET_KEY')
                    host = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
                    
                    if not public_key or not secret_key:
                        return "❌ Missing Langfuse credentials (LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY)"
                    
                    # Prepare the update request
                    headers = {
                        'Content-Type': 'application/json'
                    }
                    
                    update_data = {
                        "input": self.current_item.input,
                        "metadata": self.current_item.metadata
                    }
                    
                    # Make the API call with basic auth
                    url = f"{host}/api/public/dataset-items/{item_id}"
                    response = requests.patch(
                        url, 
                        json=update_data, 
                        headers=headers,
                        auth=(public_key, secret_key)
                    )
                    
                    if response.status_code == 200:
                        return f"✅ Dataset item updated successfully on Langfuse server (Method 2, ID: {item_id})"
                    else:
                        return f"❌ Server update failed: HTTP {response.status_code} - {response.text}"
                        
                except Exception as e2:
                    # Method 3: Try to recreate the item (delete and create new)
                    try:
                        # This is a workaround - create a new dataset item
                        # Note: This might create a duplicate, but it's better than no update
                        response = self.langfuse_client.create_dataset_item(
                            dataset_name=dataset_name,
                            input=self.current_item.input,
                            metadata=self.current_item.metadata
                        )
                        return f"✅ Dataset item created as new item on Langfuse server (Method 3). Note: This creates a new item rather than updating the existing one."
                        
                    except Exception as e3:
                        return f"❌ All update methods failed:\nMethod 1: {e1}\nMethod 2: {e2}\nMethod 3: {e3}"
            
        except Exception as e:
            return f"❌ Server update failed: {e}"
    
    def change_model(self, model_name: str) -> str:
        """Change model"""
        try:
            self._initialize_evaluator(model_name)
            return f"✅ Model changed to '{model_name}'."
        except Exception as e:
            return f"❌ Model change failed: {e}"
    
    def run_evaluation(self, progress=gr.Progress()) -> Tuple[List[Tuple[str, str]], str, str]:
        """Run evaluation"""
        if not self.current_item:
            return [], "Please select an item.", ""
        
        try:
            # Initialize conversation history and message logs
            self.current_messages = []
            self.current_scores = []
            self.current_message_logs = []
            chat_history = []
            
            # Check number of turns
            max_turn = 0
            for key in self.current_item.input:
                if key.startswith("turn_") and key[5:].isdigit():
                    turn_num = int(key[5:])
                    max_turn = max(max_turn, turn_num)
            
            if max_turn == 0:
                return [], "❌ This item has no turn information."
            
            # Perform evaluation for each turn
            run_name = f"gradio_eval_{self.current_model}_{int(time.time())}"
            messages = [self.evaluator.system_message]
            
            progress_step = 1.0 / max_turn
            
            for turn_index in range(1, max_turn + 1):
                progress(progress_step * turn_index, desc=f"Evaluating Turn {turn_index}/{max_turn}... (Model: {self.current_model})")
                
                # Store messages before evaluation for logging
                messages_before = messages.copy() if messages else []
                
                # Perform turn-based evaluation
                result, messages = self.evaluator.eval_process_by_turn(
                    self.current_item, 
                    turn_index, 
                    run_name, 
                    messages
                )
                
                # Store message log for this turn
                self.current_message_logs.append({
                    "turn": turn_index,
                    "messages_before": messages_before,
                    "messages_after": messages.copy() if messages else []
                })
                
                # Add to chat history
                if len(messages) >= 2:
                    user_message = str(messages[-2].content)
                    ai_message = str(messages[-1].content)
                    
                    chat_history.append((user_message, ai_message))
                
                # Record scores with detailed results
                self.current_scores.append({
                    "turn": turn_index,
                    "score": result.score,
                    "error": result.error,
                    "result": result  # Store the full result object for detailed information
                })
            
            # Generate scores display string
            scores_display = self._format_scores()
            
            # Generate message logs display string
            logs_display = self._format_message_logs()
            
            return chat_history, scores_display, logs_display
            
        except Exception as e:
            return [], f"❌ Evaluation execution failed: {e}", ""
    
    def _format_scores(self) -> str:
        """Format scores"""
        if not self.current_scores:
            return "No score information available."
        
        scores_text = f"### Turn-by-Turn Scores (Model: {self.current_model}):\n\n"
        
        for score_info in self.current_scores:
            turn = score_info["turn"]
            score = score_info["score"]
            error = score_info["error"]
            result = score_info.get("result")
            
            scores_text += f"**Turn {turn}:**\n"
            
            if error:
                scores_text += f"❌ Error: {error}\n\n"
            elif score is None:
                scores_text += f"❌ No score data available\n\n"
            else:
                try:
                    # Display scores
                    scores_text += f"- Overall Score: {score.overall_score:.3f}\n"
                    scores_text += f"- Strict Prompt Score: {score.strict_prompt_score:.3f}\n"
                    scores_text += f"- Strict Instruction Score: {score.strict_inst_score:.3f}\n"
                    scores_text += f"- Loose Prompt Score: {score.loose_prompt_score:.3f}\n"
                    scores_text += f"- Loose Instruction Score: {score.loose_inst_score:.3f}\n"
                    
                    # Display detailed evaluation results if available
                    if result and hasattr(result, 'result_strict') and hasattr(result, 'result_loose'):
                        scores_text += f"\n**Detailed Results:**\n"
                        
                        # Strict evaluation results
                        if hasattr(result.result_strict, 'follow_instruction_list'):
                            is_followed_strict = result.result_strict.follow_instruction_list
                            scores_text += f"- is_followed (strict): {is_followed_strict}\n"
                        
                        if hasattr(result.result_strict, 'is_evaluated_list'):
                            is_evaluated_strict = result.result_strict.is_evaluated_list
                            scores_text += f"- is_evaluated (strict): {is_evaluated_strict}\n"
                        
                        # Loose evaluation results
                        if hasattr(result.result_loose, 'follow_instruction_list'):
                            is_followed_loose = result.result_loose.follow_instruction_list
                            scores_text += f"- is_followed (loose): {is_followed_loose}\n"
                        
                        if hasattr(result.result_loose, 'is_evaluated_list'):
                            is_evaluated_loose = result.result_loose.is_evaluated_list
                            scores_text += f"- is_evaluated (loose): {is_evaluated_loose}\n"
                    
                    scores_text += f"\n"
                    
                except AttributeError as e:
                    scores_text += f"❌ Score format error: {e}\n"
                    scores_text += f"Score object type: {type(score)}\n"
                    scores_text += f"Score object content: {score}\n"
                    if result:
                        scores_text += f"Result object type: {type(result)}\n"
                        scores_text += f"Result object attributes: {dir(result)}\n"
                    scores_text += f"\n"
        
        # Calculate average scores
        if self.current_scores:
            valid_scores = [s for s in self.current_scores if s["error"] is None]
            if valid_scores:
                avg_overall = sum(s["score"].overall_score for s in valid_scores) / len(valid_scores)
                avg_strict_prompt = sum(s["score"].strict_prompt_score for s in valid_scores) / len(valid_scores)
                avg_strict_inst = sum(s["score"].strict_inst_score for s in valid_scores) / len(valid_scores)
                avg_loose_prompt = sum(s["score"].loose_prompt_score for s in valid_scores) / len(valid_scores)
                avg_loose_inst = sum(s["score"].loose_inst_score for s in valid_scores) / len(valid_scores)
                
                scores_text += "### Average Scores:\n"
                scores_text += f"- Overall Score: {avg_overall:.3f}\n"
                scores_text += f"- Strict Prompt Score: {avg_strict_prompt:.3f}\n"
                scores_text += f"- Strict Instruction Score: {avg_strict_inst:.3f}\n"
                scores_text += f"- Loose Prompt Score: {avg_loose_prompt:.3f}\n"
                scores_text += f"- Loose Instruction Score: {avg_loose_inst:.3f}\n"
        
        return scores_text
    
    def _format_message_logs(self) -> str:
        """Format message logs"""
        if not self.current_message_logs:
            return "No message log information available."
        
        import pprint
        
        logs_text = f"### Turn-by-Turn Message Logs:\n\n"
        
        for log_info in self.current_message_logs:
            turn = log_info["turn"]
            messages_before = log_info["messages_before"]
            messages_after = log_info["messages_after"]
            
            logs_text += f"**Turn {turn}:**\n"
            
            # Show messages at the start of this turn (input to the turn)
            if messages_before:
                logs_text += f"**Messages Input to Turn {turn}:**\n"
                logs_text += "```\n"
                for i, msg in enumerate(messages_before):
                    logs_text += f"Message {i+1}:\n"
                    logs_text += f"  Type: {type(msg).__name__}\n"
                    logs_text += f"  Role: {getattr(msg, 'type', 'unknown')}\n"
                    logs_text += f"  Content: {getattr(msg, 'content', 'N/A')}\n"
                    logs_text += "\n"
                logs_text += "```\n"
            else:
                logs_text += f"**Messages Input to Turn {turn}:** (empty - first turn)\n"
            
            # Show the new messages added during this turn
            if len(messages_after) > len(messages_before):
                new_messages = messages_after[len(messages_before):]
                logs_text += f"**New Messages Added in Turn {turn}:**\n"
                logs_text += "```\n"
                for i, msg in enumerate(new_messages):
                    logs_text += f"New Message {i+1}:\n"
                    logs_text += f"  Type: {type(msg).__name__}\n"
                    logs_text += f"  Role: {getattr(msg, 'type', 'unknown')}\n"
                    logs_text += f"  Content: {getattr(msg, 'content', 'N/A')}\n"
                    logs_text += "\n"
                logs_text += "```\n"
            
            logs_text += "\n" + "-"*50 + "\n\n"
        
        return logs_text

def create_gradio_app() -> gr.Blocks:
    """Create Gradio app"""
    
    # Create app instance
    app = GradioEvaluationApp()
    
    # Get dataset list
    dataset_list = app.get_dataset_list()
    
    with gr.Blocks(title="MT-IF-Eval: Langfuse Dataset Evaluation Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 MT-IF-Eval: Langfuse Dataset Evaluation Demo")
        gr.Markdown("This demo fetches mt-if-eval dataset from Langfuse and performs multi-turn evaluation.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=app.available_models,
                    label="🤖 Select LLM Model",
                    value=app.current_model
                )
                
                model_status = gr.Textbox(
                    label="Model Status",
                    value=f"Current model: {app.current_model}",
                    interactive=False
                )
                
                # Dataset selection
                dataset_dropdown = gr.Dropdown(
                    choices=dataset_list,
                    label="📊 Select Dataset",
                    value=dataset_list[0] if dataset_list else None
                )
                
                dataset_status = gr.Textbox(
                    label="Dataset Status",
                    value="Please select a dataset.",
                    interactive=False
                )
                
                # Item selection
                item_dropdown = gr.Dropdown(
                    choices=[],
                    label="📄 Select Item",
                    value=None
                )
                
                # Evaluation run button
                run_button = gr.Button("🚀 Run Evaluation", variant="primary")
                
            with gr.Column(scale=2):
                # Display item information with editable tabs
                with gr.Tab("Turn 1"):
                    turn1_prompt = gr.Textbox(
                        label="Prompt",
                        value="",
                        lines=5,
                        max_lines=10,
                        interactive=True,
                        placeholder="Select an item to edit Turn 1 prompt..."
                    )
                    turn1_instruction_ids = gr.Textbox(
                        label="Instruction ID List (JSON format)",
                        value="",
                        lines=3,
                        max_lines=5,
                        interactive=True,
                        placeholder="Select an item to edit Turn 1 instruction IDs..."
                    )
                    turn1_kwargs = gr.Textbox(
                        label="Kwargs (JSON format)",
                        value="",
                        lines=7,
                        max_lines=10,
                        interactive=True,
                        placeholder="Select an item to edit Turn 1 kwargs..."
                    )
                
                with gr.Tab("Turn 2"):
                    turn2_prompt = gr.Textbox(
                        label="Prompt",
                        value="",
                        lines=5,
                        max_lines=10,
                        interactive=True,
                        placeholder="Select an item to edit Turn 2 prompt..."
                    )
                    turn2_instruction_ids = gr.Textbox(
                        label="Instruction ID List (JSON format)",
                        value="",
                        lines=3,
                        max_lines=5,
                        interactive=True,
                        placeholder="Select an item to edit Turn 2 instruction IDs..."
                    )
                    turn2_kwargs = gr.Textbox(
                        label="Kwargs (JSON format)",
                        value="",
                        lines=7,
                        max_lines=10,
                        interactive=True,
                        placeholder="Select an item to edit Turn 2 kwargs..."
                    )
                
                with gr.Tab("Turn 3"):
                    turn3_prompt = gr.Textbox(
                        label="Prompt",
                        value="",
                        lines=5,
                        max_lines=10,
                        interactive=True,
                        placeholder="Select an item to edit Turn 3 prompt..."
                    )
                    turn3_instruction_ids = gr.Textbox(
                        label="Instruction ID List (JSON format)",
                        value="",
                        lines=3,
                        max_lines=5,
                        interactive=True,
                        placeholder="Select an item to edit Turn 3 instruction IDs..."
                    )
                    turn3_kwargs = gr.Textbox(
                        label="Kwargs (JSON format)",
                        value="",
                        lines=7,
                        max_lines=10,
                        interactive=True,
                        placeholder="Select an item to edit Turn 3 kwargs..."
                    )
                
                with gr.Tab("Metadata"):
                    metadata_textbox = gr.Textbox(
                        label="Metadata (JSON format)",
                        value="",
                        lines=15,
                        max_lines=20,
                        interactive=True,
                        placeholder="Select an item to edit metadata..."
                    )
                
                # Update dataset modification button and status (horizontal layout)
                with gr.Row():
                    update_button = gr.Button("💾 Update Dataset Modification", variant="secondary", scale=2)
                    update_status = gr.Textbox(
                        label="Update Status",
                        value="",
                        interactive=False,
                        scale=3
                    )
                
                # Update server dataset button and status (horizontal layout)
                with gr.Row():
                    server_update_button = gr.Button("🌐 Update Server Dataset", variant="secondary", scale=2)
                    server_update_status = gr.Textbox(
                        label="Server Update Status",
                        value="",
                        interactive=False,
                        scale=3
                    )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface (flexible height with minimum)
                chat_interface = gr.Chatbot(
                    label="💬 Conversation History",
                    height=1200,  # Minimum height, will expand as needed
                    value=[],
                    show_copy_button=True,
                    container=True,
                    show_share_button=False
                )
                
            with gr.Column(scale=1):
                # Score display (flexible height with minimum)
                scores_display = gr.Markdown(
                    "### 📊 Score Information\nRun evaluation to display turn-by-turn scores.",
                    height=1200,  # Minimum height, will expand with content
                    container=True
                )
        
        # Log display section at the bottom
        with gr.Row():
            with gr.Column():
                logs_display = gr.Markdown(
                    "### 📋 Message Logs\nRun evaluation to display turn-by-turn message logs.",
                    height=1000,  # Height for log display
                    container=True
                )
        
        # Event handlers
        def on_model_change(model_name):
            status = app.change_model(model_name)
            return status
        
        def on_dataset_change(dataset_name):
            item_list, status = app.load_dataset(dataset_name)
            return gr.update(choices=item_list, value=None), status
        
        def on_item_change(item_selection):
            result = app.select_item(item_selection)
            return result  # Returns 10 values: turn1_prompt, turn1_instruction_ids, turn1_kwargs, etc.
        
        def on_update_dataset(turn1_prompt, turn1_instruction_ids, turn1_kwargs,
                             turn2_prompt, turn2_instruction_ids, turn2_kwargs,
                             turn3_prompt, turn3_instruction_ids, turn3_kwargs,
                             metadata_text):
            status = app.update_dataset_modification(turn1_prompt, turn1_instruction_ids, turn1_kwargs,
                                                   turn2_prompt, turn2_instruction_ids, turn2_kwargs,
                                                   turn3_prompt, turn3_instruction_ids, turn3_kwargs,
                                                   metadata_text)
            return status
        
        def on_server_update():
            status = app.update_server_dataset()
            return status
        
        def on_run_evaluation(progress=gr.Progress()):
            chat_history, scores, logs = app.run_evaluation(progress)
            return chat_history, scores, logs
        
        # Event binding
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
            outputs=[turn1_prompt, turn1_instruction_ids, turn1_kwargs,
                    turn2_prompt, turn2_instruction_ids, turn2_kwargs,
                    turn3_prompt, turn3_instruction_ids, turn3_kwargs,
                    metadata_textbox]
        )
        
        update_button.click(
            on_update_dataset,
            inputs=[turn1_prompt, turn1_instruction_ids, turn1_kwargs,
                   turn2_prompt, turn2_instruction_ids, turn2_kwargs,
                   turn3_prompt, turn3_instruction_ids, turn3_kwargs,
                   metadata_textbox],
            outputs=[update_status]
        )
        
        server_update_button.click(
            on_server_update,
            inputs=[],
            outputs=[server_update_status]
        )
        
        run_button.click(
            on_run_evaluation,
            inputs=[],
            outputs=[chat_interface, scores_display, logs_display]
        )
    
    return demo

if __name__ == "__main__":
    # Run app
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )

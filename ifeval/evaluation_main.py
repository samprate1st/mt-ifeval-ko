# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
import json
import os
from typing import Dict, Optional, Sequence, Union

from absl import app
from absl import flags
from absl import logging

import ifeval.instructions_registry
from ifeval import evaluation_lib


_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=True,
)

_MODEL_NAME = flags.DEFINE_string(
    "model_name", None, "model name", required=True
)

_RESULTS_FNAME = flags.DEFINE_string(
    "results_fname", None, "path to results.json file", required=True
)


@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: list[bool]
  is_evaluated_list: list[bool] = None  # 각 instruction이 실제로 평가되었는지 여부


def read_prompt_list(input_jsonl_filename):
  """Read inputs from jsonl."""
  inputs = []
  with open(input_jsonl_filename, "r", encoding='utf-8') as f:
    for l in f:
      example = json.loads(l)
      inputs.append(
          InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"]))
  return inputs


def write_outputs(output_jsonl_filename, outputs):
  """Writes outputs to jsonl."""
  assert outputs
  
  # Ensure the entire directory path exists
  directory = os.path.dirname(output_jsonl_filename)
  if directory and not os.path.exists(directory):
      os.makedirs(directory, exist_ok=True)
  
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      f.write(
          json.dumps(
              {
                  attr_name: o.__getattribute__(attr_name)
                  for attr_name in [
                      name for name in dir(o) if not name.startswith("_")
                  ]
              },
              ensure_ascii=False)
      )
      f.write("\n")


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
  """Tests response to see if instrutions are followed."""
  response = prompt_to_response[inp.prompt]
  instruction_list = inp.instruction_id_list
  is_following_list = []
  is_evaluated_list = []  # 각 instruction이 실제로 평가되었는지 추적

  for index, instruction_id in enumerate(instruction_list):
    # 언어 접두어 추출 (예: "ko:keywords:existence" -> "ko")
    lang_prefix = instruction_id.split(":")[0]
    
    try:
      # 해당 언어의 instruction 클래스 사용 시도
      instruction_cls = ifeval.instructions_registry.INSTRUCTION_DICT[instruction_id]
    except KeyError:
      # 언어별 instruction이 없는 경우, 영어 버전 사용
      # 원래 instruction ID에서 언어 부분만 'en'으로 변경
      parts = instruction_id.split(":", 1)
      if len(parts) > 1:
        en_instruction_id = "en:" + parts[1]
        logging.warning(
            f"지원되지 않는 instruction ID: {instruction_id}, 대신 영어 버전 사용: {en_instruction_id}"
        )
        try:
          instruction_cls = ifeval.instructions_registry.INSTRUCTION_DICT[en_instruction_id]
        except KeyError:
          logging.error(f"영어 버전도 지원되지 않음: {en_instruction_id}")
          # 지원되지 않는 instruction은 평가에서 제외
          is_following_list.append(False)
          is_evaluated_list.append(False)  # 평가되지 않음으로 표시
          continue
      else:
        logging.error(f"잘못된 instruction ID 형식: {instruction_id}")
        is_following_list.append(False)
        is_evaluated_list.append(False)  # 평가되지 않음으로 표시
        continue
    
    instruction = instruction_cls(instruction_id)
    
    try:
      # 안전하게 kwargs 전달 - 호환되지 않는 매개변수 처리
      # 인스트럭션이 필요로 하는 인자만 전달
      compatible_kwargs = {}
      
      # 원본 kwargs에서 인스트럭션이 지원하는 매개변수만 추출
      original_kwargs = inp.kwargs[index]
      
      # instruction.build_description의 필요 매개변수 정보 가져오기
      import inspect
      build_description_params = inspect.signature(instruction.build_description).parameters
      
      # 호환되는 매개변수만 추출
      for param_name, param_value in original_kwargs.items():
        if param_name in build_description_params:
          compatible_kwargs[param_name] = param_value
      
      # 호환되는 매개변수만 사용하여 인스트럭션 구성
      instruction.build_description(**compatible_kwargs)
      
      args = instruction.get_instruction_args()
      if args and "prompt" in args:
        instruction.build_description(prompt=inp.prompt)

      if isinstance(response, str) and response.strip() and instruction.check_following(response):
        is_following_list.append(True)
      else:
        is_following_list.append(False)
      
      is_evaluated_list.append(True)  # 평가 성공으로 표시
      
    except (TypeError, ValueError) as e:
      # 매개변수 불일치 등으로 인한 오류 처리
      logging.warning(f"Instruction {instruction_id} 평가 중 오류 발생: {str(e)}")
      is_following_list.append(False)
      is_evaluated_list.append(False)  # 평가되지 않음으로 표시

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list) if all(is_evaluated_list) else False,
      follow_instruction_list=is_following_list,
      is_evaluated_list=is_evaluated_list  # 새로운 필드 추가
  )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
  """Tests response for an upper bound for following instructions."""
  response = prompt_to_response[inp.prompt]
  if isinstance(response, str):
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_quotation = response.replace('\"', '')
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
    ]
  else:
    all_responses=[]
    
  instruction_list = inp.instruction_id_list
  is_following_list = []
  is_evaluated_list = []  # 각 instruction이 실제로 평가되었는지 추적

  for index, instruction_id in enumerate(instruction_list):
    # 언어 접두어 추출
    lang_prefix = instruction_id.split(":")[0]
    
    try:
      # 해당 언어의 instruction 클래스 사용 시도
      instruction_cls = ifeval.instructions_registry.INSTRUCTION_DICT[instruction_id]
    except KeyError:
      # 언어별 instruction이 없는 경우, 영어 버전 사용
      parts = instruction_id.split(":", 1)
      if len(parts) > 1:
        en_instruction_id = "en:" + parts[1]
        logging.warning(
            f"지원되지 않는 instruction ID: {instruction_id}, 대신 영어 버전 사용: {en_instruction_id}"
        )
        try:
          instruction_cls = ifeval.instructions_registry.INSTRUCTION_DICT[en_instruction_id]
        except KeyError:
          logging.error(f"영어 버전도 지원되지 않음: {en_instruction_id}")
          # 지원되지 않는 instruction은 평가에서 제외
          is_following_list.append(False)
          is_evaluated_list.append(False)  # 평가되지 않음으로 표시
          continue
      else:
        logging.error(f"잘못된 instruction ID 형식: {instruction_id}")
        is_following_list.append(False)
        is_evaluated_list.append(False)  # 평가되지 않음으로 표시
        continue
    
    try:
      instruction = instruction_cls(instruction_id)
      
      # 안전하게 kwargs 전달 - 호환되지 않는 매개변수 처리
      # 인스트럭션이 필요로 하는 인자만 전달
      compatible_kwargs = {}
      
      # 원본 kwargs에서 인스트럭션이 지원하는 매개변수만 추출
      original_kwargs = inp.kwargs[index]
      
      # instruction.build_description의 필요 매개변수 정보 가져오기
      import inspect
      build_description_params = inspect.signature(instruction.build_description).parameters
      
      # 호환되는 매개변수만 추출
      for param_name, param_value in original_kwargs.items():
        if param_name in build_description_params:
          compatible_kwargs[param_name] = param_value
      
      # 호환되는 매개변수만 사용하여 인스트럭션 구성
      instruction.build_description(**compatible_kwargs)
      
      args = instruction.get_instruction_args()
      if args and "prompt" in args:
        instruction.build_description(prompt=inp.prompt)

      is_following = False
      for r in all_responses:
        if r.strip() and instruction.check_following(r):
          is_following = True
          break

      is_following_list.append(is_following)
      is_evaluated_list.append(True)  # 평가 성공으로 표시
      
    except (TypeError, ValueError) as e:
      # 매개변수 불일치 등으로 인한 오류 처리
      logging.warning(f"Instruction {instruction_id} 평가 중 오류 발생: {str(e)}")
      is_following_list.append(False)
      is_evaluated_list.append(False)  # 평가되지 않음으로 표시

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list) if all(is_evaluated_list) else False,
      follow_instruction_list=is_following_list,
      is_evaluated_list=is_evaluated_list  # 새로운 필드 추가
  )


def read_prompt_to_response_dict(input_jsonl_filename):
  """Creates dictionary matching prompt and response."""
  return_dict = {}
  with open(input_jsonl_filename, "r", encoding='utf-8') as f:
    for l in f:
      example = json.loads(l)
      return_dict[example["prompt"]] = example["response"]
  return return_dict


def print_report(outputs):
  """Prints a report on accuracy scores."""

  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)

  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example.follow_instruction_list
    instruction_id_list = example.instruction_id_list
    is_evaluated_list = example.is_evaluated_list if example.is_evaluated_list else [True] * len(follow_instruction_list)

    # 모든 instruction이 평가된 경우에만 prompt 레벨 정확도에 포함
    if all(is_evaluated_list):
      prompt_total += 1
      if all(follow_instruction_list):
        prompt_correct += 1

    # 평가된 instruction만 정확도 계산에 포함
    for instruction_id, followed_or_not, is_evaluated in zip(
        instruction_id_list, follow_instruction_list, is_evaluated_list
    ):
      if is_evaluated:  # 평가된 instruction만 계산에 포함
        instruction_total += 1
        if followed_or_not:
          instruction_correct += 1

        # tier0 레벨 통계 (언어 접두어 기준)
        tier0_id = instruction_id.split(":")[0]
        tier0_total[tier0_id] += 1
        if followed_or_not:
          tier0_correct[tier0_id] += 1

        # tier1 레벨 통계 (전체 instruction_id 기준)
        tier1_total[instruction_id] += 1
        if followed_or_not:
          tier1_correct[instruction_id] += 1

  # 통계 출력
  if prompt_total > 0:
    print(f"prompt-level: {prompt_correct / prompt_total}")
  else:
    print("prompt-level: 평가 가능한 prompt 없음")
    
  if instruction_total > 0:
    print(f"instruction-level: {instruction_correct / instruction_total}")
  else:
    print("instruction-level: 평가 가능한 instruction 없음")
    
  print()
  
  # tier0 (언어 접두어) 레벨 통계
  for instruction_id in sorted(tier0_total.keys()):
    accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
    print(f"{instruction_id} {accuracy}")
  
  print()
  
  # tier1 (전체 instruction_id) 레벨 통계
  for instruction_id in sorted(tier1_total.keys()):
    accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
    print(f"{instruction_id} {accuracy}")


def load_params(params_file: str) -> dict:
    """Load parameters from params.json file.
    
    Args:
        params_file: Path to params.json file
        
    Returns:
        Dictionary containing parameters
        
    Raises:
        FileNotFoundError: If params file does not exist
        json.JSONDecodeError: If params file is not valid JSON
    """
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Params file not found: {params_file}")
    
    with open(params_file, 'r') as f:
        return json.load(f)


def get_safe_model_name(model_name: str) -> str:
    """Convert model name to a safe filename.
    
    Args:
        model_name: Original model name
        
    Returns:
        Safe filename version of model name
    """
    # Replace problematic characters with underscores
    safe_name = model_name.replace('/', '__').replace(':', '_')
    return safe_name


def save_results_to_json(
    model_name: str,
    scores_fname: Dict[str, str],
    language: str,
    results_fname: str = "results.json"
) -> None:
    """평가 결과를 results.json 파일에 저장합니다."""

    # 모든 언어 지원
    dataset_name = f"{language}_input_data"

    try:
        # score 파일 읽기
        with open(scores_fname["score_results_strict"], 'r', encoding='utf-8') as f:
            strict_scores = json.load(f)
        with open(scores_fname["score_results_loose"], 'r', encoding='utf-8') as f:
            loose_scores = json.load(f)

        results = [{
            "model_name": model_name,
            "dataset_category": "IFEval",
            "dataset_task": "mifeval/train",
            "dataset_name": f"{dataset_name}",
            "metrics": {
                "prompt_level_strict_acc": strict_scores["prompt_level_accuracy"],
                "inst_level_strict_acc": strict_scores["instruction_level_accuracy"],
                "prompt_level_loose_acc": loose_scores["prompt_level_accuracy"],
                "inst_level_loose_acc": loose_scores["instruction_level_accuracy"]
            }
        }]

        # 파일에 저장
        with open(results_fname, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logging.info(f"결과가 {results_fname}에 저장되었습니다.")
    except Exception as e:
        logging.error(f"결과 저장 중 오류 발생: {str(e)}")


def main(argv):
    """Main function for evaluation."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # output 폴더 존재 여부 확인 및 생성
    output_dir = "./output"
    if not os.path.exists(output_dir):
        print(f"output 폴더가 존재하지 않습니다. 생성합니다: {output_dir}")
        os.makedirs(output_dir)

    # Load parameters
    safe_model_name = get_safe_model_name(_MODEL_NAME.value)
    logging.info("Using model: %s", _MODEL_NAME.value)

    # Check input file exists
    if not os.path.exists(_INPUT_DATA.value):
        raise FileNotFoundError(
            f"Input data file NOT found: {_INPUT_DATA.value}"
        )
    else:
        logging.info("Input data file: %s", _INPUT_DATA.value)
        language = _INPUT_DATA.value.split("/")[-1].split("_")[0]
        
    # 응답 데이터 파일 존재 여부 확인
    if _INPUT_RESPONSE_DATA.value and not os.path.exists(_INPUT_RESPONSE_DATA.value):
      raise FileNotFoundError(
          f"Response data file not found: {_INPUT_RESPONSE_DATA.value}"
      )

    # 출력 디렉토리 생성
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

    logging.info("Reading input data from: %s", _INPUT_DATA.value)
    inputs = read_prompt_list(_INPUT_DATA.value)
    logging.info("Total number of prompts: %d", len(inputs))

    logging.info("Reading response data from: %s", _INPUT_RESPONSE_DATA.value)
    prompt_to_response = read_prompt_to_response_dict(_INPUT_RESPONSE_DATA.value)
    logging.info("Total number of responses: %d", len(prompt_to_response))

    # 모든 프롬프트가 응답 데이터에 있는지 확인
    missing_prompts = [inp for inp in inputs if inp.prompt not in prompt_to_response]
    
    if missing_prompts:
      logging.warning("Missing responses for %d prompts", len(missing_prompts))
      print("=" * 64)
      print(f"응답이 없는 프롬프트 목록 ({len(missing_prompts)}개):")
      for mp in missing_prompts:
        print(f"Key: {mp.key}, Prompt 시작 부분: {mp.prompt[:50]}...")
      print("=" * 64)
      
      # 응답이 있는 입력만 필터링
      filtered_inputs = [inp for inp in inputs if inp.prompt in prompt_to_response]
      logging.info("필터링 후 평가 가능한 프롬프트 수: %d", len(filtered_inputs))
      
      # 필터링된 입력이 없으면 종료
      if not filtered_inputs:
        logging.error("응답이 있는 프롬프트가 없습니다. 평가를 종료합니다.")
        return
      
      # 필터링된 입력으로 대체
      inputs = filtered_inputs

    # get instruction following results
    for func, output_file_name, score_file_name in [
        (test_instruction_following_strict, "eval_results_strict", "score_results_strict"),
        (test_instruction_following_loose, "eval_results_loose", "score_results_loose"),
    ]:
      logging.info("Generating %s...", output_file_name)
      outputs = []
      for inp in inputs:
        outputs.append(func(inp, prompt_to_response))
      follow_all_instructions = [o.follow_all_instructions for o in outputs]
      accuracy = sum(follow_all_instructions) / len(outputs)
      logging.info("Accuracy: %f", accuracy)

      # 파일 이름에 언어 접두사 추가
      output_file_name = os.path.join(
          _OUTPUT_DIR.value, 
          f"{language}_{output_file_name}.jsonl"
      )
      write_outputs(output_file_name, outputs)
      logging.info("Generated: %s", output_file_name)

      # Prints instruction following accuracy report.
      print("=" * 64)
      print(f"{output_file_name} Accuracy Scores:")
      print_report(outputs)

      # 파일 이름에 언어 접두사 추가
      score_file_name = os.path.join(
          _OUTPUT_DIR.value, 
          f"{language}_{score_file_name}.json"
      )
      evaluation_lib.save_report(outputs, score_file_name)

    # 평가 결과 파일 경로 설정 (언어 접두사 추가)
    scores_fname = {
      "score_results_strict": os.path.join(_OUTPUT_DIR.value, f"{language}_score_results_strict.json"),
      "score_results_loose": os.path.join(_OUTPUT_DIR.value, f"{language}_score_results_loose.json")
      }

    # 언어별 결과 파일 저장
    save_results_to_json(
      _MODEL_NAME.value, 
      scores_fname,
      language,
      results_fname=os.path.join(_OUTPUT_DIR.value, f"{language}_results.json")
    )
    logging.info("Language results saved to: %s", os.path.join(_OUTPUT_DIR.value, f"{language}_results.json"))

    # 전체 결과 파일 저장
    save_results_to_json(
      _MODEL_NAME.value, 
      scores_fname,
      language,
      results_fname=os.path.join(_OUTPUT_DIR.value, "results.json")
    )
    logging.info("Results saved to: %s", os.path.join(_OUTPUT_DIR.value, "results.json"))
    logging.info("Evaluation completed.")

    save_results_to_json(
      _MODEL_NAME.value, 
      scores_fname,
      language,
      results_fname=_RESULTS_FNAME.value
    )
    logging.info("Results for EaaS saved to: %s", _RESULTS_FNAME.value)


if __name__ == "__main__":
  app.run(main)

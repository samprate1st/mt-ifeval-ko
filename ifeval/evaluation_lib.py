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
from typing import Dict, Optional, Union
import logging
import os

from ifeval import instructions_registry


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
  is_evaluated_list: list[bool]

@dataclasses.dataclass
class ScoreExample:
  prompt_level_accuracy: float
  instruction_level_accuracy: float


def read_prompt_list(input_jsonl_filename):
  """Read inputs from jsonl."""
  inputs = []
  with open(input_jsonl_filename, "r") as f:
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
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      f.write(
          json.dumps(
              {
                  attr_name: o.__getattribute__(attr_name)
                  for attr_name in [
                      name for name in dir(o) if not name.startswith("_")
                  ]
              }
          )
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

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    if response.strip() and instruction.check_following(response):
      is_following_list.append(True)
    else:
      is_following_list.append(False)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
      is_evaluated_list=[True] * len(is_following_list),
  )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
  """Tests response for an upper bound for following instructions."""
  response = prompt_to_response[inp.prompt]
  r = response.split("\n")
  response_remove_first = "\n".join(r[1:]).strip()
  response_remove_last = "\n".join(r[:-1]).strip()
  response_remove_both = "\n".join(r[1:-1]).strip()
  revised_response = response.replace("*", "")
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
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
      is_evaluated_list=[True] * len(is_following_list),
  )


def read_prompt_to_response_dict(input_jsonl_filename):
  """Creates dictionary matching prompt and response."""
  return_dict = {}
  with open(input_jsonl_filename, "r") as f:
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

    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      instruction_id = instruction_id.split(":")[0]
      tier0_total[instruction_id] += 1
      if followed_or_not:
        tier0_correct[instruction_id] += 1

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      tier1_total[instruction_id] += 1
      if followed_or_not:
        tier1_correct[instruction_id] += 1

  # 통계 계산
  prompt_level_accuracy = prompt_correct / prompt_total if prompt_total > 0 else 0
  instruction_level_accuracy = instruction_correct / instruction_total if instruction_total > 0 else 0

  print(f"prompt-level: {prompt_level_accuracy}")
  print(f"instruction-level: {instruction_level_accuracy}")
  print()
  for instruction_id in sorted(tier0_total.keys()):
    accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
    print(f"{instruction_id} {accuracy}")
  print()
  for instruction_id in sorted(tier1_total.keys()):
    accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
    print(f"{instruction_id} {accuracy}")
  print()

  return ScoreExample(
      prompt_level_accuracy=prompt_level_accuracy,
      instruction_level_accuracy=instruction_level_accuracy
  )


def save_report(outputs, output_file):
  """Saves the report to a JSON file."""
  
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
    is_evaluated_list = example.is_evaluated_list if hasattr(example, 'is_evaluated_list') and example.is_evaluated_list else [True] * len(follow_instruction_list)

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

        # tier 레벨 통계 계산
        instruction_id_split = instruction_id.split(":")
        if len(instruction_id_split) > 0:
          tier0_id = instruction_id_split[0]  # 언어 접두어
          tier0_total[tier0_id] += 1
          if followed_or_not:
            tier0_correct[tier0_id] += 1
        
        tier1_total[instruction_id] += 1
        if followed_or_not:
          tier1_correct[instruction_id] += 1

  # 통계 계산
  prompt_level_accuracy = prompt_correct / prompt_total if prompt_total > 0 else 0
  instruction_level_accuracy = instruction_correct / instruction_total if instruction_total > 0 else 0
  
  # 각 tier별 통계
  tier0_accuracy = {id: tier0_correct[id] / tier0_total[id] for id in tier0_total}
  tier1_accuracy = {id: tier1_correct[id] / tier1_total[id] for id in tier1_total}
  
  # 결과 저장
  results = {
      "prompt_level_accuracy": prompt_level_accuracy,
      "instruction_level_accuracy": instruction_level_accuracy,
      "tier0_accuracy": tier0_accuracy,
      "tier1_accuracy": tier1_accuracy,
      "total_prompts": prompt_total,
      "correct_prompts": prompt_correct,
      "total_instructions": instruction_total,
      "correct_instructions": instruction_correct,
      "evaluated_prompts_ratio": prompt_total / len(outputs) if len(outputs) > 0 else 0,
  }
  
  # 출력 디렉토리 생성
  directory = os.path.dirname(output_file)
  if directory and not os.path.exists(directory):
      os.makedirs(directory, exist_ok=True)
  
  with open(output_file, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2, ensure_ascii=False)
  
  logging.info("Report saved to %s", output_file)

  return ScoreExample(
      prompt_level_accuracy=prompt_level_accuracy,
      instruction_level_accuracy=instruction_level_accuracy
  )

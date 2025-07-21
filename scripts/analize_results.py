#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

def load_results(input_file: str) -> List[Dict[str, Any]]:
    """Load evaluation results from JSON file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_accuracy_for_turn(items: List[Dict[str, Any]], turn_index: int) -> Dict[str, float]:
    """Calculate accuracy metrics for a specific turn"""
    
    # Filter items for specific turn
    turn_items = [item for item in items if item.get('turn_index') == turn_index]
    
    if not turn_items:
        return {
            "prompt_level_strict_accuracy": 0.0,
            "prompt_level_loose_accuracy": 0.0,
            "instruction_level_strict_accuracy": 0.0,
            "instruction_level_loose_accuracy": 0.0,
            "overall_accuracy": 0.0
        }
    
    # Collect all follow_all_instructions values
    strict_prompt_results = []
    loose_prompt_results = []
    
    # Collect all follow_instruction_list values
    strict_instruction_results = []
    loose_instruction_results = []
    
    for item in turn_items:
        # Check if result exists and has the evaluation results
        if 'result' in item:
            result = item['result']
            
            # Prompt level - follow_all_instructions
            if 'result_strict' in result:
                strict_prompt_results.append(result['result_strict']['follow_all_instructions'])
            if 'result_loose' in result:
                loose_prompt_results.append(result['result_loose']['follow_all_instructions'])
            
            # Instruction level - follow_instruction_list
            if 'result_strict' in result and 'follow_instruction_list' in result['result_strict']:
                strict_instruction_results.extend(result['result_strict']['follow_instruction_list'])
            if 'result_loose' in result and 'follow_instruction_list' in result['result_loose']:
                loose_instruction_results.extend(result['result_loose']['follow_instruction_list'])
    
    # Calculate accuracies
    prompt_level_strict_accuracy = sum(strict_prompt_results) / len(strict_prompt_results) if strict_prompt_results else 0.0
    prompt_level_loose_accuracy = sum(loose_prompt_results) / len(loose_prompt_results) if loose_prompt_results else 0.0
    instruction_level_strict_accuracy = sum(strict_instruction_results) / len(strict_instruction_results) if strict_instruction_results else 0.0
    instruction_level_loose_accuracy = sum(loose_instruction_results) / len(loose_instruction_results) if loose_instruction_results else 0.0
    
    # Overall accuracy (arithmetic mean of 4 accuracies)
    overall_accuracy = (prompt_level_strict_accuracy + prompt_level_loose_accuracy + 
                       instruction_level_strict_accuracy + instruction_level_loose_accuracy) / 4
    
    return {
        "prompt_level_strict_accuracy": prompt_level_strict_accuracy,
        "prompt_level_loose_accuracy": prompt_level_loose_accuracy,
        "instruction_level_strict_accuracy": instruction_level_strict_accuracy,
        "instruction_level_loose_accuracy": instruction_level_loose_accuracy,
        "overall_accuracy": overall_accuracy
    }

def calculate_overall_metrics(turn_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate overall metrics as arithmetic mean of all turns"""
    if not turn_metrics:
        return {
            "prompt_level_strict_accuracy": 0.0,
            "prompt_level_loose_accuracy": 0.0,
            "instruction_level_strict_accuracy": 0.0,
            "instruction_level_loose_accuracy": 0.0,
            "overall_accuracy": 0.0
        }
    
    num_turns = len(turn_metrics)
    overall_metrics = {}
    
    # Calculate mean for each metric across all turns
    metric_names = ["prompt_level_strict_accuracy", "prompt_level_loose_accuracy",
                   "instruction_level_strict_accuracy", "instruction_level_loose_accuracy", 
                   "overall_accuracy"]
    
    for metric in metric_names:
        total = sum(turn_data[metric] for turn_data in turn_metrics.values())
        overall_metrics[metric] = total / num_turns
    
    return overall_metrics

def analyze_results(input_file: str, output_file: str):
    """Main function to analyze evaluation results"""
    
    # Load input data
    results = load_results(input_file)
    
    # Get unique turn indices (only 1, 2, 3 are valid)
    turn_indices = sorted(set(item.get('turn_index') for item in results if item.get('turn_index') in [1, 2, 3]))
    
    # Calculate metrics for each turn
    final_results = {}
    
    for turn_idx in turn_indices:
        turn_key = f"turn_{turn_idx}"
        final_results[turn_key] = calculate_accuracy_for_turn(results, turn_idx)
    
    # Calculate overall metrics (arithmetic mean of all turns)
    if final_results:
        final_results["overall"] = calculate_overall_metrics(final_results)
    
    # Save results to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis complete. Results saved to: {output_file}")
    return final_results

def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results and calculate metrics')
    parser.add_argument('input_file', help='Input JSON file with evaluation results')
    parser.add_argument('output_file', help='Output JSON file for calculated metrics')
    
    args = parser.parse_args()
    
    analyze_results(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
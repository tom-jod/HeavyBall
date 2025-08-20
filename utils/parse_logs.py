import re
import ast

def parse_log_file(log_file_path):
    """
    Parse log file to extract complete experiment dictionaries only, removing duplicates.
    """
    experiments_data_dict = {}
    
    with open(log_file_path, 'r') as file:
        content = file.read()
    
    # Find all potential dictionary starts and validate them
    dict_candidates = []
    
    # Find all positions where dictionaries might start
    for match in re.finditer(r"\{'tuned_indices'", content):
        start_pos = match.start()
        
        # Extract the complete dictionary by counting braces
        brace_count = 0
        pos = start_pos
        in_string = False
        escape_next = False
        string_char = None
        
        while pos < len(content):
            char = content[pos]
            
            if escape_next:
                escape_next = False
            elif char == '\\':
                escape_next = True
            elif char in ["'", '"'] and not escape_next:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete dictionary
                        dict_str = content[start_pos:pos+1]
                        dict_candidates.append(dict_str)
                        break
            pos += 1
    
    print(f"Found {len(dict_candidates)} potential dictionary candidates")
    
    # Validate and parse each candidate
    valid_experiments = []
    seen_experiments = set()  # To track duplicates
    
    for i, dict_str in enumerate(dict_candidates):
        try:
            # Parse the dictionary
            exp_data = ast.literal_eval(dict_str)
            
            # Validate that this is a complete experiment dictionary
            required_fields = ['tuned_indices', 'params', 'losses', 'test_accuracies', 'runtime']
            if not all(field in exp_data for field in required_fields):
                print(f"Candidate {i+1}: Missing required fields")
                continue
            
            # Check that params has the required parameter fields
            required_params = ['param_0_learning_rate', 'param_1_one_minus_beta1', 
                             'param_2_one_minus_beta2', 'param_5_weight_decay']
            if not all(param in exp_data['params'] for param in required_params):
                print(f"Candidate {i+1}: Missing required parameter fields")
                continue
            
            # Check that we have actual data (not empty lists)
            if (len(exp_data['losses']) == 0 or 
                len(exp_data['test_accuracies']) == 0):
                print(f"Candidate {i+1}: Empty data arrays")
                continue
            
            # Create a unique identifier for this experiment to detect duplicates
            params = exp_data['params']
            experiment_signature = (
                params['param_0_learning_rate'],
                params['param_1_one_minus_beta1'],
                params['param_2_one_minus_beta2'],
                params['param_5_weight_decay'],
                len(exp_data['losses']),
                len(exp_data['test_accuracies']),
                exp_data['runtime']
            )
            
            # Check for duplicates
            if experiment_signature in seen_experiments:
                print(f"Candidate {i+1}: Duplicate experiment (skipping)")
                continue
            
            seen_experiments.add(experiment_signature)
            
            # This looks like a valid, unique experiment
            valid_experiments.append(exp_data)
            print(f"Candidate {i+1}: Valid unique experiment with {len(exp_data['losses'])} loss points, "
                  f"{len(exp_data['test_accuracies'])} accuracy points, "
                  f"LR={params['param_0_learning_rate']:.2e}")
            
        except Exception as e:
            print(f"Candidate {i+1}: Parse error - {e}")
            continue
    
    print(f"\nFound {len(valid_experiments)} valid unique experiments")
    
    # Convert valid experiments to the desired format
    exp_counter = 1
    for exp_data in valid_experiments:
        # Extract parameters
        params = exp_data['params']
        learning_rate = params['param_0_learning_rate']
        one_minus_beta1 = params['param_1_one_minus_beta1']
        one_minus_beta2 = params['param_2_one_minus_beta2']
        weight_decay = params['param_5_weight_decay']
        
        # Calculate actual beta values
        beta1 = 1.0 - one_minus_beta1
        beta2 = 1.0 - one_minus_beta2
        
        # Store in the desired format
        exp_key = f'Exp_{exp_counter}'
        experiments_data_dict[exp_key] = {
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2,
            'weight_decay': weight_decay,
            'test_accuracies': exp_data['test_accuracies']
        }
        
        exp_counter += 1
    
    return experiments_data_dict

def validate_experiments(experiments_data_dict):
    """
    Validate that we have reasonable experiment data
    """
    print("\n=== Experiment Validation ===")
    
    # Check for duplicates in the final results too
    lr_values = []
    for exp_name, exp_data in experiments_data_dict.items():
        lr_values.append(exp_data['learning_rate'])
        print(f"{exp_name}:")
        print(f"  Learning Rate: {exp_data['learning_rate']:.2e}")
        print(f"  Beta1: {exp_data['beta1']:.6f}")
        print(f"  Beta2: {exp_data['beta2']:.6f}")
        print(f"  Weight Decay: {exp_data['weight_decay']:.2e}")
        print(f"  Test Accuracy Points: {len(exp_data['test_accuracies'])}")
        print(f"  Final Test Accuracy: {exp_data['test_accuracies'][-1]:.4f}")
        print(f"  Max Test Accuracy: {max(exp_data['test_accuracies']):.4f}")
        print()
    
    # Check for duplicate learning rates
    unique_lrs = set(lr_values)
    if len(unique_lrs) != len(lr_values):
        print(f"WARNING: Found duplicate learning rates! {len(lr_values)} total, {len(unique_lrs)} unique")
        from collections import Counter
        lr_counts = Counter(lr_values)
        duplicates = {lr: count for lr, count in lr_counts.items() if count > 1}
        print(f"Duplicate learning rates: {duplicates}")
    else:
        print(f"All {len(lr_values)} experiments have unique learning rates ✓")

def debug_log_structure(log_file_path):
    """
    Debug function to understand the log file structure
    """
    with open(log_file_path, 'r') as file:
        content = file.read()
    
    # Count different patterns
    tuned_indices_count = len(re.findall(r"\{'tuned_indices'", content))
    runtime_count = len(re.findall(r"'runtime':\s*[\d.]+\}", content))
    params_count = len(re.findall(r"'params':", content))
    
    print(f"Debug info:")
    print(f"  'tuned_indices' occurrences: {tuned_indices_count}")
    print(f"  'runtime' endings: {runtime_count}")
    print(f"  'params' occurrences: {params_count}")
    
    # Find sections between experiments
    sections = content.split("{'tuned_indices'")
    print(f"  Sections split by tuned_indices: {len(sections)}")
    
    # Look for patterns that might indicate experiment boundaries
    vocab_size_count = len(re.findall(r"Vocabulary size:", content))
    stdout_count = len(re.findall(r"STDOUT:", content))
    
    print(f"  'Vocabulary size:' occurrences: {vocab_size_count}")
    print(f"  'STDOUT:' occurrences: {stdout_count}")

def additional_deduplication(experiments_data_dict):
    """
    Additional deduplication step to ensure we don't have any remaining duplicates
    """
    print("\n=== Additional Deduplication Check ===")
    
    unique_experiments = {}
    duplicates_found = []
    
    for exp_name, exp_data in experiments_data_dict.items():
        # Create signature based on key parameters
        signature = (
            round(exp_data['learning_rate'], 10),  # Round to avoid floating point issues
            round(exp_data['beta1'], 8),
            round(exp_data['beta2'], 8),
            round(exp_data['weight_decay'], 10),
            len(exp_data['test_accuracies'])
        )
        
        if signature in unique_experiments:
            duplicates_found.append((exp_name, unique_experiments[signature]))
            print(f"Duplicate found: {exp_name} matches {unique_experiments[signature]}")
        else:
            unique_experiments[signature] = exp_name
    
    if duplicates_found:
        print(f"Found {len(duplicates_found)} duplicates to remove")
        # Remove duplicates (keep the first occurrence)
        for duplicate_name, _ in duplicates_found:
            del experiments_data_dict[duplicate_name]
        
        # Renumber experiments
        items = list(experiments_data_dict.items())
        experiments_data_dict.clear()
        for i, (_, exp_data) in enumerate(items, 1):
            experiments_data_dict[f'Exp_{i}'] = exp_data
    else:
        print("No additional duplicates found ✓")
    
    return experiments_data_dict

def main():
    log_file_path = "comparison_results/run_AdamW_42_1755502033344.log"
    
    try:
        # First, debug the log structure
        print("=== Debugging Log Structure ===")
        debug_log_structure(log_file_path)
        
        print("\n=== Parsing Experiments ===")
        experiments_data = parse_log_file(log_file_path)
        
        # Additional deduplication
        experiments_data = additional_deduplication(experiments_data)
        
        # Validate the results
        validate_experiments(experiments_data)
        
        if len(experiments_data) > 0:
            # Save to file
            import pprint
            with open("parsed_experiments.py", 'w') as f:
                f.write("experiments_data_dict = ")
                pprint.pprint(experiments_data, stream=f, width=120)
            
            print(f"\nSuccessfully parsed {len(experiments_data)} unique experiments")
            print("Data saved to 'parsed_experiments.py'")
            
            if len(experiments_data) == 20:
                print("✓ Found exactly 20 experiments as expected!")
            else:
                print(f"⚠ Expected 20 experiments but found {len(experiments_data)}")
        else:
            print("No valid experiments found!")
        
        return experiments_data
        
    except FileNotFoundError:
        print(f"Log file '{log_file_path}' not found!")
        return None
    except Exception as e:
        print(f"Error parsing log file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    experiments_data_dict = main()
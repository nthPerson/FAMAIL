"""
Step 2: State Feature Generation Processor

This module generates the 122 additional state features for passenger-seeking
trajectories, transforming the basic [x, y, time, day] state vectors from Step 1
into the full 126-element state vectors used by the cGAIL model.

The state vector structure (126 elements):
- Indices 0-3: [x_grid, y_grid, time_bucket, day_index]
- Indices 4-24: Manhattan distances to 21 POIs from train_airport.pkl
- Indices 25-49: Normalized pickup counts (5x5 window)
- Indices 50-74: Normalized traffic volumes (5x5 window)
- Indices 75-99: Normalized traffic speeds (5x5 window)
- Indices 100-124: Normalized traffic wait times (5x5 window)
- Index 125: Action code (0-9)
"""

import pickle
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import defaultdict

from config import (
    ProcessingConfig,
    Step2Stats,
    ACTION_CODES,
    FEATURE_INDICES,
    NORMALIZATION_CONSTANTS,
    FLOAT_PRECISION,
)


def load_feature_data(config: ProcessingConfig) -> Tuple[Dict, Dict, Dict]:
    """
    Load feature data files required for state feature generation.
    
    Args:
        config: Processing configuration
        
    Returns:
        Tuple of (traffic_data, volume_data, train_airport_data)
    """
    traffic_path = config.source_data_dir / config.traffic_file
    volume_path = config.source_data_dir / config.volume_file
    train_airport_path = config.source_data_dir / config.train_airport_file
    
    with open(traffic_path, 'rb') as f:
        traffic = pickle.load(f)
    
    with open(volume_path, 'rb') as f:
        volume = pickle.load(f)
    
    with open(train_airport_path, 'rb') as f:
        train_airport = pickle.load(f)
    
    return traffic, volume, train_airport


def load_step1_output(filepath: Path) -> Dict[int, List[List[List[int]]]]:
    """
    Load Step 1 output from pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Dict {driver_index: [[trajectory], ...]}
        Each trajectory is a list of states: [x_grid, y_grid, time_bucket, day]
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def judge_action(x: int, y: int, nx: int, ny: int) -> int:
    """
    Determine action code from grid position changes.
    
    IDENTICAL logic to net-dis-cGAIL.ipynb::judging_action()
    
    Action codes:
        0: North (y+, x same)
        1: Northeast (y+, x+)
        2: East (y same, x+)
        3: Southeast (y-, x+)
        4: South (y-, x same)
        5: Southwest (y-, x-)
        6: West (y same, x-)
        7: Northwest (y+, x-)
        8: Stay (no movement)
        9: Stop (terminal)
    
    Args:
        x, y: Current grid position
        nx, ny: Next grid position
        
    Returns:
        Action code [0-9]
    """
    # Handle invalid/terminal positions
    if x == 0 and y == 0:
        return 9
    if nx == 0 and ny == 0:
        return 9
    
    # Determine movement direction
    dx = nx - x  # positive = east, negative = west
    dy = ny - y  # positive = north, negative = south
    
    if dx == 0 and dy == 0:
        return 8  # stay
    elif dx == 0 and dy > 0:
        return 0  # north
    elif dx > 0 and dy > 0:
        return 1  # northeast
    elif dx > 0 and dy == 0:
        return 2  # east
    elif dx > 0 and dy < 0:
        return 3  # southeast
    elif dx == 0 and dy < 0:
        return 4  # south
    elif dx < 0 and dy < 0:
        return 5  # southwest
    elif dx < 0 and dy == 0:
        return 6  # west
    elif dx < 0 and dy > 0:
        return 7  # northwest
    
    return 8  # Default to stay for any edge cases


def compute_poi_distances(
    x: int, 
    y: int, 
    train_airport: Dict[str, Tuple]
) -> List[int]:
    """
    Compute Manhattan distances from current position to all POIs.
    
    IDENTICAL logic to net-dis-cGAIL.ipynb::processing_state_features()
    
    Args:
        x: Current x grid position
        y: Current y grid position
        train_airport: Dict of POI locations {name: [(grid_x, grid_y), (lat, lon)]}
        
    Returns:
        List of 21 Manhattan distances (one per POI)
    """
    distances = []
    for place in train_airport:
        poi_x = train_airport[place][0][0]
        poi_y = train_airport[place][0][1]
        dist = abs(x - poi_x) + abs(y - poi_y)
        distances.append(dist)
    return distances


def compute_window_features(
    x: int,
    y: int,
    t: int,
    day: int,
    traffic: Dict,
    volume: Dict
) -> Tuple[List[float], List[float], List[float], List[float], int, int, int, int]:
    """
    Compute 5x5 window features for pickup counts, volume, speed, and wait times.
    
    IDENTICAL logic to net-dis-cGAIL.ipynb::processing_state_features()
    
    The 5x5 window is centered on (x, y) and iterates:
    - x from x-2 to x+2 (inclusive)
    - y from y-2 to y+2 (inclusive)
    
    Features are normalized using baseline and scale constants.
    Missing values use (0 - baseline) / scale as the default.
    
    Args:
        x, y: Grid position
        t: Time bucket
        day: Day index
        traffic: Traffic data dict {(x, y, t, day): [speed, wait]}
        volume: Volume data dict {(x, y, t, day): [pickup_count, traffic_volume]}
        
    Returns:
        Tuple of:
        - n_p: List of 25 normalized pickup counts
        - n_v: List of 25 normalized traffic volumes
        - t_s: List of 25 normalized traffic speeds
        - t_w: List of 25 normalized traffic wait times
        - traffic_found: Count of traffic keys found
        - traffic_missing: Count of traffic keys missing
        - volume_found: Count of volume keys found
        - volume_missing: Count of volume keys missing
    """
    # Normalization constants from config
    pc = NORMALIZATION_CONSTANTS['pickup_count']
    tv = NORMALIZATION_CONSTANTS['traffic_volume']
    ts = NORMALIZATION_CONSTANTS['traffic_speed']
    tw = NORMALIZATION_CONSTANTS['traffic_wait']
    
    # 5x5 window ranges
    x_range = list(range(x - 2, x + 3))  # [x-2, x-1, x, x+1, x+2]
    y_range = list(range(y - 2, y + 3))  # [y-2, y-1, y, y+1, y+2]
    
    # Feature lists
    n_p = []  # Normalized pickup counts
    n_v = []  # Normalized traffic volumes
    t_s = []  # Normalized traffic speeds
    t_w = []  # Normalized traffic wait times
    
    # Counters for statistics
    traffic_found = 0
    traffic_missing = 0
    volume_found = 0
    volume_missing = 0
    
    # Iterate through 5x5 window
    for i in x_range:
        for j in y_range:
            key = (i, j, t, day)
            
            # Volume features (pickup count and traffic volume)
            # Use float() to convert numpy types to native Python floats for smaller pickle size
            if key in volume:
                pickup = float(round((volume[key][0] - pc['baseline']) / pc['scale'], FLOAT_PRECISION))
                vol = float(round((volume[key][1] - tv['baseline']) / tv['scale'], FLOAT_PRECISION))
                n_p.append(pickup)
                n_v.append(vol)
                volume_found += 1
            else:
                # Default: (0 - baseline) / scale
                n_p.append(float(round(-pc['baseline'] / pc['scale'], FLOAT_PRECISION)))
                n_v.append(float(round(-tv['baseline'] / tv['scale'], FLOAT_PRECISION)))
                volume_missing += 1
            
            # Traffic features (speed and wait time)
            # Use float() to convert numpy types to native Python floats for smaller pickle size
            if key in traffic:
                speed = float(round((traffic[key][0] - ts['baseline']) / ts['scale'], FLOAT_PRECISION))
                wait = float(round((traffic[key][1] - tw['baseline']) / tw['scale'], FLOAT_PRECISION))
                t_s.append(speed)
                t_w.append(wait)
                traffic_found += 1
            else:
                # Default: (0 - baseline) / scale
                t_s.append(float(round(-ts['baseline'] / ts['scale'], FLOAT_PRECISION)))
                t_w.append(float(round(-tw['baseline'] / tw['scale'], FLOAT_PRECISION)))
                traffic_missing += 1
    
    return n_p, n_v, t_s, t_w, traffic_found, traffic_missing, volume_found, volume_missing


def processing_state_features(
    input_state: List[int],
    traffic: Dict,
    volume: Dict,
    train_airport: Dict
) -> Tuple[List[float], int, int, int, int]:
    """
    Generate full state features for a single state.
    
    IDENTICAL logic to net-dis-cGAIL.ipynb::processing_state_features()
    
    Args:
        input_state: [x, y, t, day]
        traffic: Traffic data dict
        volume: Volume data dict
        train_airport: POI data dict
        
    Returns:
        Tuple of:
        - feature_vector: List of 121 features (POIs + 5x5 windows)
          (Does NOT include base state or action - those are added elsewhere)
        - traffic_found: Count of traffic keys found
        - traffic_missing: Count of traffic keys missing
        - volume_found: Count of volume keys found
        - volume_missing: Count of volume keys missing
    """
    x = int(input_state[0])
    y = int(input_state[1])
    t = int(input_state[2])
    day = int(input_state[3])
    
    # Compute POI distances (21 features)
    poi_distances = compute_poi_distances(x, y, train_airport)
    
    # Compute 5x5 window features (100 features total)
    n_p, n_v, t_s, t_w, t_found, t_missing, v_found, v_missing = compute_window_features(
        x, y, t, day, traffic, volume
    )
    
    # Build feature vector (order is CRITICAL - matches cGAIL exactly)
    whole_step = []
    whole_step.extend(poi_distances)  # Indices 4-24 (21 features)
    whole_step.extend(n_p)            # Indices 25-49 (25 features)
    whole_step.extend(n_v)            # Indices 50-74 (25 features)
    whole_step.extend(t_s)            # Indices 75-99 (25 features)
    whole_step.extend(t_w)            # Indices 100-124 (25 features)
    
    return whole_step, t_found, t_missing, v_found, v_missing


def process_trajectories(
    step1_data: Dict[int, List[List[List[int]]]],
    traffic: Dict,
    volume: Dict,
    train_airport: Dict,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Tuple[Dict[int, List[List[float]]], Step2Stats]:
    """
    Process all trajectories from Step 1 output and generate full state features.
    
    MATCHES the output format of the original all_trajs.pkl:
    {driver_index: [[state1, state2, ...], [state1, state2, ...], ...]}
    where each state is a 126-element list.
    
    Args:
        step1_data: Output from Step 1 {driver_index: [[trajectory], ...]}
        traffic: Traffic feature data
        volume: Volume/pickup feature data
        train_airport: POI location data
        progress_callback: Optional callback(stage, progress)
        
    Returns:
        Tuple of:
        - output_data: Dict {driver_index: [[full_state_trajectory], ...]}
        - stats: Step2Stats with processing statistics
    """
    start_time = time.time()
    stats = Step2Stats()
    
    def update_progress(stage: str, progress: float):
        if progress_callback:
            progress_callback(stage, progress)
    
    output_data = {}
    driver_indices = list(step1_data.keys())
    stats.unique_drivers = len(driver_indices)
    
    # Count total trajectories for progress
    total_trajs = sum(len(trajs) for trajs in step1_data.values())
    stats.input_trajectories = total_trajs
    processed_trajs = 0
    
    for driver_idx in driver_indices:
        trajectories = step1_data[driver_idx]
        driver_output = []
        
        for traj in trajectories:
            if len(traj) < 2:
                # Skip single-state trajectories (no action possible)
                processed_trajs += 1
                continue
            
            processed_traj = []
            
            for i in range(len(traj)):
                state = traj[i]
                stats.input_states += 1
                
                # Get current state: [x, y, t, day]
                x, y, t, day = state[0], state[1], state[2], state[3]
                
                # Generate state features (121 additional features)
                features, t_found, t_missing, v_found, v_missing = processing_state_features(
                    state, traffic, volume, train_airport
                )
                
                stats.traffic_keys_found += t_found
                stats.traffic_keys_missing += t_missing
                stats.volume_keys_found += v_found
                stats.volume_keys_missing += v_missing
                
                # Determine action
                if i < len(traj) - 1:
                    # Non-terminal state: compute action from transition
                    next_state = traj[i + 1]
                    nx, ny = next_state[0], next_state[1]
                    action = judge_action(x, y, nx, ny)
                else:
                    # Terminal state: action = 9 (stop)
                    action = 9
                
                # Build full state vector (126 elements)
                # Order: [x, y, t, day, 21 POIs, 25 pickups, 25 volumes, 25 speeds, 25 waits, action]
                full_state = [x, y, t, day]
                full_state.extend(features)
                full_state.append(action)
                
                processed_traj.append(full_state)
                stats.output_states += 1
            
            driver_output.append(processed_traj)
            processed_trajs += 1
            
            if processed_trajs % 100 == 0:
                progress = processed_trajs / total_trajs
                update_progress(f"Processing trajectory {processed_trajs}/{total_trajs}", progress * 0.9)
        
        output_data[driver_idx] = driver_output
    
    stats.processing_time_seconds = time.time() - start_time
    update_progress("Step 2 complete!", 1.0)
    
    return output_data, stats


def process_step2(
    config: ProcessingConfig,
    step1_input_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Tuple[Dict[int, List[List[List[float]]]], Step2Stats]:
    """
    Main Step 2 processing function.
    
    Loads Step 1 output and feature data, then generates full state features.
    
    Args:
        config: Processing configuration
        step1_input_path: Path to Step 1 output (default: config.output_dir / step1_output_filename)
        progress_callback: Optional callback(stage, progress)
        
    Returns:
        Tuple of:
        - all_trajs: Dict {driver_index: [[full_state_trajectory], ...]}
        - stats: Step2Stats with processing statistics
    """
    def update_progress(stage: str, progress: float):
        if progress_callback:
            progress_callback(stage, progress)
    
    # Determine input path
    if step1_input_path is None:
        step1_input_path = config.output_dir / config.step1_output_filename
    
    # Stage 1: Load Step 1 output
    update_progress("Loading Step 1 output...", 0.0)
    step1_data = load_step1_output(step1_input_path)
    
    # Stage 2: Load feature data files
    update_progress("Loading feature data files...", 0.1)
    traffic, volume, train_airport = load_feature_data(config)
    
    # Stage 3: Process trajectories
    update_progress("Processing trajectories...", 0.2)
    all_trajs, stats = process_trajectories(
        step1_data, 
        traffic, 
        volume, 
        train_airport,
        lambda stage, prog: update_progress(stage, 0.2 + prog * 0.8)
    )
    
    return all_trajs, stats


def save_step2_output(
    all_trajs: Dict[int, List[List[List[float]]]],
    output_path: Path
) -> None:
    """
    Save Step 2 output (all_trajs) to pickle file.
    
    Args:
        all_trajs: Dict {driver_index: [[full_state_trajectory], ...]}
        output_path: Path to save the pickle file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_trajs, f)


def validate_output_structure(
    all_trajs: Dict[int, List[List[List[float]]]],
    expected_state_length: int = 126
) -> Dict[str, Any]:
    """
    Validate the output structure matches the expected all_trajs.pkl format.
    
    Args:
        all_trajs: Output data to validate
        expected_state_length: Expected length of state vectors
        
    Returns:
        Dict with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check data type
    if not isinstance(all_trajs, dict):
        results['valid'] = False
        results['errors'].append(f"Expected dict, got {type(all_trajs).__name__}")
        return results
    
    # Check driver indices
    driver_indices = list(all_trajs.keys())
    if not all(isinstance(k, int) for k in driver_indices):
        results['warnings'].append("Not all keys are integers")
    
    results['stats']['num_drivers'] = len(driver_indices)
    
    # Check trajectories and states
    total_trajs = 0
    total_states = 0
    state_lengths = []
    
    for driver_idx, trajectories in all_trajs.items():
        if not isinstance(trajectories, list):
            results['errors'].append(f"Driver {driver_idx}: trajectories should be list")
            continue
        
        total_trajs += len(trajectories)
        
        for traj_idx, traj in enumerate(trajectories):
            if not isinstance(traj, list):
                results['errors'].append(f"Driver {driver_idx} traj {traj_idx}: should be list")
                continue
            
            total_states += len(traj)
            
            for state_idx, state in enumerate(traj):
                state_len = len(state)
                state_lengths.append(state_len)
                
                if state_len != expected_state_length:
                    results['errors'].append(
                        f"Driver {driver_idx} traj {traj_idx} state {state_idx}: "
                        f"length {state_len} != expected {expected_state_length}"
                    )
                    results['valid'] = False
    
    results['stats']['total_trajectories'] = total_trajs
    results['stats']['total_states'] = total_states
    
    if state_lengths:
        results['stats']['min_state_length'] = min(state_lengths)
        results['stats']['max_state_length'] = max(state_lengths)
        results['stats']['avg_state_length'] = np.mean(state_lengths)
    
    if len(results['errors']) > 10:
        results['errors'] = results['errors'][:10] + [f"... and {len(results['errors']) - 10} more"]
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Step 2: Generate state features for passenger-seeking trajectories"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to Step 1 output pickle file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/new_all_trajs.pkl",
        help="Output pickle file path"
    )
    args = parser.parse_args()
    
    config = ProcessingConfig()
    
    def progress_callback(stage, progress):
        print(f"[{progress*100:.1f}%] {stage}")
    
    print("Starting Step 2: State Feature Generation...")
    print("=" * 60)
    
    input_path = Path(args.input) if args.input else None
    all_trajs, stats = process_step2(config, input_path, progress_callback)
    
    print("\nProcessing complete!")
    print(f"  Input trajectories: {stats.input_trajectories:,}")
    print(f"  Input states: {stats.input_states:,}")
    print(f"  Output states: {stats.output_states:,}")
    print(f"  Unique drivers: {stats.unique_drivers}")
    print(f"  Traffic keys found: {stats.traffic_keys_found:,}")
    print(f"  Traffic keys missing: {stats.traffic_keys_missing:,}")
    print(f"  Volume keys found: {stats.volume_keys_found:,}")
    print(f"  Volume keys missing: {stats.volume_keys_missing:,}")
    print(f"  Processing time: {stats.processing_time_seconds:.2f}s")
    
    # Validate output
    print("\nValidating output structure...")
    validation = validate_output_structure(all_trajs)
    if validation['valid']:
        print("  ✓ Output structure is valid")
    else:
        print("  ✗ Output structure has errors:")
        for error in validation['errors'][:5]:
            print(f"    - {error}")
    
    output_path = Path(args.output)
    save_step2_output(all_trajs, output_path)
    print(f"\nSaved to {output_path}")

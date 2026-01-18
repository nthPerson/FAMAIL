"""
New All Trajs - Streamlit Dashboard

A tool for recreating the all_trajs.pkl dataset in two steps:
1. Extract passenger-seeking trajectories from raw GPS data
2. Generate state features using cGAIL feature generation logic

This dashboard provides tabs for:
- Generate Passenger Seeking: Step 1 - Extract trajectories
- Generate State Features: Step 2 - Add 122 additional features
- Analyze: Examine output datasets from either step
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import processors
from config import ProcessingConfig, Step1Stats, Step2Stats, FEATURE_INDICES, NORMALIZATION_CONSTANTS


# Page configuration
st.set_page_config(
    page_title="New All Trajs Generator",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'step1_output' not in st.session_state:
    st.session_state.step1_output = None
if 'step1_stats' not in st.session_state:
    st.session_state.step1_stats = None
if 'step2_output' not in st.session_state:
    st.session_state.step2_output = None
if 'step2_stats' not in st.session_state:
    st.session_state.step2_stats = None
if 'progress_text' not in st.session_state:
    st.session_state.progress_text = ""
if 'progress_value' not in st.session_state:
    st.session_state.progress_value = 0.0


def load_pickle_file(filepath: Path):
    """Load a pickle file safely."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load {filepath}: {e}")
        return None


# Sidebar configuration
st.sidebar.title("🚕 Configuration")

# Data source paths
st.sidebar.subheader("Data Sources")

config = ProcessingConfig()

raw_data_dir = st.sidebar.text_input(
    "Raw Data Directory",
    value=str(config.raw_data_dir),
    help="Directory containing taxi_record_07/08/09_50drivers.pkl files"
)

source_data_dir = st.sidebar.text_input(
    "Source Data Directory", 
    value=str(config.source_data_dir),
    help="Directory containing latest_traffic.pkl, latest_volume_pickups.pkl, train_airport.pkl"
)

output_dir = st.sidebar.text_input(
    "Output Directory",
    value=str(config.output_dir),
    help="Directory for output pickle files"
)

# Update config with user inputs
config.raw_data_dir = Path(raw_data_dir)
config.source_data_dir = Path(source_data_dir)
config.output_dir = Path(output_dir)

st.sidebar.divider()

# Processing parameters
st.sidebar.subheader("Processing Parameters")

config.min_trajectory_length = st.sidebar.number_input(
    "Min Trajectory Length",
    min_value=1,
    max_value=100,
    value=2,
    help="Minimum number of states in a trajectory to include"
)

config.max_trajectory_length = st.sidebar.number_input(
    "Max Trajectory Length",
    min_value=10,
    max_value=10000,
    value=1000,
    help="Maximum number of states in a trajectory (longer trajectories are excluded)"
)

config.max_trajectories_per_driver = st.sidebar.number_input(
    "Max Trajectories per Driver",
    min_value=100,
    max_value=50000,
    value=5000,
    help="Maximum number of trajectories to retain per driver"
)

# Note: Raw data only contains Monday-Friday, so weekends are excluded by default
config.exclude_weekends = True

# Main content tabs
tab1, tab2, tab3 = st.tabs([
    "🔄 Generate Passenger Seeking",
    "🧮 Generate State Features", 
    "📊 Analyze"
])


# Tab 1: Step 1 - Generate Passenger Seeking Trajectories
with tab1:
    st.header("Step 1: Extract Passenger-Seeking Trajectories")
    
    st.markdown("""
    This step extracts passenger-seeking trajectories from raw taxi GPS data.
    
    **Definition:** A passenger-seeking trajectory begins when the passenger indicator 
    changes from 1 to 0 (dropoff) and ends when it changes from 0 to 1 (pickup).
    
    **Output format:** `{driver_index: [[trajectory], ...]}` where each trajectory 
    is a list of states: `[x_grid, y_grid, time_bucket, day]`
    """)
    
    # Input file selection
    st.subheader("Input Files")
    
    input_files_exist = []
    for filename in config.input_files:
        filepath = config.raw_data_dir / filename
        exists = filepath.exists()
        input_files_exist.append(exists)
        status = "✅" if exists else "❌"
        st.text(f"{status} {filename}")
    
    if not any(input_files_exist):
        st.error("No input files found. Please check the raw data directory.")
    
    # Output file path
    st.subheader("Output")
    step1_output_path = st.text_input(
        "Output File",
        value=str(config.output_dir / config.step1_output_filename),
        key="step1_output_path"
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Passenger-Seeking Trajectories", type="primary", use_container_width=True):
            if not any(input_files_exist):
                st.error("No valid input files found!")
            else:
                from step1_processor import process_step1, save_step1_output
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(stage, progress):
                    progress_bar.progress(progress)
                    status_text.text(stage)
                
                try:
                    with st.spinner("Processing..."):
                        trajectories, stats, index_to_plate = process_step1(config, progress_callback)
                    
                    st.session_state.step1_output = trajectories
                    st.session_state.step1_stats = stats
                    
                    # Save output
                    output_path = Path(step1_output_path)
                    save_step1_output(trajectories, output_path, index_to_plate)
                    
                    st.success(f"✅ Successfully generated {stats.total_trajectories:,} trajectories!")
                    st.info(f"Saved to: {output_path}")
                    
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display statistics if available
    if st.session_state.step1_stats:
        st.subheader("Processing Statistics")
        stats = st.session_state.step1_stats
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Raw Records", f"{stats.total_raw_records:,}")
            st.metric("After Sunday Filter", f"{stats.records_after_sunday_filter:,}")
        with col2:
            st.metric("Unique Drivers", stats.unique_drivers)
            st.metric("Total Trajectories", f"{stats.total_trajectories:,}")
        with col3:
            st.metric("Total States", f"{stats.total_states:,}")
            st.metric("Avg Trajectory Length", f"{stats.avg_trajectory_length:.2f}")
        with col4:
            st.metric("Min Trajectory Length", stats.min_trajectory_length)
            st.metric("Max Trajectory Length", stats.max_trajectory_length)
        
        st.metric("Processing Time", f"{stats.processing_time_seconds:.2f}s")


# Tab 2: Step 2 - Generate State Features
with tab2:
    st.header("Step 2: Generate State Features")
    
    st.markdown("""
    This step adds 122 additional state features to the basic trajectories from Step 1.
    
    **Features added:**
    - 21 Manhattan distances to POIs (train stations, airports)
    - 25 normalized pickup counts (5×5 window)
    - 25 normalized traffic volumes (5×5 window)
    - 25 normalized traffic speeds (5×5 window)
    - 25 normalized traffic wait times (5×5 window)
    - Action code (0-9)
    
    **Output format:** `{driver_index: [[full_state_trajectory], ...]}` where each 
    state is a 126-element vector.
    """)
    
    # Input file selection
    st.subheader("Input")
    
    step1_input_source = st.radio(
        "Step 1 Input Source",
        ["From File", "From Memory (last Step 1 run)"],
        help="Choose whether to load from file or use the result from the last Step 1 run"
    )
    
    if step1_input_source == "From File":
        # Find all .pkl files in output directory
        output_dir = config.output_dir
        pkl_files = []
        if output_dir.exists():
            pkl_files = sorted([f.name for f in output_dir.glob("*.pkl")])
        
        if pkl_files:
            step1_input_path = st.selectbox(
                "Step 1 Output File",
                options=pkl_files,
                index=0 if config.step1_output_filename not in pkl_files else pkl_files.index(config.step1_output_filename),
                key="step2_input_path"
            )
            step1_file_path = output_dir / step1_input_path
            st.success(f"✅ Selected: {step1_file_path}")
            step1_input_path = str(step1_file_path)
        else:
            st.warning(f"⚠️ No .pkl files found in {output_dir}")
            step1_input_path = None
    else:
        if st.session_state.step1_output is not None:
            st.success(f"✅ Step 1 output available in memory ({len(st.session_state.step1_output)} drivers)")
            step1_input_path = None
        else:
            st.warning("⚠️ No Step 1 output in memory. Run Step 1 first or load from file.")
            step1_input_path = None
    
    # Feature data files check
    st.subheader("Feature Data Files")
    
    feature_files = [
        (config.traffic_file, "Traffic data (speed, wait)"),
        (config.volume_file, "Volume data (pickups, traffic)"),
        (config.train_airport_file, "POI locations (21 places)"),
    ]
    
    all_features_exist = True
    for filename, desc in feature_files:
        filepath = config.source_data_dir / filename
        exists = filepath.exists()
        all_features_exist = all_features_exist and exists
        status = "✅" if exists else "❌"
        st.text(f"{status} {filename} - {desc}")
    
    # Output file path
    st.subheader("Output")
    step2_output_path = st.text_input(
        "Output File",
        value=str(config.output_dir / config.step2_output_filename),
        key="step2_output_path"
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        can_run = all_features_exist and (
            (step1_input_source == "From File" and Path(step1_input_path).exists()) or
            (step1_input_source == "From Memory (last Step 1 run)" and st.session_state.step1_output is not None)
        )
        
        if st.button("🧮 Generate State Features", type="primary", use_container_width=True, disabled=not can_run):
            from step2_processor import process_step2, save_step2_output, load_step1_output, load_feature_data, process_trajectories
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(stage, progress):
                progress_bar.progress(progress)
                status_text.text(stage)
            
            try:
                with st.spinner("Processing..."):
                    if step1_input_source == "From Memory (last Step 1 run)":
                        # Use in-memory data
                        status_text.text("Loading feature data files...")
                        traffic, volume, train_airport = load_feature_data(config)
                        
                        status_text.text("Processing trajectories...")
                        all_trajs, stats = process_trajectories(
                            st.session_state.step1_output,
                            traffic,
                            volume,
                            train_airport,
                            lambda stage, prog: progress_callback(stage, 0.2 + prog * 0.8)
                        )
                    else:
                        # Load from file
                        all_trajs, stats = process_step2(
                            config, 
                            Path(step1_input_path), 
                            progress_callback
                        )
                
                st.session_state.step2_output = all_trajs
                st.session_state.step2_stats = stats
                
                # Save output
                output_path = Path(step2_output_path)
                save_step2_output(all_trajs, output_path)
                
                st.success(f"✅ Successfully generated {stats.output_states:,} states with full features!")
                st.info(f"Saved to: {output_path}")
                
            except Exception as e:
                st.error(f"Processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display statistics if available
    if st.session_state.step2_stats:
        st.subheader("Processing Statistics")
        stats = st.session_state.step2_stats
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Input Trajectories", f"{stats.input_trajectories:,}")
            st.metric("Input States", f"{stats.input_states:,}")
        with col2:
            st.metric("Output States", f"{stats.output_states:,}")
            st.metric("Unique Drivers", stats.unique_drivers)
        with col3:
            st.metric("Traffic Keys Found", f"{stats.traffic_keys_found:,}")
            st.metric("Traffic Keys Missing", f"{stats.traffic_keys_missing:,}")
        with col4:
            st.metric("Volume Keys Found", f"{stats.volume_keys_found:,}")
            st.metric("Volume Keys Missing", f"{stats.volume_keys_missing:,}")
        
        st.metric("Processing Time", f"{stats.processing_time_seconds:.2f}s")


# Tab 3: Analyze
with tab3:
    st.header("📊 Dataset Analysis")
    
    st.markdown("""
    Analyze output datasets from Step 1 (passenger-seeking trajectories) or 
    Step 2 (full state features).
    """)
    
    # Dataset selection
    st.subheader("Select Dataset")
    
    analysis_source = st.radio(
        "Dataset Source",
        ["Load from file", "From memory (last run)"],
        horizontal=True
    )
    
    analysis_data = None
    data_type = None
    
    if analysis_source == "Load from file":
        # Find all .pkl files in output directory
        output_dir = config.output_dir
        pkl_files = []
        if output_dir.exists():
            pkl_files = sorted([f.name for f in output_dir.glob("*.pkl")])
        
        if pkl_files:
            analysis_file = st.selectbox(
                "Select Dataset File",
                options=pkl_files,
                index=0,
                key="analysis_file"
            )
            analysis_path = output_dir / analysis_file
        else:
            st.warning(f"No .pkl files found in {output_dir}")
            analysis_path = None
        
        if pkl_files and st.button("Load Dataset"):
            if analysis_path and analysis_path.exists():
                analysis_data = load_pickle_file(analysis_path)
                if analysis_data:
                    st.session_state.analysis_data = analysis_data
                    # Detect data type
                    sample_driver = list(analysis_data.keys())[0]
                    sample_traj = analysis_data[sample_driver][0] if analysis_data[sample_driver] else None
                    if sample_traj:
                        sample_state = sample_traj[0]
                        if len(sample_state) == 4:
                            data_type = "step1"
                            st.info("Detected Step 1 output (4-element states)")
                        elif len(sample_state) == 126:
                            data_type = "step2"
                            st.info("Detected Step 2 output (126-element states)")
                        else:
                            data_type = "unknown"
                            st.warning(f"Unknown state vector length: {len(sample_state)}")
                    st.session_state.data_type = data_type
            else:
                st.error(f"File not found: {analysis_path}")
    else:
        memory_options = []
        if st.session_state.step1_output is not None:
            memory_options.append("Step 1 Output")
        if st.session_state.step2_output is not None:
            memory_options.append("Step 2 Output")
        
        if not memory_options:
            st.warning("No data in memory. Run Step 1 or Step 2 first, or load from file.")
        else:
            selected_memory = st.selectbox("Select memory data", memory_options)
            if selected_memory == "Step 1 Output":
                analysis_data = st.session_state.step1_output
                data_type = "step1"
            else:
                analysis_data = st.session_state.step2_output
                data_type = "step2"
            st.session_state.analysis_data = analysis_data
            st.session_state.data_type = data_type
    
    # Load from session state if previously loaded
    if 'analysis_data' in st.session_state and st.session_state.analysis_data is not None:
        analysis_data = st.session_state.analysis_data
        data_type = st.session_state.get('data_type', 'unknown')
    
    if analysis_data:
        st.divider()
        
        # Overview statistics
        st.subheader("Overview Statistics")
        
        num_drivers = len(analysis_data)
        all_traj_lengths = []
        all_states = []
        
        for driver_idx, trajectories in analysis_data.items():
            for traj in trajectories:
                all_traj_lengths.append(len(traj))
                all_states.extend(traj)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Drivers", num_drivers)
        with col2:
            st.metric("Trajectories", f"{len(all_traj_lengths):,}")
        with col3:
            st.metric("Total States", f"{len(all_states):,}")
        with col4:
            if all_traj_lengths:
                st.metric("Avg Trajectory Length", f"{np.mean(all_traj_lengths):.2f}")
        
        # Trajectory length distribution
        st.subheader("Trajectory Length Distribution")
        
        if all_traj_lengths:
            fig = px.histogram(
                x=all_traj_lengths,
                nbins=50,
                title="Distribution of Trajectory Lengths",
                labels={'x': 'Trajectory Length', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Min Length", min(all_traj_lengths))
            with col2:
                st.metric("Max Length", max(all_traj_lengths))
            with col3:
                st.metric("Median Length", f"{np.median(all_traj_lengths):.0f}")
            with col4:
                st.metric("Mean Length", f"{np.mean(all_traj_lengths):.2f}")
            with col5:
                st.metric("Std Dev", f"{np.std(all_traj_lengths):.2f}")
        
        # Per-driver statistics
        st.subheader("Per-Driver Statistics")
        
        driver_stats = []
        for driver_idx, trajectories in analysis_data.items():
            traj_lengths = [len(t) for t in trajectories]
            driver_stats.append({
                'Driver Index': driver_idx,
                'Num Trajectories': len(trajectories),
                'Total States': sum(traj_lengths),
                'Avg Traj Length': np.mean(traj_lengths) if traj_lengths else 0,
                'Min Traj Length': min(traj_lengths) if traj_lengths else 0,
                'Max Traj Length': max(traj_lengths) if traj_lengths else 0,
            })
        
        driver_df = pd.DataFrame(driver_stats)
        
        # Trajectories per driver
        fig = px.bar(
            driver_df,
            x='Driver Index',
            y='Num Trajectories',
            title="Trajectories per Driver"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(driver_df, use_container_width=True)
        
        # Step 2 specific analysis
        if data_type == "step2" and len(all_states) > 0 and len(all_states[0]) == 126:
            st.divider()
            st.subheader("State Feature Analysis (Step 2)")
            
            # Convert to numpy for analysis
            states_array = np.array(all_states)
            
            # Feature groups
            feature_groups = {
                'Grid Position': [0, 1],
                'Time & Day': [2, 3],
                'POI Distances': list(range(4, 25)),
                'Pickup Counts': list(range(25, 50)),
                'Traffic Volumes': list(range(50, 75)),
                'Traffic Speeds': list(range(75, 100)),
                'Traffic Wait Times': list(range(100, 125)),
                'Action': [125],
            }
            
            # Feature statistics
            st.markdown("**Feature Statistics by Group**")
            
            for group_name, indices in feature_groups.items():
                with st.expander(f"{group_name} (indices {indices[0]}-{indices[-1]})"):
                    group_data = states_array[:, indices]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Min", f"{np.min(group_data):.4f}")
                    with col2:
                        st.metric("Max", f"{np.max(group_data):.4f}")
                    with col3:
                        st.metric("Mean", f"{np.mean(group_data):.4f}")
                    with col4:
                        st.metric("Std", f"{np.std(group_data):.4f}")
                    
                    # Distribution histogram for normalized features
                    if group_name not in ['Grid Position', 'Time & Day', 'Action']:
                        fig = px.histogram(
                            x=group_data.flatten(),
                            nbins=50,
                            title=f"{group_name} Value Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Action distribution
            st.markdown("**Action Distribution**")
            actions = states_array[:, 125].astype(int)
            action_counts = pd.Series(actions).value_counts().sort_index()
            
            action_labels = {
                0: 'North', 1: 'NE', 2: 'East', 3: 'SE',
                4: 'South', 5: 'SW', 6: 'West', 7: 'NW',
                8: 'Stay', 9: 'Stop'
            }
            
            action_df = pd.DataFrame({
                'Action': [action_labels.get(i, str(i)) for i in action_counts.index],
                'Count': action_counts.values,
                'Percentage': (action_counts.values / len(actions) * 100)
            })
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(action_df, x='Action', y='Count', title="Action Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(action_df, values='Count', names='Action', title="Action Proportion")
                st.plotly_chart(fig, use_container_width=True)
            
            # Spatial distribution
            st.markdown("**Spatial Distribution**")
            x_coords = states_array[:, 0]
            y_coords = states_array[:, 1]
            
            # 2D histogram
            fig = px.density_heatmap(
                x=x_coords,
                y=y_coords,
                nbinsx=50,
                nbinsy=50,
                title="Spatial State Distribution",
                labels={'x': 'X Grid', 'y': 'Y Grid'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Time distribution
            st.markdown("**Temporal Distribution**")
            time_buckets = states_array[:, 2].astype(int)
            
            fig = px.histogram(
                x=time_buckets,
                nbins=288,
                title="States by Time of Day (5-minute buckets)",
                labels={'x': 'Time Bucket (0-287)', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data export
        st.divider()
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Statistics to CSV"):
                driver_df.to_csv(config.output_dir / "driver_statistics.csv", index=False)
                st.success(f"Saved to {config.output_dir / 'driver_statistics.csv'}")


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    New All Trajs Generator | Part of FAMAIL Project
</div>
""", unsafe_allow_html=True)

"""
Streamlit Dashboard for All Trajs Dataset Generation and Analysis

This dashboard provides:
- Configuration options for data source paths
- Dataset generation interface
- Dataset statistics and analysis
- Visualizations
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ProcessingConfig, 
    NORMALIZATION_CONSTANTS, 
    ACTION_CODES, 
    STATE_VECTOR_FIELDS
)
from processor import process_data, process_data_low_memory, save_output, load_output


# Page configuration
st.set_page_config(
    page_title="All Trajs Dataset Generator",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_default_paths():
    """Get default paths relative to the module location."""
    base_dir = Path(__file__).resolve().parent.parent
    return {
        "raw_data_dir": str(base_dir / "raw_data"),
        "source_data_dir": str(base_dir / "source_data"),
        "output_dir": str(base_dir / "new_all_trajs" / "output"),
    }


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("🚕 Configuration")
    
    defaults = get_default_paths()
    
    st.sidebar.subheader("📁 Data Source Paths")
    
    raw_data_dir = st.sidebar.text_input(
        "Raw Data Directory",
        value=defaults["raw_data_dir"],
        help="Path to directory containing taxi_record_*.pkl files"
    )
    
    source_data_dir = st.sidebar.text_input(
        "Source Data Directory",
        value=defaults["source_data_dir"],
        help="Path to directory containing feature data files"
    )
    
    output_dir = st.sidebar.text_input(
        "Output Directory",
        value=defaults["output_dir"],
        help="Path to directory for output files"
    )
    
    st.sidebar.subheader("⚙️ Processing Options")
    
    memory_mode = st.sidebar.selectbox(
        "Memory Mode",
        options=["Standard", "Low Memory"],
        index=0,
        help="Standard: Faster but uses more memory (~10GB). Low Memory: Slower but uses ~70% less memory (~3GB). Use Low Memory if you experience crashes."
    )
    
    exclude_sunday = st.sidebar.checkbox(
        "Exclude Sunday",
        value=True,
        help="Exclude Sunday data from processing"
    )
    
    grid_size = st.sidebar.number_input(
        "Grid Size (degrees)",
        value=0.01,
        min_value=0.001,
        max_value=0.1,
        step=0.001,
        format="%.3f",
        help="Size of each grid cell in degrees"
    )
    
    time_interval = st.sidebar.number_input(
        "Time Interval (minutes)",
        value=5,
        min_value=1,
        max_value=60,
        step=1,
        help="Duration of each time bucket in minutes"
    )
    
    st.sidebar.subheader("📊 Input Files")
    
    input_files = st.sidebar.multiselect(
        "Raw Data Files",
        options=[
            "taxi_record_07_50drivers.pkl",
            "taxi_record_08_50drivers.pkl",
            "taxi_record_09_50drivers.pkl",
        ],
        default=[
            "taxi_record_07_50drivers.pkl",
            "taxi_record_08_50drivers.pkl",
            "taxi_record_09_50drivers.pkl",
        ],
        help="Select which months to include"
    )
    
    return {
        "raw_data_dir": raw_data_dir,
        "source_data_dir": source_data_dir,
        "output_dir": output_dir,
        "memory_mode": memory_mode,
        "exclude_sunday": exclude_sunday,
        "grid_size": grid_size,
        "time_interval": time_interval,
        "input_files": tuple(input_files),
    }


def render_generate_tab(config_params: dict):
    """Render the Generate tab content."""
    st.header("📦 Dataset Structure Overview")
    
    # Dataset structure explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Hierarchy")
        st.markdown("""
        ```
        new_all_trajs.pkl
        ├── driver_key_0 (int)
        │   ├── trajectory_0 (list of states)
        │   │   ├── state_0 (126 elements)
        │   │   ├── state_1 (126 elements)
        │   │   └── ...
        │   ├── trajectory_1
        │   └── ...
        ├── driver_key_1
        │   └── ...
        └── ... (50 drivers total)
        ```
        """)
    
    with col2:
        st.subheader("Key Properties")
        st.markdown("""
        - **Number of Drivers:** 50
        - **State Vector Length:** 126 elements
        - **Time Buckets:** 288 (5-minute intervals)
        - **Spatial Grid:** ~50×90 cells
        - **Action Codes:** 10 (0-9)
        """)
    
    st.divider()
    
    # State vector schema
    st.header("🔢 State Vector Schema")
    
    schema_data = [
        {"Index Range": "0", "Field Name": "x_grid", "Data Type": "int", "Description": "Grid index for longitude position"},
        {"Index Range": "1", "Field Name": "y_grid", "Data Type": "int", "Description": "Grid index for latitude position"},
        {"Index Range": "2", "Field Name": "time_bucket", "Data Type": "int", "Description": "Discretized time-of-day slot ∈ [1, 288]"},
        {"Index Range": "3", "Field Name": "day_index", "Data Type": "int", "Description": "Day-of-week indicator [1-6]"},
        {"Index Range": "4–24", "Field Name": "poi_manhattan_distance", "Data Type": "float", "Description": "Manhattan distances to 21 POIs"},
        {"Index Range": "25–49", "Field Name": "pickup_count_norm", "Data Type": "float", "Description": "Normalized pickup counts (5×5 window)"},
        {"Index Range": "50–74", "Field Name": "traffic_volume_norm", "Data Type": "float", "Description": "Normalized traffic volumes (5×5 window)"},
        {"Index Range": "75–99", "Field Name": "traffic_speed_norm", "Data Type": "float", "Description": "Normalized traffic speeds (5×5 window)"},
        {"Index Range": "100–124", "Field Name": "traffic_wait_norm", "Data Type": "float", "Description": "Normalized traffic waiting times (5×5 window)"},
        {"Index Range": "125", "Field Name": "action_code", "Data Type": "int", "Description": "Movement action label (0-9)"},
    ]
    
    st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Feature generation logic
    st.header("⚙️ Feature Generation Logic")
    
    with st.expander("📍 POI Manhattan Distances (Indices 4-24)", expanded=False):
        st.markdown("""
        **Source:** `train_airport.pkl`
        
        **Logic:**
        ```python
        for place in train_airport:
            distance = abs(x - train_airport[place][0][0]) + abs(y - train_airport[place][0][1])
        ```
        
        **Description:** Computes Manhattan distance from current (x, y) position to each of 21 POIs including:
        - 深圳北站 (Shenzhen North Railway Station)
        - 深圳东站 (Shenzhen East Railway Station)
        - 深圳站 (Shenzhen Railway Station)
        - 福田站 (Futian Station)
        - 宝安机场 (Bao'an Airport)
        - And 16 additional POIs
        """)
    
    with st.expander("📊 Pickup Counts (Indices 25-49)", expanded=False):
        st.markdown(f"""
        **Source:** `latest_volume_pickups.pkl[key][0]`
        
        **Logic:**
        ```python
        for i in range(x-2, x+3):
            for j in range(y-2, y+3):
                if (i, j, t, day) in volume:
                    n_p = (volume[(i,j,t,day)][0] - {NORMALIZATION_CONSTANTS['pickup_mean']:.4f}) / {NORMALIZATION_CONSTANTS['pickup_std']:.4f}
                else:
                    n_p = -{NORMALIZATION_CONSTANTS['pickup_mean']:.4f} / {NORMALIZATION_CONSTANTS['pickup_std']:.4f}
        ```
        
        **Description:** Normalized pickup counts over a 5×5 spatial window centered on current position.
        """)
    
    with st.expander("🚗 Traffic Volumes (Indices 50-74)", expanded=False):
        st.markdown(f"""
        **Source:** `latest_volume_pickups.pkl[key][1]`
        
        **Logic:**
        ```python
        for i in range(x-2, x+3):
            for j in range(y-2, y+3):
                if (i, j, t, day) in volume:
                    n_v = (volume[(i,j,t,day)][1] - {NORMALIZATION_CONSTANTS['volume_mean']:.4f}) / {NORMALIZATION_CONSTANTS['volume_std']:.4f}
                else:
                    n_v = -{NORMALIZATION_CONSTANTS['volume_mean']:.4f} / {NORMALIZATION_CONSTANTS['volume_std']:.4f}
        ```
        
        **Description:** Normalized traffic volumes over a 5×5 spatial window centered on current position.
        """)
    
    with st.expander("🏎️ Traffic Speeds (Indices 75-99)", expanded=False):
        st.markdown(f"""
        **Source:** `latest_traffic.pkl[key][0]`
        
        **Logic:**
        ```python
        for i in range(x-2, x+3):
            for j in range(y-2, y+3):
                if (i, j, t, day) in traffic:
                    t_s = (traffic[(i,j,t,day)][0] - {NORMALIZATION_CONSTANTS['speed_mean']:.6f}) / {NORMALIZATION_CONSTANTS['speed_std']:.6f}
                else:
                    t_s = -{NORMALIZATION_CONSTANTS['speed_mean']:.6f} / {NORMALIZATION_CONSTANTS['speed_std']:.6f}
        ```
        
        **Description:** Normalized traffic speeds over a 5×5 spatial window centered on current position.
        """)
    
    with st.expander("⏱️ Traffic Waiting Times (Indices 100-124)", expanded=False):
        st.markdown(f"""
        **Source:** `latest_traffic.pkl[key][1]`
        
        **Logic:**
        ```python
        for i in range(x-2, x+3):
            for j in range(y-2, y+3):
                if (i, j, t, day) in traffic:
                    t_w = (traffic[(i,j,t,day)][1] - {NORMALIZATION_CONSTANTS['wait_mean']:.4f}) / {NORMALIZATION_CONSTANTS['wait_std']:.4f}
                else:
                    t_w = -{NORMALIZATION_CONSTANTS['wait_mean']:.4f} / {NORMALIZATION_CONSTANTS['wait_std']:.4f}
        ```
        
        **Description:** Normalized traffic waiting times over a 5×5 spatial window centered on current position.
        """)
    
    with st.expander("🎯 Action Codes (Index 125)", expanded=False):
        st.markdown("""
        **Source:** Computed from consecutive positions
        
        **Logic:**
        ```python
        def judging_action(x, y, nx, ny):
            if x == 0 and y == 0: return 9
            if nx == 0 and ny == 0: return 9  # stop
            if x == nx and ny > y: return 0   # north
            if x < nx and ny > y: return 1    # northeast
            if x < nx and ny == y: return 2   # east
            if x < nx and ny < y: return 3    # southeast
            if x == nx and ny < y: return 4   # south
            if x > nx and ny < y: return 5    # southwest
            if x > nx and ny == y: return 6   # west
            if x > nx and ny > y: return 7    # northwest
            if x == nx and y == ny: return 8  # stay
        ```
        """)
        
        action_df = pd.DataFrame([
            {"Code": k, "Movement": v} for k, v in ACTION_CODES.items()
        ])
        st.dataframe(action_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Generation interface
    st.header("🚀 Generate Dataset")
    
    output_filename = st.text_input(
        "Output File Name",
        value="new_all_trajs.pkl",
        help="Name of the output pickle file"
    )
    
    # Validate paths
    raw_data_path = Path(config_params["raw_data_dir"])
    source_data_path = Path(config_params["source_data_dir"])
    
    validation_errors = []
    if not raw_data_path.exists():
        validation_errors.append(f"❌ Raw data directory not found: {raw_data_path}")
    if not source_data_path.exists():
        validation_errors.append(f"❌ Source data directory not found: {source_data_path}")
    
    # Check for required files
    required_source_files = ["latest_traffic.pkl", "latest_volume_pickups.pkl", "train_airport.pkl"]
    for f in required_source_files:
        if source_data_path.exists() and not (source_data_path / f).exists():
            validation_errors.append(f"❌ Required file not found: {f}")
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        st.success("✅ All required files found")
    
    # Show memory mode info
    if config_params["memory_mode"] == "Low Memory":
        st.info("🐢 **Low Memory Mode**: Processing will be slower but use ~70% less RAM. Recommended for WSL2 or memory-constrained environments.")
    else:
        st.info("🚀 **Standard Mode**: Faster processing but uses more memory (~10GB). Switch to Low Memory mode if you experience crashes.")
    
    if st.button("🔄 Generate Dataset", disabled=len(validation_errors) > 0, type="primary"):
        # Create config
        config = ProcessingConfig(
            raw_data_dir=Path(config_params["raw_data_dir"]),
            source_data_dir=Path(config_params["source_data_dir"]),
            output_dir=Path(config_params["output_dir"]),
            exclude_sunday=config_params["exclude_sunday"],
            grid_size=config_params["grid_size"],
            time_interval=config_params["time_interval"],
            input_files=config_params["input_files"],
            output_filename=output_filename,
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(stage: str, progress: float):
            progress_bar.progress(progress)
            status_text.text(stage)
        
        try:
            with st.spinner("Processing..."):
                # Choose processing function based on memory mode
                if config_params["memory_mode"] == "Low Memory":
                    all_trajs, stats = process_data_low_memory(config, progress_callback)
                else:
                    all_trajs, stats = process_data(config, progress_callback)
                
                # Save output
                output_path = config.output_dir / output_filename
                save_output(all_trajs, output_path)
                
                st.success(f"✅ Dataset generated successfully!")
                
                # Display stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Drivers", stats.total_drivers)
                col2.metric("Total Trajectories", f"{stats.total_trajectories:,}")
                col3.metric("Total States", f"{stats.total_states:,}")
                col4.metric("Processing Time", f"{stats.processing_time_seconds:.2f}s")
                
                st.info(f"💾 Saved to: {output_path}")
                
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def compute_dataset_stats(data: Dict) -> Dict:
    """Compute comprehensive statistics for the dataset."""
    stats = {
        "num_drivers": len(data),
        "trajectories_per_driver": [],
        "states_per_trajectory": [],
        "all_states": [],
    }
    
    for driver_id, trajectories in data.items():
        stats["trajectories_per_driver"].append(len(trajectories))
        for traj in trajectories:
            stats["states_per_trajectory"].append(len(traj))
            for state in traj:
                stats["all_states"].append(state)
    
    return stats


def render_analyze_tab(config_params: dict):
    """Render the Analyze tab content."""
    st.header("📊 Dataset Analysis")
    
    # File selector
    output_dir = Path(config_params["output_dir"])
    source_dir = Path(config_params["source_data_dir"])
    
    # List available files
    available_files = []
    
    if output_dir.exists():
        available_files.extend([str(f) for f in output_dir.glob("*.pkl")])
    
    if source_dir.exists():
        all_trajs_path = source_dir / "all_trajs.pkl"
        if all_trajs_path.exists():
            available_files.append(str(all_trajs_path))
    
    if not available_files:
        st.warning("No dataset files found. Generate a dataset first or check your paths.")
        return
    
    selected_file = st.selectbox(
        "Select Dataset File",
        options=available_files,
        help="Choose a dataset file to analyze"
    )
    
    if st.button("📥 Load Dataset", type="primary"):
        try:
            with st.spinner("Loading dataset..."):
                data = load_output(Path(selected_file))
                st.session_state["loaded_data"] = data
                st.session_state["loaded_file"] = selected_file
                st.success(f"✅ Loaded {len(data)} drivers from {Path(selected_file).name}")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
    
    if "loaded_data" not in st.session_state:
        st.info("👆 Load a dataset to view analysis")
        return
    
    data = st.session_state["loaded_data"]
    
    st.divider()
    
    # Compute stats
    with st.spinner("Computing statistics..."):
        stats = compute_dataset_stats(data)
    
    # Overview metrics
    st.subheader("📈 Overview")
    
    total_trajectories = sum(stats["trajectories_per_driver"])
    total_states = len(stats["all_states"])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Drivers", stats["num_drivers"])
    col2.metric("Total Trajectories", f"{total_trajectories:,}")
    col3.metric("Total States", f"{total_states:,}")
    col4.metric("Avg States/Traj", f"{np.mean(stats['states_per_trajectory']):.1f}")
    
    st.divider()
    
    # State Vector Statistics
    st.subheader("🔢 State Vector Statistics")
    
    if len(stats["all_states"]) > 0:
        # Sample states for efficiency
        sample_size = min(100000, len(stats["all_states"]))
        sample_indices = np.random.choice(len(stats["all_states"]), sample_size, replace=False)
        sample_states = [stats["all_states"][i] for i in sample_indices]
        
        state_array = np.array(sample_states)
        
        # Build statistics table
        stat_rows = []
        feature_groups = [
            (0, 0, "x_grid", "Longitude grid index"),
            (1, 1, "y_grid", "Latitude grid index"),
            (2, 2, "time_bucket", "Time bucket [1-288]"),
            (3, 3, "day_index", "Day of week [1-6]"),
            (4, 24, "poi_distances", "POI Manhattan distances"),
            (25, 49, "pickup_counts", "Normalized pickup counts"),
            (50, 74, "traffic_volumes", "Normalized traffic volumes"),
            (75, 99, "traffic_speeds", "Normalized traffic speeds"),
            (100, 124, "traffic_waits", "Normalized traffic waits"),
            (125, 125, "action_code", "Movement action [0-9]"),
        ]
        
        for start_idx, end_idx, name, desc in feature_groups:
            if end_idx < state_array.shape[1]:
                values = state_array[:, start_idx:end_idx+1].flatten()
                stat_rows.append({
                    "Index Range": f"{start_idx}" if start_idx == end_idx else f"{start_idx}-{end_idx}",
                    "Feature Group": name,
                    "Description": desc,
                    "Min": f"{np.min(values):.4f}",
                    "Max": f"{np.max(values):.4f}",
                    "Range": f"{np.max(values) - np.min(values):.4f}",
                    "Mean": f"{np.mean(values):.4f}",
                    "Std": f"{np.std(values):.4f}",
                })
        
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Driver Statistics
    st.subheader("👤 Driver Statistics")
    
    driver_stats = []
    for driver_id in sorted(data.keys()):
        trajectories = data[driver_id]
        traj_lengths = [len(t) for t in trajectories]
        
        driver_stats.append({
            "Driver ID": driver_id,
            "Trajectory Count": len(trajectories),
            "Avg Traj Length": f"{np.mean(traj_lengths):.1f}" if traj_lengths else "N/A",
            "Min Traj Length": min(traj_lengths) if traj_lengths else "N/A",
            "Max Traj Length": max(traj_lengths) if traj_lengths else "N/A",
            "Total States": sum(traj_lengths),
        })
    
    st.dataframe(pd.DataFrame(driver_stats), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Trajectory Length Distribution",
        "Spatial Coverage",
        "Temporal Distribution",
        "Action Distribution"
    ])
    
    with tab1:
        fig = px.histogram(
            x=stats["states_per_trajectory"],
            nbins=50,
            title="Distribution of Trajectory Lengths",
            labels={"x": "Trajectory Length (states)", "y": "Count"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if len(stats["all_states"]) > 0:
            sample_size = min(50000, len(stats["all_states"]))
            sample_indices = np.random.choice(len(stats["all_states"]), sample_size, replace=False)
            sample_states = [stats["all_states"][i] for i in sample_indices]
            
            x_coords = [s[0] for s in sample_states]
            y_coords = [s[1] for s in sample_states]
            
            fig = px.density_heatmap(
                x=x_coords,
                y=y_coords,
                nbinsx=50,
                nbinsy=90,
                title="Spatial Coverage Heatmap",
                labels={"x": "X Grid", "y": "Y Grid"},
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if len(stats["all_states"]) > 0:
            time_buckets = [s[2] for s in stats["all_states"][:100000]]
            
            fig = px.histogram(
                x=time_buckets,
                nbins=288,
                title="Temporal Distribution (5-minute buckets)",
                labels={"x": "Time Bucket", "y": "Count"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        if len(stats["all_states"]) > 0:
            actions = [s[125] for s in stats["all_states"]]
            action_counts = pd.Series(actions).value_counts().sort_index()
            
            action_df = pd.DataFrame({
                "Action Code": action_counts.index,
                "Count": action_counts.values,
                "Movement": [ACTION_CODES.get(i, "Unknown") for i in action_counts.index]
            })
            
            fig = px.bar(
                action_df,
                x="Action Code",
                y="Count",
                text="Movement",
                title="Action Code Distribution",
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main entry point for the Streamlit app."""
    st.title("🚕 All Trajs Dataset Generator")
    st.markdown("""
    Generate and analyze feature-augmented taxi trajectory datasets for the FAMAIL project.
    This tool processes raw GPS data and creates state vectors with spatial and temporal features.
    """)
    
    # Render sidebar and get configuration
    config_params = render_sidebar()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📦 Generate", "📊 Analyze"])
    
    with tab1:
        render_generate_tab(config_params)
    
    with tab2:
        render_analyze_tab(config_params)


if __name__ == "__main__":
    main()

"""
Active Taxis Dataset Generation - Streamlit Dashboard

A dashboard for generating and validating active taxi count datasets.
These datasets are used by the Spatial Fairness and Causal Fairness
objective function terms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
import time
from typing import Dict, Tuple, Optional

try:
    from .config import ActiveTaxisConfig
    from .generation import (
        generate_active_taxi_counts,
        generate_test_dataset,
        save_output,
        load_output,
        validate_key_format,
    )
    from .processor import ProcessingStats
except ImportError:
    from config import ActiveTaxisConfig
    from generation import (
        generate_active_taxi_counts,
        generate_test_dataset,
        save_output,
        load_output,
        validate_key_format,
    )
    from processor import ProcessingStats


# Page configuration
st.set_page_config(
    page_title="Active Taxis Dataset Generator",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #00cc66;
    }
</style>
""", unsafe_allow_html=True)


def get_default_paths() -> Dict[str, Path]:
    """Get default file paths based on project structure."""
    base_dir = Path(__file__).resolve().parent
    return {
        'raw_data_dir': base_dir.parent / "raw_data",
        'output_dir': base_dir / "output",
    }


def check_raw_data_files(raw_data_dir: Path) -> Dict[str, dict]:
    """Check which raw data files exist."""
    expected_files = [
        "taxi_record_07_50drivers.pkl",
        "taxi_record_08_50drivers.pkl",
        "taxi_record_09_50drivers.pkl",
    ]
    
    status = {}
    for filename in expected_files:
        filepath = raw_data_dir / filename
        status[filename] = {
            'exists': filepath.exists(),
            'path': filepath,
            'size_mb': filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0
        }
    return status


def plot_active_taxi_heatmap(
    data: Dict[Tuple, int],
    period_key: Optional[Tuple],
    grid_dims: Tuple[int, int],
    title: str = "Active Taxis per Cell"
) -> go.Figure:
    """Create a heatmap showing active taxi counts."""
    x_max, y_max = grid_dims
    
    # Initialize grid
    grid = np.zeros((x_max, y_max))
    
    # Fill grid with values for the specified period
    for key, count in data.items():
        x, y = key[0], key[1]
        key_period = key[2:]
        
        if period_key is None or key_period == period_key:
            if 1 <= x <= x_max and 1 <= y <= y_max:
                grid[x - 1, y - 1] += count
    
    fig = px.imshow(
        grid.T,
        labels=dict(x="X Grid", y="Y Grid", color="Active Taxis"),
        title=title,
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    
    fig.update_layout(height=500)
    return fig


def plot_active_taxi_distribution(data: Dict[Tuple, int]) -> go.Figure:
    """Plot histogram of active taxi counts."""
    counts = list(data.values())
    
    fig = px.histogram(
        x=counts,
        nbins=50,
        labels={"x": "Active Taxi Count", "y": "Frequency"},
        title="Distribution of Active Taxi Counts Across All Cells/Periods"
    )
    
    # Add mean line
    mean_val = np.mean(counts)
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.1f}"
    )
    
    fig.update_layout(height=400)
    return fig


def plot_temporal_pattern(
    data: Dict[Tuple, int],
    period_type: str,
    grid_dims: Tuple[int, int]
) -> Optional[go.Figure]:
    """Plot average active taxis over time periods."""
    if period_type == 'all':
        return None
    
    # Aggregate by period
    period_totals = {}
    period_counts = {}
    
    for key, count in data.items():
        x, y = key[0], key[1]
        period_key = key[2:]
        
        if period_key not in period_totals:
            period_totals[period_key] = 0
            period_counts[period_key] = 0
        
        period_totals[period_key] += count
        period_counts[period_key] += 1
    
    # Compute averages
    periods = sorted(period_totals.keys())
    averages = [period_totals[p] / period_counts[p] for p in periods]
    
    # Create labels
    if period_type == 'hourly':
        labels = [f"Hour {p[0]}, Day {p[1]}" for p in periods]
        x_label = "Hour of Day"
        # Group by hour only
        hour_avgs = {}
        for p, avg in zip(periods, averages):
            hour = p[0]
            if hour not in hour_avgs:
                hour_avgs[hour] = []
            hour_avgs[hour].append(avg)
        
        hours = sorted(hour_avgs.keys())
        avg_by_hour = [np.mean(hour_avgs[h]) for h in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=avg_by_hour,
            mode='lines+markers',
            name='Avg Active Taxis'
        ))
        fig.update_layout(
            title="Average Active Taxis by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Average Active Taxis per Cell",
            height=400,
        )
        return fig
        
    elif period_type == 'time_bucket':
        # Group by time bucket only
        bucket_avgs = {}
        for p, avg in zip(periods, averages):
            bucket = p[0]
            if bucket not in bucket_avgs:
                bucket_avgs[bucket] = []
            bucket_avgs[bucket].append(avg)
        
        buckets = sorted(bucket_avgs.keys())
        avg_by_bucket = [np.mean(bucket_avgs[b]) for b in buckets]
        
        # Convert to hours for readability
        hours = [b / 12 for b in buckets]  # 12 buckets per hour
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=avg_by_bucket,
            mode='lines',
            name='Avg Active Taxis',
            line=dict(width=1)
        ))
        fig.update_layout(
            title="Average Active Taxis by Time of Day (5-min intervals)",
            xaxis_title="Hour of Day",
            yaxis_title="Average Active Taxis per Cell",
            height=400,
        )
        return fig
        
    elif period_type == 'daily':
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        days = sorted(set(p[0] for p in periods))
        
        # Group by day
        day_avgs = {}
        for p, avg in zip(periods, averages):
            day = p[0]
            if day not in day_avgs:
                day_avgs[day] = []
            day_avgs[day].append(avg)
        
        avg_by_day = [np.mean(day_avgs.get(d, [0])) for d in days]
        labels = [day_names[d - 1] if d <= 6 else f"Day {d}" for d in days]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=avg_by_day,
            name='Avg Active Taxis'
        ))
        fig.update_layout(
            title="Average Active Taxis by Day of Week",
            xaxis_title="Day",
            yaxis_title="Average Active Taxis per Cell",
            height=400,
        )
        return fig
    
    return None


def main():
    st.title("🚖 Active Taxis Dataset Generator")
    st.markdown("""
    This tool generates datasets that count active taxis in an n×n grid neighborhood
    for each cell and time period. These datasets support the **Spatial Fairness** and
    **Causal Fairness** objective function terms.
    
    An "active taxi" is any taxi that was present in the neighborhood during the time period.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    paths = get_default_paths()
    
    # Mode selection
    mode = st.sidebar.radio(
        "Mode",
        ["Generate Dataset", "Load Existing Dataset"],
        help="Generate a new dataset or load an existing one"
    )
    
    if mode == "Generate Dataset":
        # Path configuration
        with st.sidebar.expander("📁 File Paths", expanded=False):
            raw_data_dir = st.text_input(
                "Raw Data Directory",
                value=str(paths['raw_data_dir']),
                help="Directory containing taxi_record_XX_50drivers.pkl files"
            )
            
            output_dir = st.text_input(
                "Output Directory",
                value=str(paths['output_dir']),
                help="Directory to save generated datasets"
            )
        
        # Check raw data availability
        raw_data_path = Path(raw_data_dir)
        file_status = check_raw_data_files(raw_data_path)
        
        with st.sidebar.expander("📊 Raw Data Files", expanded=True):
            for filename, status in file_status.items():
                if status['exists']:
                    st.success(f"✅ {filename} ({status['size_mb']:.1f} MB)")
                else:
                    st.warning(f"⚠️ {filename} (not found)")
        
        # Available files for selection
        available_files = [f for f, s in file_status.items() if s['exists']]
        
        if not available_files:
            st.error("No raw data files found. Please check the raw data directory.")
            return
        
        # Generation parameters
        with st.sidebar.expander("🔧 Generation Parameters", expanded=True):
            neighborhood_size = st.slider(
                "Neighborhood Size (k)",
                min_value=0,
                max_value=5,
                value=2,
                help="Neighborhood is (2k+1) × (2k+1). k=2 means 5×5 neighborhood."
            )
            st.caption(f"Neighborhood: {2*neighborhood_size+1} × {2*neighborhood_size+1} cells")
            
            period_type = st.selectbox(
                "Period Type",
                options=['hourly', 'daily', 'time_bucket', 'all'],
                index=0,
                help="Time aggregation granularity"
            )
            
            selected_files = st.multiselect(
                "Input Files",
                options=available_files,
                default=available_files[:1],
                help="Select which months to process"
            )
            
            test_mode = st.checkbox(
                "Test Mode (small subset)",
                value=True,
                help="Process only a small subset for validation"
            )
            
            if test_mode:
                test_drivers = st.number_input(
                    "Sample Drivers",
                    min_value=1,
                    max_value=50,
                    value=5,
                    help="Number of drivers to sample"
                )
                test_days = st.number_input(
                    "Sample Days",
                    min_value=1,
                    max_value=21,
                    value=3,
                    help="Number of days per month to sample"
                )
            else:
                test_drivers = 50
                test_days = 21
        
        # Output filename
        default_filename = f"active_taxis_{2*neighborhood_size+1}x{2*neighborhood_size+1}_{period_type}"
        if test_mode:
            default_filename += "_test"
        default_filename += ".pkl"
        
        output_filename = st.sidebar.text_input(
            "Output Filename",
            value=default_filename,
            help="Name for the output file"
        )
        
        # Generate button
        st.sidebar.divider()
        generate_button = st.sidebar.button(
            "🚀 Generate Dataset",
            type="primary",
            use_container_width=True
        )
        
        # Main content area
        if generate_button:
            # Create config
            config = ActiveTaxisConfig(
                neighborhood_size=neighborhood_size,
                period_type=period_type,
                test_mode=test_mode,
                test_sample_size=test_drivers,
                test_days=test_days,
                raw_data_dir=Path(raw_data_dir),
                output_dir=Path(output_dir),
                input_files=tuple(selected_files),
            )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(stage: str, progress: float):
                progress_bar.progress(progress)
                status_text.text(stage)
            
            try:
                # Generate dataset
                with st.spinner("Generating dataset..."):
                    data, stats = generate_active_taxi_counts(config, update_progress)
                
                st.success(f"✅ Dataset generated successfully!")
                
                # Save to session state for visualization
                st.session_state['active_taxi_data'] = data
                st.session_state['active_taxi_stats'] = stats
                st.session_state['active_taxi_config'] = config
                
                # Save to file
                output_path = config.output_dir / output_filename
                save_output(data, stats, config, output_path)
                st.info(f"💾 Saved to: {output_path}")
                
            except Exception as e:
                st.error(f"Error generating dataset: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    else:  # Load Existing Dataset
        with st.sidebar.expander("📁 Load Dataset", expanded=True):
            output_dir = st.text_input(
                "Output Directory",
                value=str(paths['output_dir']),
            )
            
            output_path = Path(output_dir)
            if output_path.exists():
                pkl_files = list(output_path.glob("*.pkl"))
                if pkl_files:
                    selected_file = st.selectbox(
                        "Select Dataset",
                        options=pkl_files,
                        format_func=lambda x: x.name
                    )
                    
                    if st.button("Load Dataset", type="primary"):
                        try:
                            data, stats_dict, config_dict = load_output(selected_file)
                            
                            # Convert stats dict to ProcessingStats
                            stats = ProcessingStats(**{
                                k: v for k, v in stats_dict.items()
                                if k != 'global_bounds'
                            })
                            
                            # Recreate config
                            period_type = config_dict.get('period_type', 'hourly')
                            neighborhood_size = config_dict.get('neighborhood_size', 2)
                            
                            config = ActiveTaxisConfig(
                                period_type=period_type,
                                neighborhood_size=neighborhood_size,
                            )
                            
                            st.session_state['active_taxi_data'] = data
                            st.session_state['active_taxi_stats'] = stats
                            st.session_state['active_taxi_config'] = config
                            
                            st.success(f"✅ Loaded {len(data):,} records")
                        except Exception as e:
                            st.error(f"Error loading dataset: {e}")
                else:
                    st.warning("No .pkl files found in output directory")
            else:
                st.warning("Output directory does not exist")
    
    # Visualization section
    st.divider()
    
    if 'active_taxi_data' in st.session_state:
        data = st.session_state['active_taxi_data']
        stats = st.session_state['active_taxi_stats']
        config = st.session_state['active_taxi_config']
        
        # Display statistics
        st.header("📊 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{stats.total_output_keys:,}")
        
        with col2:
            st.metric("Max Active Taxis", f"{stats.max_active_taxis_in_cell}")
        
        with col3:
            st.metric("Avg Active Taxis", f"{stats.avg_active_taxis_per_cell:.2f}")
        
        with col4:
            st.metric("Processing Time", f"{stats.processing_time_seconds:.1f}s")
        
        # Additional stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("GPS Records", f"{stats.total_records:,}")
        
        with col2:
            st.metric("Unique Taxis", f"{stats.unique_taxis}")
        
        with col3:
            st.metric("Unique Periods", f"{stats.unique_periods}")
        
        with col4:
            st.metric("Unique Cells", f"{stats.unique_cells}")
        
        # Key format validation
        validation = validate_key_format(data, config.period_type)
        if validation['valid']:
            st.success(f"✅ Key format valid for period_type='{config.period_type}'")
        else:
            st.warning(f"⚠️ Key format may not match expected for '{config.period_type}'")
        
        with st.expander("📋 Sample Keys"):
            st.code(str(validation['sample_keys']))
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ Spatial Distribution",
            "📈 Temporal Patterns",
            "📊 Count Distribution",
            "🔍 Data Explorer"
        ])
        
        with tab1:
            st.subheader("Active Taxi Counts Heatmap")
            
            # Aggregate across all periods
            fig_heatmap = plot_active_taxi_heatmap(
                data,
                period_key=None,
                grid_dims=config.grid_dims,
                title="Total Active Taxi Count (All Periods)"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab2:
            st.subheader("Temporal Patterns")
            
            fig_temporal = plot_temporal_pattern(data, config.period_type, config.grid_dims)
            if fig_temporal:
                st.plotly_chart(fig_temporal, use_container_width=True)
            else:
                st.info("No temporal pattern visualization for 'all' period type.")
        
        with tab3:
            st.subheader("Distribution of Active Taxi Counts")
            
            fig_dist = plot_active_taxi_distribution(data)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Summary statistics
            counts = list(data.values())
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min", min(counts))
            with col2:
                st.metric("Max", max(counts))
            with col3:
                st.metric("Mean", f"{np.mean(counts):.2f}")
            with col4:
                st.metric("Std Dev", f"{np.std(counts):.2f}")
        
        with tab4:
            st.subheader("Data Explorer")
            
            # Cell lookup
            st.markdown("#### Look Up Active Taxis for a Cell")
            
            col1, col2 = st.columns(2)
            with col1:
                x_lookup = st.number_input(
                    "X Grid",
                    min_value=1,
                    max_value=config.grid_dims[0],
                    value=25
                )
            with col2:
                y_lookup = st.number_input(
                    "Y Grid",
                    min_value=1,
                    max_value=config.grid_dims[1],
                    value=45
                )
            
            # Find all entries for this cell
            cell_entries = {k: v for k, v in data.items() if k[0] == x_lookup and k[1] == y_lookup}
            
            if cell_entries:
                st.markdown(f"**Found {len(cell_entries)} periods for cell ({x_lookup}, {y_lookup}):**")
                
                # Convert to DataFrame
                df_entries = pd.DataFrame([
                    {'Key': str(k), 'Period': str(k[2:]), 'Active Taxis': v}
                    for k, v in sorted(cell_entries.items())
                ])
                st.dataframe(df_entries, use_container_width=True, height=300)
                
                # Summary for this cell
                st.markdown(f"""
                - **Min active taxis:** {min(cell_entries.values())}
                - **Max active taxis:** {max(cell_entries.values())}
                - **Mean active taxis:** {np.mean(list(cell_entries.values())):.2f}
                """)
            else:
                st.info(f"No data found for cell ({x_lookup}, {y_lookup})")
        
        # Configuration details
        with st.expander("🔧 Configuration Details"):
            st.json(config.to_dict())
    
    else:
        st.info("Generate or load a dataset to see visualizations.")
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **Active Taxis Dataset Generator** | Part of FAMAIL Objective Function
    
    The generated datasets enable efficient lookup of active taxi counts for
    spatial fairness calculations without processing raw GPS data during runtime.
    """)


if __name__ == "__main__":
    main()

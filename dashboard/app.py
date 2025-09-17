import os
from pathlib import Path
import streamlit as st
from dashboard.ui import load_css, card_header
from dashboard.data import get_app_data
import dashboard.data as data
import dashboard.plots as plots
from millify import prettify
from dashboard.constants import TARGET_SPECIES

##########################################################################
# INITIALISE LAYOUT
##########################################################################

st.set_page_config(
    page_title="FrogID",
    page_icon="🐸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS stylesheet
load_css("dashboard/styles.css")

##########################################################################
# Get Initial Data
##########################################################################

# Define the project root
PROJECT_ROOT = Path(os.getcwd())

df, summary = get_app_data(PROJECT_ROOT, TARGET_SPECIES)

##########################################################################
# Generate Layout
##########################################################################

tab_overview, tab_single_species, tab_multi_species, tab_experiments = st.tabs(["Overview", "Single Species Overview", "Multi Species Overview", "Experiment Exploration"])

##########################################################################
# Summary of the Overall Dataset
##########################################################################
with tab_overview:

    raw = summary['raw']
    filtered = summary['filtered']

    # Show summary statistics
    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Dataset Size",
        value=prettify(raw['count_total']),
        delta=f"Original dataset",
        delta_color="normal",
        border=True
    )
    col2.metric(
        label="Filtered Data",
        value=prettify(filtered['count_total']),
        delta=f"{filtered['percent_filtered:raw']}% of full dataset",
        delta_color="normal",
        border=True
    )
    col3.metric(
        label="Species Count", 
        value=prettify(filtered['num_species']), 
        delta=f"Note: Full data had {raw['num_species']} species",
        delta_color="normal",
        border=True
    )

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border = True):
            card_header(
                title = f"Filtering Criteria of the Raw Data",
                desc = f"""
                    The dataset started with {prettify(summary['raw']['count_total'])} recordings.
                    However, not all these recordings were suitable for training and evaluation.
                    Therefore, we filtered the data according to a specific criteria, which culminated
                    in {prettify(summary['filtered']['count_total'])}, representing 
                    {prettify(summary['filtered']['percent_filtered:raw'])}% of the original data.
                """
            )
            st.plotly_chart(plots.plot_filtering_process(summary), use_container_width=True)
        with st.container(border = True):
            card_header(
                title = f"Species Occurrence Distribution",
                desc = f"""
                    Of the {prettify(filtered['count_total'])} filtered recordings, there
                    were a total of {prettify(filtered['count_species_occurrences'])} species
                    occurrences across {filtered['num_species']} species types. Note: The 
                    increase in number is because multi-species recordings have co-occurrence. 
                    The distribution of the species can be seen below.
                """
            )
            st.plotly_chart(plots.plot_occurrence_distribution(df), use_container_width=True)
        
    with col2:
        with st.container(border = True):
            card_header(
                title = f"Temporal Analysis",
                desc = f"""
                    Across the {prettify(filtered['count_total'])} recordings, we were
                    interested in the temporal patterns that may exist around the recordings. We stratified
                    the recordings based on when they were captured, and identified that recordings
                    tended to be captured in November and around 8-9pm most prevalently.
                """
            )
            species_by_year, species_by_month, species_by_hour = st.tabs(["Records by Year", "Records by Month", "Records by Hour"])
            with species_by_year:
                st.plotly_chart(plots.plot_records_by_year(df), use_container_width=True)
            with species_by_month:
                st.plotly_chart(plots.plot_records_by_month(df), use_container_width=True)
            with species_by_hour:
                st.plotly_chart(plots.plot_records_by_hour(df), use_container_width=True)
        with st.container(border = True):
            card_header(
                title = f"Target Species Occurrences",
                desc = f"""
                    Within the {prettify(filtered['count_species_occurrences'])}, there are
                    {prettify(filtered['count_target_occurrences'])} occurrences of
                    the {len(TARGET_SPECIES)} target species. These {len(TARGET_SPECIES)} species account for 
                    {filtered['percent_target:species']}% of the total dataset.
                """
            )
            target_by_species, target_by_state, target_by_year = st.tabs(["Explore by Species", "Explore by State", "Explore by Year"])
            with target_by_species:
                st.plotly_chart(plots.plot_target_occurrence_distribution(df, TARGET_SPECIES), use_container_width=True)
            with target_by_state:
                st.plotly_chart(plots.plot_target_occurrence_by_state(df, TARGET_SPECIES), use_container_width=True)
            with target_by_year:
                st.plotly_chart(plots.plot_target_occurrence_by_year(df, TARGET_SPECIES), use_container_width=True)
    
    df_temp = data.get_occurrence_distribution(df, TARGET_SPECIES)
    df_temp['Count (#)'] = df_temp['Count'].apply(lambda x: str(x))
    df_temp['Percentage (%)'] = df_temp['Percentage'].apply(lambda x: str(x) + '%')
    df_temp = df_temp[['Species', 'Count (#)', 'Percentage (%)']]

    st.dataframe(df_temp, height=700)

##########################################################################
# Summary of the Single Species Data
##########################################################################
with tab_single_species:

    single = summary['single']
    # Show summary statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="Single Species Recordings", 
        value=prettify(single['count_total']), 
        delta=f"{prettify(single['percent_single:total'])}% of filtered data",
        delta_color="normal",
        border=True
    )
    col2.metric(
        label="Target Species Count", 
        value=prettify(single['count_target']), 
        delta=f"{prettify(single['percent_target:single'])}% of single species data",
        delta_color="normal",
        border=True
    )
    col3.metric(
        label="Other Species Count", 
        value=prettify(single['count_other']), 
        delta=f"{prettify(single['percent_other:single'])}% of single species data",
        delta_color="normal",
        border=True
    )
    col4.metric(
        label="Single Species Types", 
        value=prettify(single['num_species']), 
        delta=f"Found in the single species data",
        delta_color="normal",
        border=True
    )

    # Show the plots for the target vs other species
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            card_header(
                title = f"Distribution of {single['num_target_species']} Target Species",
                subtitle = "Found in Single Species Recordings",
                desc = f"""
                    There are {prettify(single['count_target'])} target species recordings, 
                    that make up {single['percent_target:single']}% of the single species data, 
                    and {single['percent_target:total']}% of the full data.
                """
            )
            st.plotly_chart(plots.plot_single_species_target_distribution(df), use_container_width=True)
            
    with col2:
        with st.container(border=True):
            card_header(
                title = f"Distribution of {single['num_other_species']} Other Species",
                subtitle = "Found in Single Species Recordings",
                desc = f"""
                    There are {prettify(single['count_other'])} other species recordings, 
                    that make up {single['percent_other:single']}% of the single species data, 
                    and {single['percent_other:total']}% of the full data.
                """
            )
            plots.plot_single_species_other_distribution(df)



##########################################################################
# Summary of the Multi Species Data
##########################################################################
with tab_multi_species:
    multi = summary['multi']
    # Show summary statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="Multi Species Recordings", 
        value=prettify(multi['count_total']), 
        delta=f"{prettify(multi['percent_multi:total'])}% of filtered data",
        delta_color="normal",
        border=True
    )
    col2.metric(
        label="Target Species Count", 
        value=prettify(multi['count_target']), 
        delta=f"{prettify(multi['percent_target:multi'])}% of multi species data",
        delta_color="normal",
        border=True
    )
    col3.metric(
        label="Other Species Count", 
        value=prettify(multi['count_other']), 
        delta=f"{prettify(multi['percent_other:multi'])}% of multi species data",
        delta_color="normal",
        border=True
    )
    col4.metric(
        label="Multi Species Types", 
        value=prettify(multi['num_species']), 
        delta=f"Found in the multi species data",
        delta_color="normal",
        border=True
    )

    col1, col2 = st.columns(2)
    with col1:
         with st.container(border = True):
            card_header(
                title = f"Distribution of {len(TARGET_SPECIES)} Target Species",
                subtitle = "Found in Multi Species Recordings",
                desc = f"""
                    There are {prettify(multi['count_target'])} target species recordings, 
                    that make up {multi['percent_target:multi']}% of the multi species data, 
                    and {multi['percent_other:multi']}% of the full data.
                """
            )
            st.plotly_chart(plots.plot_multi_species_target_distribution(df, TARGET_SPECIES) , use_container_width=True)
         with st.container(border=True):
            card_header(
                title = f"Number of Species per Recording",
                desc = f"""
                    Every recording can have many frog species associated with it, as frogs can co-occur
                    in the same habitat. Each recording has, on average, {round(df['species_count'].mean(),2)} 
                    species within it. The distribution of the number of species across recordings is shown below.
                """
            )
            st.plotly_chart(plots.plot_number_of_species_per_recording(df), use_container_width=True)
    with col2:
        with st.container(border = True):
            card_header(
                title = f"Distribution of {multi['num_other_species']} Other Species",
                subtitle = "Found in Multi Species Recordings",
                desc = f"""
                    There are {prettify(multi['count_other'])} other species recordings, 
                    that make up {multi['percent_other:multi']}% of the multi species data, 
                    and {multi['percent_other:multi']}% of the full data.
                """
            )
            st.plotly_chart(plots.plot_multi_species_other_distribution(df, TARGET_SPECIES), use_container_width=True)
        with st.container(border = True):
            card_header(
                title = f"Co-Occurrence Analysis (2 Species)",
                desc = f"""
                    Since recordings with 2 species make up the majority of the multi-species recordings,
                    we wanted to understand the nature of its distribution. The possible combinations
                    are a Target Species with another Target, Target with Other species, or Other with an Other.
                """
            )
            st.plotly_chart(plots.plot_multi_two_species(df, TARGET_SPECIES), use_container_width=True)
            
            
    


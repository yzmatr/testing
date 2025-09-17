import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dashboard.data as data
from dashboard.constants import TARGET_SPECIES
from typing import List

DEFAULT_CHART_COLOR = "teal"
DEFAULT_CHART_MARGIN = dict(t=40, l=100, r=80, b=60)
DEFAULT_CHART_HEIGHT = 500

##########################################################################
# Plots for Overall Dataset
##########################################################################

def plot_filtering_process(summary: dict):
    # Generate a Data Frame showing the filter steps
    df_filters = pd.DataFrame(summary['raw']['filter_process'], columns=['Step', 'Count'])
    df_filters['Step Number'] = df_filters.index.map(lambda i: f"Step {i+1}")
    df_filters['Percentage'] = round(df_filters['Count'] / summary['raw']['count_total'] * 100)

    # Generate a plotly figure
    fig = px.bar(
        df_filters,
        x='Count',
        y='Step Number',
        color_discrete_sequence=[DEFAULT_CHART_COLOR],
        text="Step",
        orientation="h"
    )

    # Improve the layout of the traces
    fig.update_traces(
        texttemplate='%{text}',
        textposition='inside',
        hovertemplate=(
            '<b>Remaining Count:</b> %{x:,}<br>'
            '<b>Percentage of Data:</b> %{customdata[0]:.1f}%'
            '<extra></extra>'
        ),
        customdata=df_filters[['Percentage']].values

    )

    # Improve the layout of the figure itself
    fig.update_layout(
        xaxis_title="# of Records",
        yaxis_title="Step",
        margin=DEFAULT_CHART_MARGIN,
        height=DEFAULT_CHART_HEIGHT
    )

    return fig

def plot_occurrence_distribution(df: pd.DataFrame):
    df_species_counts = data.get_occurrence_distribution(df)
    
    # Generate a tree map
    fig = px.treemap(
        df_species_counts,
        path=['Species'],
        values='Count',
        color='Count',
        color_continuous_scale='PuBuGn'
    )

    fig.update_traces(
        texttemplate=(
            '%{label}<br>'
            '%{value}<br>'
            '%{customdata[0]:.1f}%'
        ),
        textposition='middle center',
        textfont_size=14,
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Count: %{value:,}'
            'Percentage: %{customdata[0]:.1f}'
            '<extra></extra>'
        ),
        customdata=df_species_counts[['Percentage']].values
    )

    fig.update_layout(
        height=DEFAULT_CHART_HEIGHT,
        margin=dict(t=0, l=0, r=0, b=0)
    )

    return fig

def plot_target_occurrence_distribution(df: pd.DataFrame, target_species: List[str]):
    counts = data.get_occurrence_distribution(df, target_species)
    fig = px.bar(
        counts,
        x="Species",
        y="Count",
        color_discrete_sequence=[DEFAULT_CHART_COLOR],
        text="Percentage"
    )

    # Improve Layout
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate=(
            '<b>Species:</b> %{x}<br>'
            '<b>Count:</b> %{y:,}<br>'
            '<b>Percentage:</b> %{customdata[0]:.1f}%<extra></extra>'
        ),
        customdata=counts[['Percentage']].values
    )

    fig.update_layout(
        xaxis_title="Species",
        yaxis_title="Count",
        margin=DEFAULT_CHART_MARGIN,
        height=DEFAULT_CHART_HEIGHT
    )
    return fig

def plot_target_occurrence_by_state(df: pd.DataFrame, target_species: List[str]):
    df_plot = data.get_target_distribution_by_state(df, target_species)
    # -------------------------
    # Compute state order
    # -------------------------

    state_totals = (
        df_plot
        .groupby("State")["Count"]
        .sum()
        .sort_values(ascending=False)
    )

    state_order = state_totals.index.tolist()

    # -------------------------
    # Compute species order
    # -------------------------

    species_totals = (
        df_plot
        .groupby("Species")["Count"]
        .sum()
        .sort_values(ascending=False)
    )

    species_order = species_totals.index.tolist()

    # -------------------------
    # Set categories
    # -------------------------

    df_plot["State"] = pd.Categorical(
        df_plot["State"],
        categories=state_order,
        ordered=True
    )

    df_plot["Species"] = pd.Categorical(
        df_plot["Species"],
        categories=species_order,
        ordered=True
    )

    # -------------------------
    # Assign colors
    # -------------------------

    state_colors = {
        state: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, state in enumerate(state_order)
    }

    # -------------------------
    # Plot stacked bars
    # -------------------------

    fig = px.bar(
        df_plot,
        x="Species",
        y="Count",
        color="State",
        category_orders={
            "State": state_order,
            "Species": species_order
        },
        color_discrete_map=state_colors,
        text="Count"
    )

    fig.update_traces(textposition='inside')

    fig.update_layout(
        barmode="stack",
        yaxis_title="Count",
        xaxis_title="Species",
        margin=DEFAULT_CHART_MARGIN,
        height=DEFAULT_CHART_HEIGHT
    )

    return fig

def plot_target_occurrence_by_year(df: pd.DataFrame, target_species: List[str]):
    # Filter based on capture_time
    df_temporal = df.copy()
    # df_temporal['year'] = df_temporal['capture_time'].dt.year
    df_temporal['year'] = df["capture_time"].apply(
        lambda x: x.year if pd.notnull(x) else None
    )
    rows = []
    for species in target_species:
        df_filtered = df_temporal[
            df_temporal['sorted_species_names'].apply(lambda species_list: species in species_list)
        ]
        counts = df_filtered['year'].value_counts().sort_index()
        df_counts = counts.reset_index()
        df_counts.columns = ['Year', 'Count']
        df_counts['Species'] = species
        rows.append(df_counts)

    df_species = pd.concat(rows)
    df_grouped = (
        df_species
        .groupby(['Species', 'Year'])["Count"]
        .sum()
        .reset_index()
    )

    # Identify top species
    species_totals = (
        df_grouped
        .groupby("Species")["Count"]
        .sum()
        .sort_values(ascending=False)
    )

    top_species = species_totals.head(3).index.tolist()

    # Colors
    species_list = df_grouped["Species"].unique().tolist()
    colors = px.colors.qualitative.D3
    color_map = {
        species: colors[i % len(colors)]
        for i, species in enumerate(species_list)
    }

    # Plot
    fig = go.Figure()

    for species in species_list:
        df_species_data = df_grouped[df_grouped["Species"] == species]

        fig.add_trace(
            go.Scatter(
                x=df_species_data["Year"],
                y=df_species_data["Count"],
                mode="lines+markers",
                name=species,
                line=dict(color=color_map[species]),
                visible=True if species in top_species else "legendonly",
                customdata=df_species_data[["Species"]],
                hovertemplate=(
                    "Species: %{customdata[0]}<br>"
                    "Year: %{x}<br>"
                    "Count: %{y:,}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Count",
        margin=DEFAULT_CHART_MARGIN,
        height=DEFAULT_CHART_HEIGHT
    )

    return fig

def plot_records_by_year(df: pd.DataFrame):
    # Filter based on capture_tim
    df_temporal = df.copy()
    # df_temporal['year'] = df_temporal['capture_time'].dt.year
    df_temporal['year'] = df["capture_time"].apply(
        lambda x: x.year if pd.notnull(x) else None
    )
    counts = df_temporal['year'].value_counts().sort_index()
    df_counts = counts.reset_index()
    df_counts.columns = ['Year', 'Count']
    df_counts['Percentage'] = round(df_counts['Count'] / len(df_temporal) * 100)

    fig = px.line(
        df_counts,
        x="Year",
        y="Count",
        markers=True,
        color_discrete_sequence=[DEFAULT_CHART_COLOR]
    )

    # Improve the layout of the traces
    fig.update_traces(
        hovertemplate=(
            '<b>Year:</b> %{x}<br>'
            '<b>Count:</b> %{y:,}<br>'
            '<b>Percentage of Data:</b> %{customdata[0]:.1f}%'
            '<extra></extra>'
        ),
        customdata=df_counts[['Percentage']].values
    )

    # Improve the layout of the figure itself
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Records",
        margin=DEFAULT_CHART_MARGIN,
        height=DEFAULT_CHART_HEIGHT - 60
    )

    return fig

def plot_records_by_month(df: pd.DataFrame):
    # Filter based on capture_time
    df_temporal = df.copy()

    # Logic for month
    # df_temporal['month'] = df_temporal['capture_time'].dt.month
    df_temporal['month'] = df["capture_time"].apply(
        lambda x: x.month if pd.notnull(x) else None
    )
    monthly_counts = df_temporal['month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_counts.index = monthly_counts.index.map(lambda m: month_names[m - 1])
    df_counts = monthly_counts.reset_index()
    df_counts.columns = ['Month', 'Count']
    df_counts['Percentage'] = round(df_counts['Count'] / len(df_temporal) * 100)

    fig = px.bar(
        df_counts,
        x="Month",
        y="Count",
        color_discrete_sequence=[DEFAULT_CHART_COLOR],
        text="Percentage"
    )

    # Improve the layout of the traces
    fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            hovertemplate=(
                '<b>Month:</b> %{x}<br>'
                '<b>Count:</b> %{y:,}<br>'
                '<b>Percentage:</b> %{customdata[0]:.1f}%<extra></extra>'
            ),
            customdata=df_counts[['Percentage']].values
        )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Records",
        margin=DEFAULT_CHART_MARGIN,
        height=DEFAULT_CHART_HEIGHT - 60
    )

    return fig

def plot_records_by_hour(df: pd.DataFrame):
    # Filter based on capture_time
    df_temporal = df.copy()

    # Logic for month
    df_temporal['hour'] = df["capture_time"].apply(
        lambda x: x.hour if pd.notnull(x) else None
    )
    hourly_counts = df_temporal['hour'].value_counts().sort_index()
    df_counts = hourly_counts.reset_index()
    df_counts.columns = ['Hour', 'Count']
    df_counts['Percentage'] = round(df_counts['Count'] / len(df_temporal) * 100)

    fig = px.bar(
        df_counts,
        x="Hour",
        y="Count",
        color_discrete_sequence=[DEFAULT_CHART_COLOR],
        text="Percentage"
    )

    # Improve the layout of the traces
    fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            hovertemplate=(
                '<b>Hour:</b> %{x}<br>'
                '<b>Count:</b> %{y:,}<br>'
                '<b>Percentage:</b> %{customdata[0]:.1f}%<extra></extra>'
            ),
            customdata=df_counts[['Percentage']].values
        )

    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Number of Records",
        margin=DEFAULT_CHART_MARGIN,
        height=DEFAULT_CHART_HEIGHT - 60
    )
    
    return fig

##########################################################################
# Plots for Single Species Dataset
# 1. Target Species Distribution
# 2. Other Species Distribution
##########################################################################

def plot_single_species_target_distribution(df: pd.DataFrame):
    """
    Plot the distribution of the number of recordings for each target species
    in the dataset as a bar chart using Plotly.
    """
    # Create a subset of the dataframe that only includes target species and single-species recordings
    df_subset = df[(df['includes_target_species'] == True) & (df['is_multi_species'] == False)]

    counts = df_subset['first_species_name'].value_counts()
    counts = counts.reset_index()
    counts.columns = ['Species', 'Count']
    
    counts['Percentage'] = counts['Count'] / counts['Count'].sum() * 100

    # Sort species descending
    counts = counts.sort_values(by='Count', ascending=False)

    # Create the bar chart
    fig = px.bar(
        counts,
        x='Species',
        y='Count',
        color_discrete_sequence=[DEFAULT_CHART_COLOR],
        text='Percentage',
    )

    # Improve Layout
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate=(
            '<b>Species:</b> %{x}<br>'
            '<b>Count:</b> %{y:,}<br>'
            '<b>Percentage:</b> %{customdata[0]:.1f}%<extra></extra>'
        ),
        customdata=counts[['Percentage']].values
    )

    fig.update_layout(
        xaxis_title="Species",
        yaxis_title="Count",
        margin=DEFAULT_CHART_MARGIN,
        height=DEFAULT_CHART_HEIGHT
    )

    return fig

def plot_single_species_other_distribution(df: pd.DataFrame):
    # Get the other species subset
    df_subset = df[(df['includes_target_species'] == False) & (df['is_multi_species'] == False)]
    counts = df_subset['first_species_name'].value_counts()
    counts = counts.reset_index()
    counts.columns = ['Species', 'Count']

    # Create the treemap
    fig = px.treemap(
        counts,
        path=['Species'],
        values='Count',
        color='Count',
        color_continuous_scale='PuBuGn',
    )

    fig.update_traces(
        texttemplate='%{label}<br>%{value}',
        textposition='middle center',
        textfont_size=14,
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<extra></extra>'
    )

    fig.update_layout(
        height=DEFAULT_CHART_HEIGHT,
        margin=dict(t=0, l=0, r=0, b=0)
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

##########################################################################
# Plots for Multi Species Dataset
# 1. Target Species Distribution
# 2. Other Species Distribution
##########################################################################

def plot_multi_species_target_distribution(df: pd.DataFrame, target_species: list[str]):
    """
    Plot the distribution of the number of recordings for each target species
    in the dataset as a bar chart using Plotly.
    """

    # Create a subset of the dataframe that only includes target species and single-species recordings
    df_subset = df[(df['includes_target_species'] == True) & (df['is_multi_species'] == True)]

    # Get the counts for the target species
    counts = df_subset['sorted_species_names'].explode().value_counts()
    counts = counts[counts.index.isin(target_species)]
    counts = counts.reset_index()
    counts.columns = ['Species', 'Count']
    counts['Percentage'] = counts['Count'] / df_subset.shape[0] * 100

    # Create the bar chart
    fig = px.bar(
        counts, 
        x='Species', 
        y='Count', 
        color_discrete_sequence=[DEFAULT_CHART_COLOR],
        text='Percentage',
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate=(
            '<b>Species:</b> %{x}<br>'
            '<b>Count:</b> %{y:,}<br>'
            '<b>Percentage:</b> %{customdata[0]:.1f}%<extra></extra>'
        ),
        customdata=counts[['Percentage']].values
    )
    fig.update_layout(
        xaxis_title="Species",
        yaxis_title="Count",
        margin=dict(t=40, l=100, r=80, b=60),
        height=500
    )

    return fig

def plot_multi_species_other_distribution(df: pd.DataFrame, target_species: list[str]):
    # Get the other species subset
    df_subset = df[(df['includes_target_species'] == False) & (df['is_multi_species'] == True)]
    counts = df_subset['sorted_species_names'].explode().value_counts()
    # Remove the target species from the count
    counts = counts[~counts.index.isin(target_species)]
    counts = counts.reset_index()
    counts.columns = ['Species', 'Count']

    # Create the treemap
    fig = px.treemap(
        counts,
        path=['Species'],
        values='Count',
        color='Count',
        color_continuous_scale='PuBuGn',
    )

    fig.update_traces(
        texttemplate='%{label}<br>%{value}',
        textposition='middle center',
        textfont_size=14,
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<extra></extra>'
    )

    fig.update_layout(
        height=500,
        margin=dict(t=0, l=0, r=0, b=0)
    )

    return fig

def plot_number_of_species_per_recording(df: pd.DataFrame):
    # Count the number of species per recording
    df_species = df.copy()

    # Logic for month
    counts = df_species['species_count'].value_counts().sort_index()
    df_counts = counts.reset_index()
    df_counts.columns = ['Species', 'Count']
    df_counts['Percentage'] = round(df_counts['Count'] / len(df_species) * 100)

    fig = px.bar(
        df_counts,
        x="Species",
        y="Count",
        color_discrete_sequence=[DEFAULT_CHART_COLOR],
        text="Percentage"
    )

    # Improve the layout of the traces
    fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            hovertemplate=(
                '<b>Number of Species:</b> %{x}<br>'
                '<b>Count:</b> %{y:,}<br>'
                '<b>Percentage:</b> %{customdata[0]:.1f}%<extra></extra>'
            ),
            customdata=df_counts[['Percentage']].values
        )

    fig.update_layout(
        xaxis_title="Number of Species per Recording",
        yaxis_title="Number of Records",
        margin=DEFAULT_CHART_MARGIN,
        height=DEFAULT_CHART_HEIGHT
    )

    return fig

def plot_multi_two_species(df: pd.DataFrame, target_species: List[str]):
    # Filter to two species records
    df_two_species = df[df['species_count'] == 2]
    # Count the number of species per recording
    df_pairs = data.get_combination_pairs(df_two_species, target_species)

    # Logic for month
    df_counts = df_pairs['combination_type'].value_counts()
    df_counts = df_counts.reset_index()
    df_counts.columns = ['Combination', 'Count']
    df_counts['Percentage'] = round(df_counts['Count'] / len(df_two_species) * 100)

    fig = px.bar(
        df_counts,
        x="Combination",
        y="Count",
        color_discrete_sequence=[DEFAULT_CHART_COLOR],
        text="Percentage"
    )

    # Improve the layout of the traces
    fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            hovertemplate=(
                '<b>Count:</b> %{y:,}<br>'
                '<b>Percentage:</b> %{customdata[0]:.1f}%<extra></extra>'
            ),
            customdata=df_counts[['Percentage']].values
        )

    fig.update_layout(
        xaxis_title="Combination Types for Two Species Recordings",
        yaxis_title="Number of Records",
        margin=DEFAULT_CHART_MARGIN,
        height=DEFAULT_CHART_HEIGHT
    )

    return fig
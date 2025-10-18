import os
import sys
import re
import ast
import inspect
import json
from narwhals import col
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, time
from pandas.tseries.holiday import USFederalHolidayCalendar
from unidecode import unidecode
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from tabulate import tabulate
from itertools import combinations
from collections import defaultdict
import logging
from collections.abc import Iterable
from pprint import pprint
import numba
from concurrent.futures import ProcessPoolExecutor

global logger
from config import (
    logger,
    logger_transform,
    raw_data,
    intermediate_data,
    processed_data,
    casos_audit,
)

logging.getLogger().setLevel(logging.CRITICAL)
pd.options.display.max_columns = None
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 500)
pd.set_option('display.precision', 3)
pd.set_option('display.colheader_justify', 'center')
pd.options.mode.chained_assignment = None
# Set the pandas option to opt-in to future behavior
pd.set_option('future.no_silent_downcasting', True)
# Revert the pandas option to its default setting
# pd.reset_option('future.no_silent_downcasting')

# ---
# ---
# ---


def print_shapes():
    """
    Creates and returns a DataFrame with shapes of all pandas DataFrames
    in the calling namespace (e.g., a Jupyter notebook).

    Returns:
        pd.DataFrame: A DataFrame with columns ['DataFrame', 'Rows', 'Columns', 'Memory (MB)', 'NUMBER_UNIQUES']
    """
    # Get the calling frame (the Jupyter cell that called this function)
    frame = inspect.currentframe().f_back

    # Get the globals and locals from the calling frame
    calling_namespace = {**frame.f_globals, **frame.f_locals}

    # Initialize lists to store data
    df_names = []
    rows = []
    columns = []
    memory_usage = []
    number_uniques = []

    # Important ticket dataframes
    important_dfs = [
        'df_incidents',
        'df_problem',
        'df_problem_tasks',
        'df_created_incidents',
        'df_resolved_incidents',
        'df_problems_incidents',
        'df_incidents_grouped',
    ]

    # ServiceNow vs PowerBI mappings
    servicenow_dfs = [
        'df_incidents',
        'df_problem',
        'df_problem_tasks',
        'df_incidents_grouped',
    ]
    powerbi_dfs = [
        'df_created_incidents',
        'df_resolved_incidents',
        'df_problems_incidents',
    ]

    # Relationship mapping between dataframes
    df_relationships = {
        'df_incidents': {'key': 'NUMBER', 'related': {'df_problem': 'PROBLEM'}},
        'df_problem': {
            'key': 'NUMBER',
            'related': {
                'df_incidents': 'CREATED_OUT_OF_INCIDENT',
                'df_problem_tasks': 'PROBLEM',
            },
        },
        'df_problem_tasks': {'key': 'NUMBER', 'related': {'df_problem': 'PROBLEM'}},
        'df_created_incidents': {'key': 'NUMBER', 'related': {'df_problem': 'PROBLEM'}},
        'df_resolved_incidents': {
            'key': 'NUMBER',
            'related': {'df_problem': 'PROBLEM'},
        },
        'df_problems_incidents': {
            'key': 'NUMBER',
            'related': {'df_incidents': 'CREATED_OUT_OF_INCIDENT'},
        },
        'df_incidents_grouped': {'key': 'NUMBER', 'related': {'df_problem': 'PROBLEM'}},
    }

    # Find all DataFrames in the namespace
    for name, obj in calling_namespace.items():
        # Only include objects that start with 'df' or 'df_' and are pandas DataFrames
        if isinstance(obj, pd.DataFrame) and (name.startswith('df') or name == 'df'):
            # Ignore shapes_df_last
            if name == "shapes_df_last":
                continue

            df_names.append(name)
            rows.append(obj.shape[0])
            columns.append(obj.shape[1])

            # Calculate memory usage in MB
            memory_mb = obj.memory_usage(deep=True).sum() / (1024 * 1024)
            memory_usage.append(round(memory_mb, 2))

            # Calculate NUMBER_UNIQUES if 'NUMBER' column exists
            if 'NUMBER' in obj.columns:
                number_uniques.append(obj['NUMBER'].nunique())
            else:
                number_uniques.append(pd.NA)

    # Create the DataFrame
    if df_names:
        shapes_df = pd.DataFrame(
            {
                'DataFrame': df_names,
                'Rows': rows,
                'Columns': columns,
                'Memory (MB)': memory_usage,
                'NUMBER_UNIQUES': number_uniques,
            }
        )

        # Sort by name
        shapes_df = shapes_df.sort_values('DataFrame').reset_index(drop=True)

        # Print summary
        print(f"Found {len(shapes_df)} DataFrames in the current namespace:")

        # Calculate total memory usage
        total_memory = shapes_df['Memory (MB)'].sum()
        print(f"Total memory usage: {total_memory:.2f} MB")

        def compare_last_shapes(current_df, last_df):
            """Compare current DataFrame shapes with the last recorded shapes"""
            # Create a copy to avoid modifying the input
            result_df = current_df.copy()

            # Remove any existing change columns
            for col in result_df.columns:
                if col.endswith('_change') or col.endswith('_last'):
                    result_df = result_df.drop(columns=[col])

            # Get the set of DataFrames that exist in both current and last
            common_dfs = set(result_df['DataFrame']).intersection(
                set(last_df['DataFrame'])
            )

            # Add change columns
            result_df['Rows_last'] = pd.NA
            result_df['Columns_last'] = pd.NA
            result_df['Memory (MB)_last'] = pd.NA
            result_df['NUMBER_UNIQUES_last'] = pd.NA
            result_df['Rows_change'] = pd.NA
            result_df['Columns_change'] = pd.NA
            result_df['Memory (MB)_change'] = pd.NA
            result_df['NUMBER_UNIQUES_change'] = pd.NA

            # Update values for common DataFrames
            for df_name in common_dfs:
                current_row = result_df[result_df['DataFrame'] == df_name].index[0]
                last_row = last_df[last_df['DataFrame'] == df_name].index[0]

                for col in ['Rows', 'Columns', 'Memory (MB)', 'NUMBER_UNIQUES']:
                    # Skip if column doesn't exist in either dataframe
                    if col not in result_df.columns or col not in last_df.columns:
                        continue

                    last_val = last_df.loc[last_row, col]
                    current_val = result_df.loc[current_row, col]

                    # Skip if either value is NA
                    if pd.isna(last_val) or pd.isna(current_val):
                        continue

                    result_df.loc[current_row, f'{col}_last'] = last_val
                    result_df.loc[current_row, f'{col}_change'] = current_val - last_val

            return result_df

        # Check if shapes_df_last exists in the global namespace
        if 'shapes_df_last' in calling_namespace:
            shapes_df = compare_last_shapes(
                shapes_df, calling_namespace['shapes_df_last']
            )

        # Update shapes_df_last in the caller's namespace
        frame.f_globals['shapes_df_last'] = shapes_df.copy()

        shapes_df = shapes_df.sort_values(
            by='Memory (MB)', ascending=False
        ).reset_index(drop=True)
        tabulate_df(shapes_df, None)
        display(shapes_df)
        return shapes_df
    else:
        print("No DataFrames found in the current namespace.")
        return pd.DataFrame(
            columns=['DataFrame', 'Rows', 'Columns', 'Memory (MB)', 'NUMBER_UNIQUES']
        )


def create_structured_relationship_map(dataframes_dict):
    """
    Creates a semantically aware relationship map based on dataframe metadata
    with special handling for tasks and other specific entity types

    Args:
        dataframes_dict: Dictionary with dataframe metadata

    Returns:
        Nested dictionary of relationships organized by source type and dataframe
    """
    # Initialize the structured output
    relationship_map = {
        'cross_source': {},  # ServiceNow to PowerBI and vice versa
        'within_source': {},  # ServiceNow to ServiceNow, PowerBI to PowerBI
    }

    # Initialize nested structure for each dataframe
    for category in ['cross_source', 'within_source']:
        for df_name in dataframes_dict:
            relationship_map[category][df_name] = {}

    # Classify dataframes by type based on their name
    df_types = {}
    for df_name in dataframes_dict:
        if 'task' in df_name.lower():
            df_types[df_name] = 'task'
        elif 'problem' in df_name.lower():
            df_types[df_name] = 'problem'
        elif 'incident' in df_name.lower():
            df_types[df_name] = 'incident'
        else:
            df_types[df_name] = 'other'

    # Generate semantically valid relationships
    for from_df_name, from_metadata in dataframes_dict.items():
        from_db = from_metadata['db']
        from_pk = from_metadata['pk']
        from_fk = from_metadata['fk']
        from_type = df_types[from_df_name]

        for to_df_name, to_metadata in dataframes_dict.items():
            if from_df_name == to_df_name:
                continue  # Skip self-relationships

            to_db = to_metadata['db']
            to_pk = to_metadata['pk']
            to_fk = to_metadata['fk']
            to_type = df_types[to_df_name]

            is_cross_source = from_db != to_db
            category = 'cross_source' if is_cross_source else 'within_source'

            # RULE 1: Same entity type primary keys can link across databases
            # (But tasks are a special case - their primary keys don't link to anything)
            if from_type == to_type and from_type != 'task':
                add_relationship(
                    relationship_map,
                    category,
                    from_df_name,
                    from_pk,
                    to_df_name,
                    to_pk,
                    from_db,
                    to_db,
                    f"Same entity type ({from_type}) primary key relationship",
                )

            # RULE 2: Problem references - PROBLEM column should link to problem table NUMBER
            if from_fk == 'PROBLEM' and to_type == 'problem':
                add_relationship(
                    relationship_map,
                    category,
                    from_df_name,
                    from_fk,
                    to_df_name,
                    to_pk,
                    from_db,
                    to_db,
                    "Problem reference relationship",
                )

            # RULE 3: Created from incident - CREATED_OUT_OF_INCIDENT should link to incident table NUMBER
            if from_fk == 'CREATED_OUT_OF_INCIDENT' and to_type == 'incident':
                add_relationship(
                    relationship_map,
                    category,
                    from_df_name,
                    from_fk,
                    to_df_name,
                    to_pk,
                    from_db,
                    to_db,
                    "Created from incident relationship",
                )

            # RULE 4: Same foreign key types can relate across tables
            # (e.g., PROBLEM to PROBLEM across incident tables)
            if from_fk == to_fk:
                # But only if they're semantically the same kind of reference
                # (don't link PROBLEM to CREATED_OUT_OF_INCIDENT)
                add_relationship(
                    relationship_map,
                    category,
                    from_df_name,
                    from_fk,
                    to_df_name,
                    to_fk,
                    from_db,
                    to_db,
                    f"Related through common {from_fk} references",
                )

    return relationship_map


def add_relationship(
    relationship_map,
    category,
    from_df,
    from_col,
    to_df,
    to_col,
    from_db,
    to_db,
    semantic,
):
    """Helper function to add a relationship to the map"""
    relation_key = f"{to_df}.{to_col}"

    # Define the relationship details
    relationship = {
        'from_df': from_df,
        'from_col': from_col,
        'to_df': to_df,
        'to_col': to_col,
        'from_db': from_db,
        'to_db': to_db,
        'semantic': semantic,
    }

    # Add to the structured map
    if from_col not in relationship_map[category][from_df]:
        relationship_map[category][from_df][from_col] = {}

    relationship_map[category][from_df][from_col][relation_key] = relationship


def get_flat_relationships(relationship_map):
    """
    Flattens the nested relationship map to a list of relationship dictionaries.
    """
    flat_relationships = []

    for category in ['cross_source', 'within_source']:
        for from_df, cols in relationship_map[category].items():
            for from_col, relations in cols.items():
                for relation_key, details in relations.items():
                    flat_relationships.append(
                        {
                            'category': category,
                            'from_df': from_df,
                            'from_col': from_col,
                            'relation_key': relation_key,
                            **details,
                        }
                    )

    return flat_relationships


def create_table_comparisons(flat_relationships, dataframes):
    """
    Analyzes the actual data in dataframes based on identified relationships.
    Compares unique values and provides overlap metrics.
    """
    comparison_results = []

    seen_pairs = set()
    for rel in flat_relationships:
        from_df_name = rel['from_df']
        to_df_name = rel['to_df']
        from_col = rel['from_col']
        to_col = rel['to_col']

        # Skip if any dataframe is missing
        if from_df_name not in dataframes or to_df_name not in dataframes:
            continue

        from_df = dataframes[from_df_name]
        to_df = dataframes[to_df_name]

        # Skip if any column is missing
        if from_col not in from_df.columns or to_col not in to_df.columns:
            continue

        # Normalize key so (A,B) == (B,A)
        pair_key = tuple(sorted([f"{from_df_name}.{from_col}", f"{to_df_name}.{to_col}"]))

        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # Get unique values from both columns
        from_values = set(from_df[from_col].dropna().unique())
        to_values = set(to_df[to_col].dropna().unique())

        # Calculate comparison metrics
        intersection = from_values.intersection(to_values)
        from_only = from_values - to_values
        to_only = to_values - from_values

        # Calculate percentages
        from_count = len(from_values)
        to_count = len(to_values)
        intersection_count = len(intersection)

        # Calculate overlap percentages
        if from_count > 0:
            from_overlap_pct = (intersection_count / from_count) * 100
        else:
            from_overlap_pct = 0

        if to_count > 0:
            to_overlap_pct = (intersection_count / to_count) * 100
        else:
            to_overlap_pct = 0

        # Create result dictionary
        result = {
            'category': rel['category'],
            'from_df': from_df_name,
            'from_col': from_col,
            'to_df': to_df_name,
            'to_col': to_col,
            'from_db': rel['from_db'],
            'to_db': rel['to_db'],
            'semantic': rel['semantic'],
            'from_unique_count': from_count,
            'to_unique_count': to_count,
            'intersection_count': intersection_count,
            'from_only_count': len(from_only),
            'to_only_count': len(to_only),
            'from_overlap_pct': round(from_overlap_pct, 2),
            'to_overlap_pct': round(to_overlap_pct, 2),
            'overall_match_score': round((from_overlap_pct + to_overlap_pct) / 2, 2),
            # Include sample values
            'sample_common_values': list(intersection)[:5] if intersection else [],
            'sample_from_only': list(from_only)[:5] if from_only else [],
            'sample_to_only': list(to_only)[:5] if to_only else [],
        }

        comparison_results.append(result)

    # Convert to DataFrame for easy analysis
    comparisons_df = pd.DataFrame(comparison_results)

    # Sort by overall match score in descending order
    if not comparisons_df.empty:
        comparisons_df = comparisons_df.sort_values(
            'overall_match_score', ascending=False
        )

    comparisons_df = comparisons_df.reset_index(drop=True)
    tabulate_df(comparisons_df, None)
    display(comparisons_df)
    return comparisons_df


def get_best_relationships(comparisons_df, min_score=50, min_intersection=1):
    """
    Filters the comparison results to find the best relationships
    """
    if comparisons_df.empty:
        return pd.DataFrame()

    mask = (comparisons_df['overall_match_score'] >= min_score) & (
        comparisons_df['intersection_count'] >= min_intersection
    )

    return comparisons_df[mask].copy()


def print_valid_relationships(flat_relationships):
    """
    Prints all valid relationships identified, grouped by source dataframe
    """
    # Group by source dataframe
    by_source = {}
    for rel in flat_relationships:
        from_df = rel['from_df']
        if from_df not in by_source:
            by_source[from_df] = []
        by_source[from_df].append(rel)

    # Print grouped relationships
    for df_name, relations in by_source.items():
        print(f"\nRelationships from {df_name}:")
        for rel in relations:
            print(
                f"  {rel['from_col']} → {rel['to_df']}.{rel['to_col']} ({rel['semantic']})"
            )


# ---


def read_file_to_dataframe(path, file_name, delimiter=',', header=0):
    try:
        # Combine the path and file name to get the full file path
        full_path = os.path.join(path, file_name)

        # Get the file extension
        _, file_extension = os.path.splitext(full_path)

        # Based on the file extension, use the appropriate pandas function to read the file
        if file_extension in ['.csv', '.txt']:
            df = pd.read_csv(
                full_path,
                delimiter=delimiter,
                engine='pyarrow',
                dtype_backend='pyarrow',
                header=header,
            )
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(full_path, dtype_backend='pyarrow', header=header)
        elif file_extension in ['.json']:
            df = pd.read_json(full_path, dtype_backend='pyarrow')
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        return df
    except Exception as e:
        function_name = "read_file_to_dataframe"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def log_separator(title=None, level=1, logger=logger):
    try:
        if level == 1:
            separator_line = "=" * 100
        elif level == 2:
            separator_line = "-" * 50
        elif level == 3:
            separator_line = "-" * 30
        else:
            separator_line = "-" * 10  # Default to secondary if level is unrecognized

        if title:
            logger.info(f"\n\n{separator_line}\n    {title}\n{separator_line}\n")
        else:
            logger.info(separator_line)
    except Exception as e:
        function_name = "log_separator"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def log_clean_str_from_df(array: pd.Series) -> pd.Series:
    try:
        array = array.map(
            lambda x: unidecode(str(x)).strip() if isinstance(x, str) else x
        )
        # if string contains \n, replace with space
        array = array.replace(r'\n', ' ', regex=True)
        # if len of string > 70, truncate to 70 chars
        array = array.map(
            lambda x: x[:70] + '...' if isinstance(x, str) and len(x) > 70 else x
        )
        return array
    except Exception as e:
        function_name = "log_clean_str_from_df"
        error_message = f"[ERROR]: {function_name} {e}"
        print(error_message)
        logger.error(error_message)
        return None


def tabulate_df(df, num_rows=10, logger=logger):
    try:
        if df.empty:
            logger.warning("DataFrame is empty.")
            return

        if not num_rows:
            # no limit
            logger.warning(
                f"Shape: {df.shape}\n"
                + tabulate(
                    df.pipe(log_clean_str_from_df), headers='keys', tablefmt='psql'
                )
            )
            return
        if len(df) > (num_rows * 2):
            # Concatenate head and tail
            head_and_tail = pd.concat([df.head(num_rows), df.tail(num_rows)])
            logger.warning(
                f"Original shape: {df.shape}. Head and Tail:\n"
                + tabulate(
                    head_and_tail.pipe(log_clean_str_from_df),
                    headers='keys',
                    tablefmt='psql',
                )
            )
        else:
            logger.warning(
                f"Shape: {df.shape}\n"
                + tabulate(
                    df.pipe(log_clean_str_from_df), headers='keys', tablefmt='psql'
                )
            )
            return
    except Exception as e:
        function_name = "tabulate_df"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def memoria_usada(df, logger=logger):
    try:
        logger.info(f"shape: {df.shape}")
        logger.info(f"memoria usada: {(df.memory_usage(deep=True).sum()):,.0f}")
    except Exception as e:
        function_name = "memoria_usada"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def int_printer(n, logger=logger):
    try:
        return str(f"{(n):,.0f}")
    except Exception as e:
        function_name = "int_printer"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def remove_accents(input_str):
    try:
        if isinstance(input_str, str):
            return unidecode(input_str)
        return input_str
    except Exception as e:
        function_name = "remove_accents"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def clean_column_name(col_name, logger=logger):
    try:
        col_name = remove_accents(col_name)
        col_name = col_name.upper()
        col_name = col_name.replace(' ', '_')
        col_name = re.sub(r'[^A-Z0-9_]', '', col_name)
        col_name = re.sub(r'_+', '_', col_name)
        return col_name
    except Exception as e:
        function_name = "clean_column_name"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def clean_headers(df, logger=logger):
    try:
        df.columns = [clean_column_name(col) for col in df.columns]
        logger.info(
            f"Date columns: {df.select_dtypes(include=['datetime']).columns.tolist()}"
        )
        logger.info(
            f"Other columns: {df.select_dtypes(exclude=['datetime']).columns.tolist()}"
        )
        return df
    except Exception as e:
        function_name = "clean_headers"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def create_dt_time_wvariants(df_original, logger=logger):
    try:
        df = df_original.copy()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        for date_col in date_cols:
            # ignore updated
            if date_col in ['UPDATED', 'OUTAGE_BEGIN', 'OUTAGE_END']:
                continue
            if is_datetime(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df[f"_{date_col}_year"] = df[date_col].dt.year
                df[f"_{date_col}_month"] = df[date_col].dt.month
                df[f"_{date_col}_period"] = df[date_col].dt.to_period('M')
                df[f"_{date_col}_day_name"] = df[date_col].dt.day_name()
                df[f"_{date_col}_isnull"] = df[date_col].isnull()
                df[f"_{date_col}_str"] = df[date_col].dt.strftime('%Y-%m-%d %H:%M')
                logger.info(f"Created datetime variants for columns: {date_col}")
        return df
    except Exception as e:
        function_name = "create_dt_time_wvariants"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def dates_diff_tuples(df, date_pairs, comparison_date='2025-09-01', logger=logger):
    try:
        # Ensure the comparison_date is a datetime object
        comparison_date = pd.to_datetime(comparison_date)

        for col1, col2 in date_pairs:
            # Convert the first column to datetime if it is not already
            df[col1] = pd.to_datetime(df[col1], errors='coerce')

            if col2 == 'TODAY':
                # Calculate the difference in days between col1 and the comparison_date
                diff_col_name = f"_{col1}_diff_{col2}"
                df[diff_col_name] = (comparison_date - df[col1]).dt.days
            else:
                # If col2 is another column, convert it to datetime and calculate the difference
                df[col2] = pd.to_datetime(df[col2], errors='coerce')
                diff_col_name = f"_{col1}_diff_{col2}"
                df[diff_col_name] = (df[col2] - df[col1]).dt.days
        logger.info(f"Calculated date differences for pairs: {col1} to {col2}")
        return df
    except Exception as e:
        function_name = "dates_diff_tuples"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def calculate_working_days(
    df, created_date_col, versus_date, comparison_date, logger=logger
):
    try:
        comparison_date = pd.to_datetime(comparison_date).normalize()

        # Ensure the date columns are in datetime format
        df[created_date_col] = pd.to_datetime(df[created_date_col], errors='coerce')
        df[versus_date] = pd.to_datetime(df[versus_date], errors='coerce')

        # Calculate working days where 'versus_date' is NaT
        mask = df[versus_date].isna()
        start_dates = df.loc[mask, created_date_col].dt.to_period('D').dt.to_timestamp()
        end_dates = pd.Series(comparison_date).dt.to_period('D').dt.to_timestamp()

        # Use np.busday_count for vectorized calculation
        df.loc[mask, f'NOT_{versus_date}_WD_{comparison_date.date()}'] = (
            np.busday_count(
                start_dates.values.astype('datetime64[D]'),
                end_dates.values.astype('datetime64[D]'),
            )
        )

        # Set the column to None where 'versus_date' is not NaT
        df.loc[~mask, f'NOT_{versus_date}_WD_{comparison_date.date()}'] = None

        logger.info(
            f"Calculated working days between {created_date_col} and {comparison_date.date()} if {versus_date} is NaT"
        )
        return df
    except Exception as e:
        function_name = "calculate_working_days"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def transform_df_to_dict_dynamic(df, logger=logger):
    try:
        result_dict = {}

        for _, row in df.iterrows():
            # Extract the principal key
            principal_key = row['NAME_DF']

            # Initialize the dictionary for the principal key
            value_dict = {}

            # Iterate over the columns dynamically, excluding 'NAME_DF'
            for column_name in df.columns:
                if column_name != 'NAME_DF':
                    value_dict[column_name] = row[column_name]

            # Assign the constructed dictionary to the principal key
            result_dict[principal_key] = value_dict

        return result_dict
    except Exception as e:
        function_name = "transform_df_to_dict_dynamic"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def compare_cols_dicts(dict1, dict2, message=None, logger=logger):
    try:
        comparison_results = []

        for df_name in dict1.keys():
            if df_name in dict2:
                # Get the original and transformed columns
                original_cols = set(dict1[df_name])
                transformed_cols = set(dict2[df_name])

                # Determine dropped and new columns
                dropped_columns = list(original_cols - transformed_cols)
                new_columns = list(transformed_cols - original_cols)

                # Append the result to the list
                comparison_results.append(
                    {
                        "NAME_DF": df_name,
                        "DROPED_COLUMNS": dropped_columns,
                        "NEW_COLUMNS": new_columns,
                    }
                )

        df = pd.DataFrame(comparison_results)
        logger.info(f"Comparison of original vs transformed columns:\n{message}")
        tabulate_df(df)
        dict_transform = transform_df_to_dict_dynamic(df)
        return dict_transform
    except Exception as e:
        function_name = "compare_cols"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def compare_shapes_dicts(dict1, dict2, message=None, logger=logger):
    try:
        comparison_results = []

        for df_name in dict1.keys():
            if df_name in dict2:
                # Get the original and transformed shapes
                original_shape = dict1[df_name]
                transformed_shape = dict2[df_name]

                # Calculate the change in the number of rows
                row_change = transformed_shape[0] - original_shape[0]

                # Determine the change description
                if row_change > 0:
                    shape_changed = f"+{row_change} rows"
                elif row_change < 0:
                    shape_changed = f"{row_change} rows"
                else:
                    shape_changed = "0 rows"

                # Append the result to the list
                comparison_results.append(
                    {
                        "NAME_DF": df_name,
                        "ORIGINAL_SHAPE": original_shape,
                        "TRANSFORMED_SHAPE": transformed_shape,
                        "SHAPE_CHANGED": shape_changed,
                    }
                )

        df = pd.DataFrame(comparison_results)
        logger.info(f"Comparison of original vs transformed shapes:\n{message}")
        tabulate_df(df)
        dict_transform = transform_df_to_dict_dynamic(df)
        return dict_transform
    except Exception as e:
        function_name = "compare_shapes_dicts"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def find_ticket_number(df, index_col: str, linked_col: str, value: str, logger=None):
    try:
        df = df.copy()

        # Define a function to check if a cell contains the value
        def contains_value(cell):
            if isinstance(cell, str):
                return re.search(re.escape(value), cell) is not None
            elif isinstance(cell, (list, tuple, set)):
                return any(
                    re.search(re.escape(value), str(item)) is not None for item in cell
                )
            elif isinstance(cell, dict):
                return any(
                    re.search(re.escape(value), str(item)) is not None
                    for item in cell.values()
                )
            return False

        # Select columns of types 'object', 'string', and 'category', ignoring index_col and linked_col
        columns_to_check = df.select_dtypes(
            include=['object', 'string', 'category']
        ).columns.difference([index_col, linked_col])

        # Create a mask for string columns using str.contains
        string_mask = df[columns_to_check].apply(
            lambda col: col.astype(str).str.contains(re.escape(value), na=False)
        )

        # Create a mask for nested structures
        nested_mask = df[columns_to_check].applymap(contains_value)

        # Combine the masks
        match_matrix = string_mask | nested_mask

        # Create a mask for filtering rows based on the specified columns
        mask = (
            (df[linked_col] == value)
            | (df[index_col] == value)
            | match_matrix.any(axis=1)
        )

        # Create a new column with the list of matching column names
        df['match_columns'] = match_matrix.apply(
            lambda row: list(columns_to_check[row]), axis=1
        )

        return df[mask].sort_values(by=['CREATED', 'UPDATED']).reset_index(drop=True)
    except Exception as e:
        function_name = "find_ticket_number"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        if logger:
            logger.error(message)
        raise e


def flatten_cell(cell, logger=logger):
    """Flatten lists, sets, and dictionaries into a string.
    df.map(flatten_cell)
    """
    try:
        if isinstance(cell, (list, set)):
            return ','.join(map(str, cell))
        elif isinstance(cell, dict):
            return ','.join(map(str, cell.values()))
        return str(cell)  # Ensure the cell is converted to a string
    except Exception as e:
        function_name = "flatten_cell"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def flatten_column(df, column_name, logger=logger):
    """Flatten the specified column in the DataFrame."""
    try:
        df = df.copy()

        if column_name in df.columns:
            # Use applymap for more efficient element-wise operation
            df[column_name] = df[column_name].apply(flatten_cell)
        else:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        return df
    except Exception as e:
        function_name = "flatten_column"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def extend_df_with_pattern_matches(df, patterns, filter=False, logger=logger):
    try:
        df = df.copy()

        def clean_specialchar(pattern):
            pattern = re.sub(
                r'\W+', '_', pattern
            )  # Replace non-alphanumeric characters with underscores
            return pattern

        # Iterate over each pattern
        for pattern in patterns:
            # Use a vectorized approach to check for pattern matches across all columns
            df[f'_{clean_specialchar(pattern)}'] = df.apply(
                lambda row: any(re.search(pattern, str(value)) for value in row), axis=1
            )

        if filter:
            df = df[
                df[[f'_{clean_specialchar(pattern)}' for pattern in patterns]].any(
                    axis=1
                )
            ]

        return df
    except Exception as e:
        function_name = "extend_df_with_pattern_matches"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def set_df_column_order(
    df,
    start_cols: list,
    after_start_cols: list = None,
    end_columns: list = None,
    date_columns=True,
    logger=logger,
):
    try:
        all_columns = df.columns.tolist()
        after_start_cols = (
            [col for col in after_start_cols if col in all_columns]
            if after_start_cols
            else []
        )
        start_cols = (
            [col for col in start_cols if col in all_columns] if start_cols else []
        )
        end_columns = (
            [col for col in end_columns if col in all_columns] if end_columns else []
        )

        middle_columns = [
            col
            for col in all_columns
            if col not in after_start_cols + start_cols + end_columns
        ]

        date_cols = [
            col
            for col in middle_columns
            if pd.api.types.is_datetime64_any_dtype(df[col])
        ]
        non_date_cols = [col for col in middle_columns if col not in date_cols]

        if date_columns:
            ordered_columns = (
                start_cols + after_start_cols + date_cols + non_date_cols + end_columns
            )
        else:
            ordered_columns = (
                after_start_cols + start_cols + date_cols + non_date_cols + end_columns
            )

        # Reorder the DataFrame
        return (
            df[ordered_columns]
            .sort_values(by=date_cols, ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        function_name = "set_df_column_order"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def sample_dataframe(
    df: pd.DataFrame, seed: int = 42, num_rows: int = 2000, logger=logger
):
    try:
        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Ensure the number of rows requested does not exceed the DataFrame's size
        num_rows = min(num_rows, len(df))

        # Sample the DataFrame
        df_sample = df.sample(n=num_rows, random_state=seed)

        return df_sample
    except Exception as e:
        function_name = "sample_dataframe"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def calculate_unique_values(df, index_col, value_col, log=True, logger=logger):
    try:
        df_copy = df[[index_col, value_col]]
        grouped = (
            df_copy.groupby(index_col)[value_col]
            .agg(
                # nunique=lambda x: x.nunique(dropna=False),
                nunique='nunique',
                values=lambda x: ','.join(set(val for val in x if pd.notnull(val))),
                contains_null=lambda x: x.isnull().any(),
            )
            .sort_values(by='nunique', ascending=False)
            .reset_index()
            .rename(
                columns={
                    'nunique': f"_nunique_{value_col}",
                    'values': f"_values_{value_col}",
                    'contains_null': f"_contains_null_{value_col}",
                }
            )
        )
        # aggregate columns if nan inside a set in values
        if log:
            logger.info(
                f"'{index_col}' tickets are {df_copy[index_col].nunique()} related to {df_copy[value_col].nunique()} '{value_col}'s tickets"
            )
            tabulate_df(grouped)
        return grouped
    except Exception as e:
        function_name = "calculate_unique_values"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def log_results(df_value_counts, index_col, value_col, df_name='TABLE', logger=logger):
    try:
        nunique_counts_col = f"_nunique_{value_col}"
        if df_value_counts.empty:
            logger.info(
                f"No entries found for '{index_col}' vs '{value_col}' in {df_name}"
            )
        else:
            mask_nulls = (df_value_counts[nunique_counts_col] == 0) & (
                df_value_counts[f"_contains_null_{value_col}"] == True
            )
            mask_ones = df_value_counts[nunique_counts_col] == 1
            mask = mask_nulls | mask_ones
            logger.warning(
                f"Occurrences: {len(df_value_counts[~mask])} distinct to 1:1 relationships"
            )
            logger.info(
                f"Occurrences: {len(df_value_counts[mask])} with 1:1 relationships"
            )

            # df_value_counts = df_value_counts.query(f'{nunique_counts_col} != 1')
            if len(df_value_counts) > 20:
                log_large_results(df_value_counts, index_col, value_col, df_name)
            elif len(df_value_counts) == 0 or df_value_counts.empty:
                logger.info(
                    f"No distinct '{value_col}' tickets linked in '{df_name}' for '{index_col}' ticket."
                )
            else:
                logger.info(
                    f"For '{index_col}' ticket {df_value_counts.iloc[0, 0]}, there are {df_value_counts.iloc[0, 1]} distinct '{value_col}' tickets linked in '{df_name}'."
                )
                tabulate_df(df_value_counts)
    except Exception as e:
        function_name = "log_results"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def log_large_results(
    df_value_counts, index_col, value_col, df_name='TABLE', logger=logger
):
    try:
        list_values_col = f"_values_{value_col}"
        nunique_counts_col = f"_nunique_{value_col}"
        freq_table = (
            df_value_counts.drop(columns=list_values_col)
            .set_index(index_col)
            .value_counts(dropna=False)
            .reset_index(name='Frequency')
        )
        logger.warning(
            f"Grouping by '{index_col}' reveals multiple ticket linkages to '{value_col}' in '{df_name}'."
        )
        logger.warning(f"Frequency table generated with {len(freq_table)} entries.")

        def log_occurrence(
            freq_table_to_describe,
            df_value_counts,
            nunique_counts_col,
            index_col,
            list_values_col,
            value_col,
            df_name,
        ):
            try:
                for idx, row in freq_table_to_describe.iterrows():
                    occurrence_value = row.iloc[0]
                    contains_null = row.iloc[1]
                    occurrence_count = row.iloc[2]

                    # Filter the dataframe to find a ticket example
                    ticket = df_value_counts.loc[
                        lambda x: (x[nunique_counts_col] == occurrence_value)
                        & (x[f"_contains_null_{value_col}"] == contains_null)
                    ][index_col].iloc[0]

                    # Get linked tickets for the example ticket
                    _linked_tickets = (
                        df_value_counts.loc[
                            lambda x: (x[index_col] == ticket)
                            & (x[f"_contains_null_{value_col}"] == contains_null)
                        ][list_values_col]
                        .iloc[0]
                        .split(',')
                    )

                    # Limit the number of linked tickets displayed
                    if isinstance(_linked_tickets, list) and len(_linked_tickets) > 5:
                        _linked_tickets = _linked_tickets[:5] + ['...']
                    _linked_tickets = ','.join(map(str, _linked_tickets))

                    null_info = (
                        "contains null values"
                        if contains_null
                        else "does not contain null values"
                    )
                    logger.info(
                        f"Row {idx}: Occurrence {occurrence_value} - '{value_col}' tickets linked in '{df_name}', happening {occurrence_count} times, {null_info}."
                    )
                    logger.info(
                        f" --> Example: '{index_col}' ticket '{ticket}' has {occurrence_value} distinct '{value_col}' tickets: {_linked_tickets}."
                    )
            except Exception as e:
                function_name = "log_occurrence"
                message = f"[ERROR]: {function_name} {e}"
                print(message)
                logger.error(message)
                raise e

        if len(freq_table) > 10:
            freq_table_to_describe = pd.concat([freq_table.head(5), freq_table.tail(5)])
        else:
            freq_table_to_describe = freq_table
        log_occurrence(
            freq_table_to_describe,
            df_value_counts,
            nunique_counts_col,
            index_col,
            list_values_col,
            value_col,
            df_name,
        )

        logger.info(
            f"Displaying frequency (of value counts) for '{df_name}' {index_col}->{value_col}:"
        )
        tabulate_df(freq_table)
        logger.info(
            f"Displaying value counts for '{df_name}' {index_col}->{value_col}:"
        )
        tabulate_df(df_value_counts)

    except Exception as e:
        function_name = "log_large_results"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def get_number_nuniques(
    df, index_col, value_col, df_name='TABLE', log=True, logger=logger
):
    try:
        df_value_counts = calculate_unique_values(df, index_col, value_col, log=False)
        if log:
            log_results(df_value_counts, index_col, value_col, df_name)
        return df_value_counts
    except Exception as e:
        function_name = "get_number_nuniques"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def handle_duplicates(df, df_name='TABLE', log=True, logger=logger):
    try:
        df = df.copy()
        # Identify duplicated rows
        duplicated_mask = df.duplicated(keep=False)
        # Create a DataFrame of duplicated cases
        df_duplicates = df[duplicated_mask]
        # Create a DataFrame without duplicates, keeping the first occurrence
        df_unique = df.drop_duplicates(keep='first')

        if log:
            logger.info(f"Found {df_duplicates.shape[0]} duplicated rows in {df_name}.")
        if log:
            logger.info(f"DataFrame reduced to {df_unique.shape[0]} unique rows.")
        if log:
            logger.info("Duplicated Cases:")
            tabulate_df(df_duplicates)

        return df_unique
    except Exception as e:
        function_name = "handle_duplicates"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def drop_columns(df, columns_to_drop, df_name="DataFrame", logger=logger):
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Determine which columns are present and which are missing
        existing_columns = [col for col in columns_to_drop if col in df_copy.columns]
        missing_columns = [col for col in columns_to_drop if col not in df_copy.columns]

        # Drop only the existing columns
        if existing_columns:
            df_copy = df_copy.drop(columns=existing_columns)
            logger.info(f"Columns dropped from {df_name}: {existing_columns}")
        else:
            logger.info(f"No columns to drop from {df_name}.")

        # Inform about missing columns
        if missing_columns:
            logger.warning(
                f"Columns not found in {df_name} and could not be dropped: {missing_columns}"
            )

        return df_copy
    except Exception as e:
        function_name = "drop_columns"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def reset_df(df):
    """Reiniciar numeracion de filas de un DataFrame
    --> Devuelve DF"""
    return df.reset_index(drop=True)


# IMPLEMENT IN FLOW
def expand_from_split(df, link_to_column, logger=logger):
    try:
        if link_to_column not in df.columns:
            raise ValueError(
                f"The column '{link_to_column}' does not exist in the DataFrame."
            )

        # If cells contain lists, convert them to comma-separated strings
        df[link_to_column] = df[link_to_column].apply(
            lambda x: ','.join(x) if isinstance(x, list) else x
        )

        # Split the column values into lists
        df[link_to_column] = df[link_to_column].str.split(',')

        # Explode the lists into separate rows
        expanded_df = df.explode(link_to_column)

        return expanded_df
    except Exception as e:
        function_name = "expand_from_split"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------
# FUNCIONES POR TRATAR - USAR - MEJORAR - LOGGEAR - WIP


def compararDescribir_listas(set1, set2, nombres=('A', 'B')):
    try:
        A = set(set1)
        B = set(set2)

        # Calculate various set operations
        symmetric_difference = A.symmetric_difference(B)
        difference_A_B = A - B
        difference_B_A = B - A
        intersection = A & B
        union = A | B

        # Print results with descriptive names
        print(f"Elementos en '{nombres[0]}': {len(A)}")
        print(f"Elementos en '{nombres[1]}': {len(B)}")
        print(f"Total elementos combinados: {len(A) + len(B)}")
        print(f"Elementos únicos combinados: {len(union)}")
        print(f"Diferencia en número de elementos: {len(symmetric_difference)}")
        print(f"- En '{nombres[0]}' pero no en '{nombres[1]}': {len(difference_A_B)}")
        print(f"- En '{nombres[1]}' pero no en '{nombres[0]}': {len(difference_B_A)}")
        print(f"Elementos diferentes: {symmetric_difference}")
        print(f"- En '{nombres[0]}' pero no en '{nombres[1]}': {len(difference_A_B)}")
        print(difference_A_B)
        print(f"- En '{nombres[1]}' pero no en '{nombres[0]}': {len(difference_B_A)}")
        print(difference_B_A)
        print(f'Elementos comunes en ambas: {len(intersection)}')
        print(intersection)
        print(
            f"Elementos únicos en ambas: {len(union)} = {len(symmetric_difference)} + {len(intersection)}"
        )
    except Exception as e:
        function_name = "compararDescribir_listas"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def from_array_get_list_values_col_match_pattern(array, patterns):
    try:
        # Compile all patterns
        compiled_patterns = [re.compile(pattern) for pattern in patterns]
        matches = []
        for item in array:
            for pattern in compiled_patterns:
                matches.extend(pattern.findall(str(item)))
        return matches
    except Exception as e:
        function_name = "from_array_get_list_values_col_match_pattern"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# try:
#     patterns = [
#             r'PRB\d{7}',   # Matches 'PRB' followed by 7 digits
#             r'INC\d{8}',   # Matches 'INC' followed by 8 digits
#             r'PTASK\d{7}', # Matches 'PTASK' followed by 7 digits
#             r'CHG\d{7}'    # Matches 'CHG' followed by 7 digits
#         ]
#     set1 = set(df.NUMBER.unique())
#     set2 = set(from_array_get_list_values_col_match_pattern(df.PERSONAL_NOTES, patterns))
#     compararDescribir_listas(set1, set2, nombres=('Incidentes', 'Notas Personales'))
# except Exception as e:
# function_name = ""
# print.error(f"[ERROR]: {function_name} {e}") # por editar

# para las fechas, diferencia de fechas, medias, sumas_dias_grupos, value_counts(dropna=False), col.isna().sum() resumen de abiertos
# df.pipe(dates_diff_tuples, date_pairs=time_cols_pairs['df_problem']) #.pipe(min_max, 'CREATED_to_UPDATED_days') #idea usar conficionales .loc[mask, col_to_find_min_max] # df[col].agg(("min", "max"))


# agregar a mis tablas resumen
def count_lengths(lista):
    try:
        # Initialize a defaultdict to store counts
        length_counts = defaultdict(int)

        for elem in lista:
            if elem:
                length = len(elem)
            else:
                length = None

            # Increment the count for the length or None
            length_counts[length] += 1

        return dict(length_counts)
    except Exception as e:
        function_name = "count_lengths"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def largos_valores_cols_df(df):
    try:
        for col in df.columns:
            # print(f"{(col):<15}", largos_listasLista(list(df[col].astype("string").fillna("").unique())))
            print(
                f"{(col):<15}",
                count_lengths(list(df[col].astype("string").fillna("").unique())),
            )
    except Exception as e:
        function_name = "largos_valores_cols_df"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# agregar a mis tablas resumen
def find_duplicates_print(df, subsets_comb=False, unique_id=""):
    try:
        print("TOTAL DUPLICATES", len(df[df.duplicated()]))
        if unique_id:
            marker = "   <======="
        print(
            f"{('DUPLI'):>10}", f"{('DUPLI_keep'):>10}", f"{('UNIQUE'):>10}", "COLUMN"
        )
        num_cols = len(df.columns)
        for col in df.columns:
            duplicates = len(df[df.duplicated(subset=col, keep=False)])
            duplicates_k = len(df[df.duplicated(subset=col)])
            uniques = len(df[col].unique())
            # nuniques = df[col].nunique() # same result
            if col == unique_id:
                print(
                    f"{(int_printer(duplicates)):>10}",
                    f"{(int_printer(duplicates_k)):>10}",
                    f"{(int_printer(uniques)):>10}",
                    col + marker,
                )
            else:
                print(
                    f"{(int_printer(duplicates)):>10}",
                    f"{(int_printer(duplicates_k)):>10}",
                    f"{(int_printer(uniques)):>10}",
                    col,
                )
        if subsets_comb:  # any char
            print()
            for r in range(2, num_cols + 1):
                for subset in combinations(df.columns, r):
                    duplicates = len(df[df.duplicated(subset=list(subset), keep=False)])
                    duplicates_k = len(df[df.duplicated(subset=list(subset))])
                    uniques = len(df[list(subset)].value_counts(dropna=False))
                    print(
                        f"{(int_printer(duplicates)):>10}",
                        f"{(int_printer(duplicates_k)):>10}",
                        f"{(int_printer(uniques)):>10}",
                        "//".join(subset),
                    )
    except Exception as e:
        function_name = "find_duplicates_print"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# .lt(0) # .gt(0)
# .filter(regex=r'^_')
# .pipe(lambda df_: df_[df_.lt(0).any(axis='columns')])
# (df[col_num].clip(upper=20, lower=10).value_counts()) #.min().max()

# def all_ceros(df):
#     return(df.select_dtypes(int)
#              .eq(0)
#              .all()
#              .pipe(lambda df_: df_[df_]==True)
#             ).index

# # make function
# def clean_housing(df):
#     return (df
#             .assign(**df.select_dtypes('string')
#                     .replace('', 'Missing')
#                     .replace(' ', 'Missing')
#                     .replace('NA', 'Missing').astype('category')
#                    )
#             .drop(columns=all_ceros(df))
#             .pipe(shrink_ints)
#             )

# clean_housing(df).dtypes

# def shrink_ints(df):
#     mapping = {}
#     for col in df.dtypes[df.dtypes=='int64[pyarrow]'].index:
#         max_ = df[col].max()
#         min_ = df[col].min()
#         if min_ < 0:
#             continue
#         if max_ < 255:
#             mapping[col] = 'uint8[pyarrow]'
#         elif max_ < 65_535:
#             mapping[col] = 'uint16[pyarrow]'
#         elif max_ <  4294967295:
#             mapping[col] = 'uint32[pyarrow]'
#     return df.astype(mapping)

# ------------------------------------------------------
# ------------------------------------------------------


def get_first_3_from_split(df, cols):
    df_copy = df.copy()
    for col in cols:
        df_copy[f'_REGION_{col}'] = df_copy[col].str.split('.').str[:3].str.join('.')
    return df_copy


def selected_cols_same_value(df, cols):
    df_copy = df.copy()
    df_copy['_SAME_GROUP'] = df_copy[cols].nunique(axis=1) == 1
    return df_copy


def any_of_cols_contain_ES(df, cols, contain='ES'):
    df_copy = df.copy()
    df_copy['_ANY_OF_GROUP_ES'] = df_copy[cols].apply(
        lambda x: x.str.contains(contain).any(), axis=1
    )
    return df_copy


def process_dataframe_INC(df):
    oe_col = 'AFFECTED_OES'
    oe_value = 'Spain'
    company_col = 'COMPANY'
    company_value = 'Compania de Seguros'

    logger.info(
        f'Processing DataFrame with filters: {oe_col} -> {oe_value}, {company_col} -> {company_value}'
    )

    oe_mask = df[oe_col].fillna('').str.contains(oe_value)
    mask_nulls = df[oe_col].isna()
    company_mask = df[company_col].fillna('').str.contains(company_value)

    logger.warning('Null values found in AFFECTED_OES column')
    tabulate_df(df[mask_nulls])
    casos_audit['null_affected_oes'] = df[mask_nulls]

    # Create additional columns based on masks
    df['_N_OEs'] = df[oe_col].fillna('').str.split(',').str.len()
    # df['_NULLS_OE'] = mask_nulls
    df['_OE_SPAIN'] = oe_mask
    df['_COMPANY_CDS'] = company_mask

    # Apply transformations to specific columns
    cols = ['LAST_ASSIGNMENT_GROUP', 'ASSIGNMENT_GROUP', 'CREATOR_GROUP']
    df = get_first_3_from_split(df, cols)
    # cols = [f'_REGION_{col}' for col in cols]
    # df = selected_cols_same_value(df, cols)
    # df = any_of_cols_contain_ES(df, cols, contain='ES')

    # Create final mask and filter DataFrame
    # group_mask = df['_ANY_OF_GROUP_ES']
    mask_final = oe_mask | company_mask  # | group_mask
    df['_SPAIN_ANY'] = mask_final
    df = df[mask_final]

    casos_audit['many_affected_oes'] = df[df['_N_OEs'] != 1]
    mask = df['_N_OEs'] == 1
    incidents = df.loc[mask, 'NUMBER'].unique()
    df = df[df['NUMBER'].isin(incidents)]

    return df


def set_pct_completed(df):
    try:
        df['_PCT_ROW_COMP'] = df.notnull().mean(axis=1) * 100
        return df
    except Exception as e:
        function_name = "set_pct_completed"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# ------------------------------------------------------
# ------------------------------------------------------


def load_CRE_INC():
    try:
        log_separator("CRE_INC.xlsx Source -> PowerBI", level=2)
        # Load and process CRE_INC.xlsx
        df = (
            pd.read_excel(raw_data + "//CRE_INC.xlsx")
            .pipe(clean_headers)
            .iloc[0:-2]
            .loc[lambda x: x['DATE_CREATED'] < pd.to_datetime('2025-09-01')]
            .rename(columns={"DATE_CREATED": "CREATED", "DATE_RESOLVED": "RESOLVED"})
            .loc[lambda x: x['CREATED'].dt.year > 2022]
        )
        logger.info(f"Loaded and processed CRE_INC.xlsx with shape {df.shape}")
        return df
    except Exception as e:
        function_name = "load_CRE_INC"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def load_RES_INC():
    try:
        log_separator("RES_INC.xlsx Source -> PowerBI", level=2)
        # Load and process RES_INC.xlsx
        logger.info("Loading and processing RES_INC.xlsx")
        df = (
            pd.read_excel(raw_data + "//RES_INC.xlsx")
            .pipe(clean_headers)
            .iloc[0:-2]
            .loc[lambda x: x['CREATED'] < pd.to_datetime('2025-09-01')]
            .loc[lambda x: x['CREATED'].dt.year > 2022]
        )
        logger.info(f"Loaded and processed RES_INC.xlsx with shape {df.shape}")
        return df
    except Exception as e:
        function_name = "load_RES_INC"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def load_PRO_INC():
    try:
        log_separator("PRO_INC.xlsx Source -> PowerBI", level=2)
        # Load and process PRO_INC.xlsx
        logger.info("Loading and processing PRO_INC.xlsx")
        df = (
            pd.read_excel(raw_data + "//PRO_INC.xlsx")
            .pipe(clean_headers)
            .iloc[0:-2]
            .loc[lambda x: x['CREATED'] < pd.to_datetime('2025-09-01')]
            .loc[lambda x: x['CREATED'].dt.year > 2022]
        )
        logger.info(f"Loaded and processed PRO_INC.xlsx with shape {df.shape}")
        return df
    except Exception as e:
        function_name = "load_PRO_INC"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def load_problem_task():
    try:
        log_separator("problem_task.xlsx Source -> ServiceNow", level=2)
        # Load and process problem_task.xlsx
        logger.info("Loading and processing problem_task.xlsx")
        df = (
            pd.read_excel(raw_data + "//problem_task.xlsx")
            .pipe(clean_headers)
            .loc[lambda x: x['CREATED'].dt.year > 2022]
            .loc[lambda x: x['CREATED'] < pd.to_datetime('2025-09-01')]
            .loc[lambda x: x['UPDATED'] < pd.to_datetime('2025-09-01')]
            .sort_values(
                by=['NUMBER', 'CREATED', 'UPDATED'], ascending=[True, True, False]
            )
            .pipe(set_df_column_order, ['NUMBER', 'PROBLEM'])
        )
        logger.info(f"Loaded and processed problem_task.xlsx with shape {df.shape}")
        logger.info(
            "Filters used: CREATED > 2022, CREATED < 2025-09-01, UPDATED < 2025-09-01"
        )
        tabulate_df(df)
        return df
    except Exception as e:
        function_name = "load_problem_task"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def load_problem():
    try:
        log_separator("problem.xlsx Source -> ServiceNow", level=2)
        # Load and process problem.xlsx
        logger.info("Loading and processing problem.xlsx")
        df = (
            pd.read_excel(raw_data + "//problem.xlsx")
            .pipe(clean_headers)
            .loc[lambda x: x['CREATED'].dt.year > 2022]
            .loc[lambda x: x['CREATED'] < pd.to_datetime('2025-09-01')]
            .loc[lambda x: x['UPDATED'] < pd.to_datetime('2025-09-01')]
            .sort_values(
                by=['NUMBER', 'CREATED', 'UPDATED'], ascending=[True, True, False]
            )
            .pipe(set_df_column_order, ['NUMBER', 'CREATED_OUT_OF_INCIDENT'])
        )
        logger.info(f"Loaded and processed problem.xlsx with shape {df.shape}")
        logger.info(
            "Filters used: CREATED > 2022, CREATED < 2025-09-01, UPDATED < 2025-09-01"
        )
        tabulate_df(df)
    except Exception as e:
        function_name = "load_problem"
        print(f"[ERROR]: {function_name} {e}")
        logger.error(f"[ERROR]: {function_name} {e}")
        raise
    return df


def load_Data_INC():
    function_name = "load_Data_INC"
    try:
        log_separator("Data_INC.xlsx Source -> ServiceNow", level=2)
        logger.info("Loading and processing Data_INC.xlsx")

        # Ruta al pickle y fallback a Excel si no existe
        pickle_path = os.path.join(intermediate_data, "Data_INC.pkl")
        if os.path.exists(pickle_path):
            logger.info(f"Loading Data_INC from pickle: {pickle_path}")
            df = pd.read_pickle(pickle_path)
        else:
            excel_path = os.path.join(raw_data, "Data_INC.xlsx")
            logger.info(f"Loading Data_INC from Excel: {excel_path}")
            df = pd.read_excel(excel_path)

        # Procesamiento en cadena (pandas pipe)
        df = (
            df.pipe(clean_headers)
            .loc[lambda x: x["CREATED"].dt.year > 2022]
            .loc[lambda x: x["CREATED"] < pd.to_datetime("2025-09-01")]
            .loc[lambda x: x["UPDATED"] < pd.to_datetime("2025-09-01")]
              # .loc[lambda x: x["AFFECTED_OES"] == "Allianz Spain"]
            .pipe(set_df_column_order, ["NUMBER", "PROBLEM", "STATE", "TAG"])
            .pipe(set_pct_completed)
            .sort_values(
                by=["NUMBER", "CREATED", "UPDATED", "_PCT_ROW_COMP"],
                ascending=[True, True, False, False],
            )
            .pipe(process_dataframe_INC)
        )

        logger.info(f"Loaded and processed Data_INC with shape {df.shape}")
        logger.info(
            "Filters applied: CREATED > 2022, CREATED < 2025-09-01, UPDATED < 2025-09-01"
        )
        tabulate_df(df)
        return df

    except Exception as e:
        message = f"[ERROR]: {function_name} {e}"
        logger.error(message)
        raise


def load_and_process_data(load_from_pickle=False):
    try:
        df_created_incidents = load_CRE_INC()
        df_resolved_incidents = load_RES_INC()
        df_problems_incidents = load_PRO_INC()
        df_problem_tasks = load_problem_task()
        df_problem = load_problem()
        # since load_Data_INC is heavy excel lets make a export to pickle and create a if that pickle exists load pickle if not load excel
        pickle_path = os.path.join(intermediate_data, "Data_INC.pkl")
        if os.path.exists(pickle_path) and load_from_pickle:
            logger.info("Loading Data_INC from pickle")
            df_incidents = pd.read_pickle(pickle_path)
            logger.info(
                f"Date columns: {df_incidents.select_dtypes(include=['datetime']).columns.tolist()}"
            )
            logger.info(
                f"Other columns: {df_incidents.select_dtypes(exclude=['datetime']).columns.tolist()}"
            )
            logger.info(
                f"Loaded and processed Data_INC.xlsx with shape {df_incidents.shape}"
            )
            logger.info(
                "Filters used: CREATED > 2022, CREATED < 2025-09-01, UPDATED < 2025-09-01"
            )
            tabulate_df(df_incidents)
        else:
            df_incidents = load_Data_INC()
            logger.info("Saving Data_INC to pickle")
            df_incidents.to_pickle(pickle_path)
        logger.info("All data loaded and processed successfully.")
        return (
            df_created_incidents,
            df_resolved_incidents,
            df_problems_incidents,
            df_problem_tasks,
            df_problem,
            # df_incidents,
        )
    except Exception as e:
        function_name = "load_and_process_data"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------


def drop_constant_columns(df):
    try:
        columns_to_drop = []

        for column in df.columns:
            col_data = df[column]

            # Check if the column contains dictionaries
            if col_data.apply(lambda x: isinstance(x, dict)).any():
                # Handle dictionary columns separately
                # For example, you might want to skip them or apply custom logic
                continue

            # Identify columns with a single unique value (including NaN)
            if col_data.nunique(dropna=False) <= 1:
                columns_to_drop.append(column)

        # Drop these columns
        df_dropped = df.drop(columns=columns_to_drop)

        return df_dropped
    except Exception as e:
        function_name = "drop_constant_columns"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        raise e


def transform_datetime_to_date(df):
    try:
        for column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                # Check if the column contains time information
                has_time_info = (
                    df[column]
                    .apply(
                        lambda x: (
                            x.time() != pd.Timestamp.min.time()
                            if pd.notnull(x)
                            else False
                        )
                    )
                    .any()
                )

                if has_time_info:
                    # Create a new column with date-only values
                    new_column_name = f'_DT_{column}'
                    df[new_column_name] = df[column].dt.date

        return df
    except Exception as e:
        function_name = "transform_datetime_to_date"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        raise e


def truncate_string(s, length=25):
    try:
        if isinstance(s, str):
            return s if len(s) <= length else s[:length] + '...'
        return s
    except Exception as e:
        function_name = "truncate_string"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def get_pct(value, total):
    try:
        pct = round(value / total * 100, 2) if total > 0 else 0.0
        if pct > 0:
            return f"{pct:<4.2f} %"
        else:
            return ''
    except Exception as e:
        function_name = "get_pct"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def analyze_dataframe_columns(df_original):
    try:
        df = df_original.copy()
        analysis_results = []
        df = transform_datetime_to_date(df)

        for column in df.columns:
            col_data = df[column]
            total_count = len(col_data)
            null_count = col_data.isnull().sum()
            value_count = total_count - null_count
            nunique_values = col_data.nunique(dropna=True)
            sample_nunique = col_data.value_counts(dropna=False).head(5).to_dict()
            data_type = str(col_data.dtype)

            # Initialize additional metrics
            zero_count = negative_count = blank_space_count = 0
            min_value = max_value = mean_value = median_value = mode_value = std_dev = (
                None
            )
            skewness = kurtosis = None

            if pd.api.types.is_numeric_dtype(col_data):
                zero_count = (col_data == 0).sum()
                negative_count = (col_data < 0).sum()
                min_value = col_data.min()
                max_value = col_data.max()
                mean_value = col_data.mean()
                median_value = col_data.median()
                mode_value = (
                    col_data.mode().iloc[0] if not col_data.mode().empty else None
                )
                std_dev = col_data.std()
                skewness = col_data.skew()
                kurtosis = col_data.kurt()
                # Mean: The average of all values. It is calculated by summing all the values and dividing by the number of values.
                # Median: The middle value when the data is sorted. If the number of values is even, the median is the average of the two middle numbers.
                # Mode: The value that appears most frequently in the data. There can be more than one mode if multiple values have the same highest frequency.
                # Standard Deviation (std_dev): A measure of the amount of variation or dispersion in a set of values. A low standard deviation means the values are close to the mean, while a high standard deviation means the values are spread out.
                # Skewness: A measure of the asymmetry of the distribution. A skewness of zero indicates a symmetrical distribution. Positive skewness indicates a distribution with an asymmetric tail extending towards more positive values, and negative skewness indicates a tail extending towards more negative values.
                # Kurtosis: A measure of the "tailedness" of the distribution. High kurtosis indicates heavy tails, while low kurtosis indicates light tails.

            if pd.api.types.is_object_dtype(col_data):
                blank_space_count = col_data.apply(
                    lambda x: isinstance(x, str) and x.strip() == ''
                ).sum()

            truncated_sample_nunique = {
                truncate_string(k): v for k, v in sample_nunique.items()
            }

            analysis_results.append(
                {
                    'COLUMN': column,
                    'DATA_TYPE': data_type,
                    'COUNT_NULLS': null_count,
                    'PCT_NULLS': get_pct(null_count, total_count),
                    'COUNT_VALUES': value_count,
                    'PCT_VALUES': get_pct(value_count, total_count),
                    'NUNIQUE_VALUES': nunique_values,
                    'PCT_NUNIQUES': get_pct(nunique_values, total_count),
                    'COUNT_CEROS': zero_count,
                    'PCT_CEROS': get_pct(zero_count, total_count),
                    'COUNT_NEGATIVES': negative_count,
                    'PCT_NEGATIVES': get_pct(negative_count, total_count),
                    'COUNT_BLANK_SPACES': blank_space_count,
                    'PCT_BLANK_SPACES': get_pct(blank_space_count, total_count),
                    'MIN_VALUE': min_value,
                    'MAX_VALUE': max_value,
                    'MEAN_VALUE': mean_value,
                    'MEDIAN_VALUE': median_value,
                    'MODE_VALUE': mode_value,
                    'STD_DEV': std_dev,
                    'SKEWNESS': skewness,
                    'KURTOSIS': kurtosis,
                    'TOP_SAMPLE_NUNIQUE': truncated_sample_nunique,
                }
            )

        df = (
            pd.DataFrame(analysis_results)
            #   .pipe(drop_constant_columns)
            .sort_values(by=['DATA_TYPE', 'COUNT_NULLS'], ascending=[True, False])
            #   .reset_index(drop=True)
        )

        mask = df['DATA_TYPE'].str.startswith(('float', 'int', 'bool'))
        df_no_numeric = df[~mask]
        df_numeric = df[mask]
        logger.info("DataFrame shape: %s", df_original.shape)
        tabulate_df(df_no_numeric.pipe(drop_constant_columns), num_rows=None)
        tabulate_df(df_numeric.pipe(drop_constant_columns), num_rows=None)

        return df
    except Exception as e:
        function_name = "analyze_dataframe_columns"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def analyze_dataframe_columns_export(df_original):
    df = df_original.copy()
    analysis_results = []
    df = transform_datetime_to_date(df)

    for column in df.columns:
        col_data = df[column]
        total_count = len(col_data)
        null_count = col_data.isnull().sum()
        value_count = total_count - null_count
        nunique_values = col_data.nunique(dropna=True)
        sample_nunique = col_data.value_counts(dropna=False).head(20).to_dict()
        data_type = str(col_data.dtype)
        len_col_data = col_data.astype(str).str.len().value_counts().to_dict()

        # Initialize additional metrics
        zero_count = negative_count = blank_space_count = 0
        min_value = max_value = mean_value = median_value = mode_value = std_dev = None
        skewness = kurtosis = None

        if pd.api.types.is_numeric_dtype(col_data):
            zero_count = (col_data == 0).sum()
            negative_count = (col_data < 0).sum()
            min_value = col_data.min()
            max_value = col_data.max()
            mean_value = col_data.mean()
            median_value = col_data.median()
            mode_value = col_data.mode().iloc[0] if not col_data.mode().empty else None
            std_dev = col_data.std()
            skewness = col_data.skew()
            kurtosis = col_data.kurt()
            # Mean: The average of all values. It is calculated by summing all the values and dividing by the number of values.
            # Median: The middle value when the data is sorted. If the number of values is even, the median is the average of the two middle numbers.
            # Mode: The value that appears most frequently in the data. There can be more than one mode if multiple values have the same highest frequency.
            # Standard Deviation (std_dev): A measure of the amount of variation or dispersion in a set of values. A low standard deviation means the values are close to the mean, while a high standard deviation means the values are spread out.
            # Skewness: A measure of the asymmetry of the distribution. A skewness of zero indicates a symmetrical distribution. Positive skewness indicates a distribution with an asymmetric tail extending towards more positive values, and negative skewness indicates a tail extending towards more negative values.
            # Kurtosis: A measure of the "tailedness" of the distribution. High kurtosis indicates heavy tails, while low kurtosis indicates light tails.

        if pd.api.types.is_object_dtype(col_data):
            blank_space_count = col_data.apply(
                lambda x: isinstance(x, str) and x.strip() == ''
            ).sum()

        # truncated_sample_nunique = {truncate_string(k): v for k, v in sample_nunique.items()}
        # truncated_sample_len = {truncate_string(k): v for k, v in len_col_data.items()}

        analysis_results.append(
            {
                'COLUMN': column,
                'DATA_TYPE': data_type,
                'COUNT_NULLS': null_count,
                'PCT_NULLS': get_pct(null_count, total_count),
                'COUNT_VALUES': value_count,
                'PCT_VALUES': get_pct(value_count, total_count),
                'NUNIQUE_VALUES': nunique_values,
                'PCT_NUNIQUES': get_pct(nunique_values, total_count),
                'COUNT_CEROS': zero_count,
                'PCT_CEROS': get_pct(zero_count, total_count),
                'COUNT_NEGATIVES': negative_count,
                'PCT_NEGATIVES': get_pct(negative_count, total_count),
                'COUNT_BLANK_SPACES': blank_space_count,
                'PCT_BLANK_SPACES': get_pct(blank_space_count, total_count),
                'MIN_VALUE': min_value,
                'MAX_VALUE': max_value,
                'MEAN_VALUE': mean_value,
                'MEDIAN_VALUE': median_value,
                'MODE_VALUE': mode_value,
                'STD_DEV': std_dev,
                'SKEWNESS': skewness,
                'KURTOSIS': kurtosis,
                'TOP_SAMPLE_NUNIQUE': sample_nunique,
                'LEN_COL_DATA': len_col_data,
            }
        )

    df = (
        pd.DataFrame(analysis_results)
        #   .pipe(drop_constant_columns)
        .sort_values(by=['DATA_TYPE', 'COUNT_NULLS'], ascending=[True, False])
        #   .reset_index(drop=True)
    )

    return df


def analyze_dataframe_rows(df_original, main_key, *args):
    try:
        df = df_original.copy()

        # Check for duplicates based on the main key
        key_duplicates = df[df.duplicated(subset=[main_key], keep=False)]
        if not key_duplicates.empty:
            logger.warning(
                f"Found {len(key_duplicates)} duplicate rows based on the key '{main_key}':"
            )
            tabulate_df(key_duplicates)
        else:
            logger.info(f"No duplicates found based on the key '{main_key}'.")

        # Check for duplicates based on the main key + args
        key_duplicates = df[df.duplicated(subset=[main_key, *args], keep=False)]
        mask = key_duplicates[list(args)].notnull().all(axis=1)
        key_duplicates = key_duplicates[mask]
        if not key_duplicates.empty:
            logger.warning(
                f"Found {len(key_duplicates)} duplicate rows based on the key '{main_key}' and args {args}:"
            )
            tabulate_df(key_duplicates)
        else:
            logger.info(
                f"No duplicates found based on the key '{main_key}' and args {args}."
            )

        # Check for completely duplicate rows
        key_duplicates = df[df.duplicated(keep=False)]
        if not key_duplicates.empty:
            logger.warning(f"Found {len(key_duplicates)} completely duplicate rows:")
            tabulate_df(key_duplicates)
        else:
            logger.info("No completely duplicate rows found.")

        # Calculate the frequency of non-null values in each row
        row_completion_percentage = df.notnull().mean(axis=1) * 100

        # Create a frequency table for row completion percentages
        frequency_table = row_completion_percentage.value_counts(normalize=True) * 100

        # Construct a DataFrame to tabulate the frequency table along with examples
        frequency_df = pd.DataFrame(
            {
                'Completion Percentage (%)': [
                    get_pct(value, 100) for value in frequency_table.index
                ],
                'Frequency (%)': [
                    get_pct(freq, 100) for freq in frequency_table.values
                ],
                'Example Keys': [
                    df.loc[row_completion_percentage == value, main_key]
                    .head(5)
                    .tolist()
                    for value in frequency_table.index
                ],
            }
        )

        # Sort the DataFrame by Completion Percentage
        frequency_df['Completion Percentage (%)'] = (
            frequency_df['Completion Percentage (%)']
            .str.extract(r'(\d+\.\d+)')
            .astype(float)
        )
        frequency_df.sort_values(
            by='Completion Percentage (%)', ascending=False, inplace=True
        )

        # Reformat the Completion Percentage column after sorting
        frequency_df['Completion Percentage (%)'] = frequency_df[
            'Completion Percentage (%)'
        ].apply(lambda x: get_pct(x, 100))

        logger.info("Rows completion percentage DataFrame:")
        tabulate_df(frequency_df, num_rows=None)

    except Exception as e:
        function_name = "analyze_dataframe_rows"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------


# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------


def dtypes_dict_to_df(dtype_dictionary):
    try:
        # Convert the dictionary to a DataFrame
        recommendations_df = pd.DataFrame.from_dict(
            dtype_dictionary,
            orient='index',
            columns=['Recommended Type', 'Sample Unique Values 5'],
        )

        # Reset the index and rename the column for clarity
        recommendations_df = recommendations_df.reset_index().rename(
            columns={"index": "COLUMNS"}
        )

        tabulate_df(recommendations_df, num_rows=None, logger=logger_transform)
        return recommendations_df
    except Exception as e:
        function_name = "dtypes_dict_to_df"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger_transform.error(message)
        raise e


def actual_dtypes(df):
    try:
        df = df.copy()
        log_separator(
            title="function: actual_dtypes()", level=3, logger=logger_transform
        )

        logger_transform.info("Starting actual dtype retrieval.")
        actual_dtypes_dict = {}

        for col in df.columns:
            col_data = df[col]
            dtype = col_data.dtype
            sample_nunique = col_data.value_counts().head(5).to_dict()
            # logger_transform.info(f"Processing column: '{col}', dtype: {dtype}.")

            actual_dtypes_dict[col] = (str(dtype), sample_nunique)

        logger_transform.info("Completed actual dtype retrieval.")
        dtypes_dict_to_df(actual_dtypes_dict)
        return actual_dtypes_dict
    except Exception as e:
        function_name = "actual_dtypes"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger_transform.error(message)
        raise e


def assign_recommendation_dtypes(df_original, n_min_obj=50):
    try:
        df = df_original.copy()
        log_separator(
            title="function: assign_recommendation_dtypes()",
            level=3,
            logger=logger_transform,
        )

        logger_transform.info("Starting dtype assignment.")
        recommendations = {}
        drops = []

        def process_column(col, col_data, dtype_name, checks):
            sample_nunique = col_data.value_counts().head(5).to_dict()

            if col_data.isnull().all():
                logger_transform.warning(
                    f"Recommendation: Drop column '{col}', original type: {df[col].dtype} since 100% of values are null."
                )
                drops.append(col)
                return

            for dtype_recommendation, dtype_check in checks:
                if dtype_check(col_data):
                    recommendations[col] = (dtype_recommendation, sample_nunique)
                    # logger_transform.info(f"Processing {dtype_name} column: '{col}', original type: {df[col].dtype} -> to {dtype_recommendation}")
                    # #, sample: {sample_nunique}")
                    return

            recommendations[col] = ("unknown", sample_nunique)
            logger_transform.info(f"Recommendation: '{col}' will be marked as unknown.")

        def is_float32(col_data):
            return (
                col_data.max() <= np.finfo(np.float32).max
                and col_data.min() >= np.finfo(np.float32).min
            )

        def is_int32(col_data):
            return (
                col_data.max() <= np.iinfo(np.int32).max
                and col_data.min() >= np.iinfo(np.int32).min
            )

        dtype_map = {
            'float': [('float32', is_float32), ('float64', lambda x: True)],
            'integer': [('int32', is_int32), ('int64', lambda x: True)],
            'bool': [('bool', lambda x: True)],
            'object': [
                ('category', lambda x: len(x.unique()) < n_min_obj),
                ('string', lambda x: True),
            ],
            'category': [('category', lambda x: True)],
            'datetime': [('datetime64[ns]', lambda x: True)],
        }

        for dtype, checks in dtype_map.items():
            for col in df.select_dtypes(include=dtype).columns:
                col_data = df[col]
                process_column(col, col_data, dtype, checks)

        # Handle columns without specific dtype recommendations
        for col in df.columns:
            if col not in recommendations and col not in drops:
                col_data = df[col]
                sample_nunique = col_data.value_counts().head(5).to_dict()
                logger_transform.info(
                    f"Processing unclassified column: '{col}', dtype: {df[col].dtype}, sample: {sample_nunique}"
                )

                if col_data.isnull().all():
                    logger_transform.warning(
                        f"Recommendation: Drop column '{col}' since 100% of values are null."
                    )
                    drops.append(col)
                    continue

                recommendations[col] = ("unknown", sample_nunique)
                logger_transform.info(
                    f"Recommendation: '{col}' will be marked as unknown."
                )

        logger_transform.info("Completed dtype assignment recommendations.")
        dtypes_dict_to_df(recommendations)
        if drops:
            logger_transform.warning(f"Columns recommended for dropping: {drops}")

        for col in df_original.columns:
            if col not in recommendations and col not in drops:
                logger_transform.warning(
                    f"Column '{col}' has no recommendations. Type: {df_original[col].dtype}"
                )

        return recommendations
    except Exception as e:
        function_name = "assign_recommendation_dtypes"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger_transform.error(message)
        raise e


def optimize_dataframe(df_original, dtype_recommendations, fill=False, drop=False):
    try:
        df = df_original.copy()
        log_separator(
            title="function: optimize_dataframe()", level=3, logger=logger_transform
        )

        logger_transform.info("Starting DataFrame optimization.")

        # Drop columns with empty samples if they are consistently empty
        columns_to_drop = [
            col
            for col in df_original.columns
            if col not in dtype_recommendations.keys()
        ]
        if drop:
            logger_transform.warning(
                f"Columns dropped due to empty samples: {columns_to_drop}"
            )
            df = df.drop(columns=columns_to_drop)

        # Update the dtype recommendations dictionary
        updated_dtype_recommendations = {
            col: (dtype, sample)
            for col, (dtype, sample) in dtype_recommendations.items()
            if col not in columns_to_drop
        }

        if fill:
            # Fill NA values based on context
            for col, (dtype, sample) in updated_dtype_recommendations.items():
                if dtype.startswith('float') or dtype.startswith('int'):
                    fill_value = df[col].mean()
                    df[col] = df[col].fillna(
                        fill_value
                    )  # Fill numeric columns with mean
                    logger_transform.info(
                        f"Filled NA in column '{col}' with mean value: {fill_value}"
                    )
                elif dtype == 'category':
                    fill_value = df[col].mode()[0]
                    df[col] = df[col].fillna(
                        fill_value
                    )  # Fill categorical columns with mode
                    logger_transform.info(
                        f"Filled NA in column '{col}' with mode value: {fill_value}"
                    )
                elif dtype == 'string':
                    df[col] = df[col].fillna(
                        'Unknown'
                    )  # Fill string columns with 'Unknown'
                    logger_transform.info(f"Filled NA in column '{col}' with 'Unknown'")

        # Convert columns to recommended data types
        for col, (dtype, sample) in updated_dtype_recommendations.items():
            original_dtype = df[col].dtype
            if original_dtype == dtype:
                logger_transform.info(
                    f"Column '{col}' is already of dtype '{dtype}', skipping conversion."
                )
                continue
            # if in object columns there are objects like tuples, dict or list then skip
            if pd.api.types.is_object_dtype(df[col]):
                if df[col].apply(lambda x: isinstance(x, (tuple, dict, list))).any():
                    logger_transform.info(
                        f"Skipping conversion for column '{col}' due to complex objects."
                    )
                    continue
            logger_transform.warning(
                f"Converting column '{col}' from dtype '{original_dtype}' to dtype '{dtype}'"
            )
            if dtype in ['float16', 'float32', 'float64']:
                df[col] = df[col].astype(dtype)
            elif dtype in ['int16', 'int32', 'int64']:
                df[col] = df[col].astype(dtype)
            elif dtype == 'category':
                df[col] = df[col].astype('category')
            elif dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])

        logger_transform.info("Completed DataFrame optimization.")
        dtypes_dict_to_df(updated_dtype_recommendations)
        return df
    except Exception as e:
        function_name = "optimize_dataframe"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def optimize_dataframe_procedure(df_original, df_name="DataFrame"):
    try:
        df = df_original.copy()
        log_separator(
            title="function: optimize_dataframe_procedure()",
            level=3,
            logger=logger_transform,
        )

        logger_transform.info(f"Original '{df_name}': shape {df.shape}")
        tabulate_df(df, logger=logger_transform)

        logger_transform.info(
            "Starting example usage of dtype assignment and optimization."
        )
        actual_dtypes(df)
        dtype_recommendations = assign_recommendation_dtypes(df)
        optimized_df = optimize_dataframe(df, dtype_recommendations)

        # Compare memory usage
        original_memory = df.memory_usage(deep=True).sum()
        # memoria_usada(df, logger=logger_transform)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        # memoria_usada(optimized_df, logger=logger_transform)

        logger_transform.info(
            f"Original '{df_name}' Memory Usage: {original_memory:,.0f} bytes"
        )
        logger_transform.info(
            f"Optimized '{df_name}' Memory Usage: {optimized_memory:,.0f} bytes"
        )
        logger_transform.info(
            f"Memory Reduction: {original_memory - optimized_memory:,.0f} bytes"
        )
        logger_transform.info(
            f"Percentage Reduction: {((original_memory - optimized_memory) / original_memory) * 100:.2f}%"
        )
        logger_transform.info(
            f"Optimized '{df_name}': shape {optimized_df.shape} vs Original '{df_name}': shape {df.shape}"
        )
        return optimized_df
    except Exception as e:
        function_name = "optimize_dataframe_procedure"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------


def process_single_combination(df):
    try:
        df['_COUNTER'] = df.apply(
            lambda row: 1 if pd.notna(row['PROBLEM']) and row['PROBLEM'] != '' else 0,
            axis=1,
        )
        df['_PROBLEM_TO_NULL'] = False
        return df
    except Exception as e:
        function_name = "process_single_combination"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def process_multi_combination(df):
    try:
        # Initialize columns
        df['_COUNTER'] = 0
        df['_PROBLEM_TO_NULL'] = False
        df['PROBLEM'] = df['PROBLEM'].fillna('')
        # Process each group by NUMBER
        for number, group in df.groupby('NUMBER'):
            problems = group['PROBLEM'].values
            indices = group.index

            counter = 0
            last_problem = ''

            for i in range(len(problems)):
                current_problem = problems[i]

                if pd.isna(current_problem) or current_problem == '':
                    if last_problem != '':
                        counter += 1
                        df.loc[indices[i], '_PROBLEM_TO_NULL'] = True
                else:
                    if last_problem == '' or last_problem != current_problem:
                        counter += 1

                df.loc[indices[i], '_COUNTER'] = counter
                last_problem = (
                    current_problem if pd.notna(current_problem) else last_problem
                )

        # now change back '' to na
        df['PROBLEM'] = df['PROBLEM'].replace({'': np.nan})
        return df
    except Exception as e:
        function_name = "process_multi_combination"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def process_incidents(df_original):
    try:
        df = df_original.copy()
        df = df.sort_values(
            by=['NUMBER', 'CREATED', 'UPDATED', '_PCT_ROW_COMP'],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
        df = df.assign(
            _ROW_IN_GROUP=lambda x: (x.groupby('NUMBER').cumcount() + 1).astype(str)
            + '_'
            + x.groupby('NUMBER').transform('size').astype(str)
        )

        # Count occurrences of each NUMBER
        count_per_number = df.groupby('NUMBER')['PROBLEM'].transform(
            lambda x: x.nunique(dropna=False)
        )
        df['_PROBLEM_NUNIQUE_WNULLS'] = count_per_number

        # Separate groups based on count
        single_combination = df[count_per_number == 1].reset_index(drop=True)
        multi_combination = df[count_per_number > 1].reset_index(drop=True)

        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Rows with count_per_number == 1: {len(single_combination)}")
        logger.warning(f"Rows with count_per_number > 1: {len(multi_combination)}")

        # Process each group
        single_combination = process_single_combination(single_combination)
        multi_combination = process_multi_combination(multi_combination)

        # Combine the processed data
        processed_df = (
            pd.concat([single_combination, multi_combination])
            .sort_values(by=['NUMBER', 'CREATED', 'UPDATED'])
            .reset_index(drop=True)
        )

        # add +1 to _COUNT column only those that have nan on 'PROBLEM'
        processed_df.loc[processed_df['PROBLEM'].isna(), '_COUNTER'] += 1

        # Count total rows per NUMBER
        # processed_df['_PROBLEM_COUNT'] = processed_df.groupby('NUMBER')['PROBLEM'].transform('count') # NO CUENTA NULOS
        # Nunique total rows per NUMBER
        # processed_df['_PROBLEM_NUNIQUE'] = processed_df.groupby('NUMBER')['PROBLEM'].transform('nunique') # NO CUENTA NULOS
        # Group by NUMBER and PROBLEM to create a unique group identifier
        # processed_df['_PROBLEM_GROUP'] = processed_df.groupby(['NUMBER', 'PROBLEM']).ngroup() # INDICADOR DE NUMERO DE GRUPO

        df = processed_df[
            processed_df.NUMBER.isin(
                processed_df.query('_PROBLEM_TO_NULL==True').NUMBER.unique()
            )
        ].reset_index(drop=True)
        logger.warning(f"Cases with change from PROBLEM to NULL. Shape: {len(df)}")
        tabulate_df(df)

        return processed_df
    except Exception as e:
        function_name = "process_incidents"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------


def mark_closed_but_not_resolved(
    df, number_col='NUMBER', closed_col='CLOSED', resolved_col='RESOLVED'
):
    try:
        # Identify incident numbers where 'CLOSED' is not null and 'RESOLVED' is null
        incident_numbers = df.loc[
            lambda x: x[closed_col].notna() & x[resolved_col].isna(), number_col
        ].unique()

        # Create a new column to mark these incidents with 1 if true, otherwise null
        df['_CLOSED_BUT_NRES'] = np.where(
            df[number_col].isin(incident_numbers), 1, np.nan
        )

        return df
    except Exception as e:
        function_name = "mark_closed_but_not_resolved"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------


def convert_categorical_to_str(df):
    try:
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['category']).columns

        # Convert categorical columns to strings while preserving nulls
        for col in categorical_cols:
            df[col] = df[col].astype(str).replace('nan', np.nan)

        return df
    except Exception as e:
        function_name = "convert_categorical_to_str"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def track_changes(group, link_to_column):
    try:
        changes_dict = {}
        changes_for_column_tracking = {}

        columns = [
            col for col in group.columns if col != 'UPDATED' and not col.startswith('_')
        ]

        # Sort the DataFrame to ensure chronological order
        group = group.sort_values(
            by='UPDATED', ascending=True
        )  # sort datetime from oldest to newest, starting with the earliest date

        shifted_group = group.shift(1)
        updated_values = group['UPDATED'].values

        # Identify changes between current and previous row
        for column in columns:
            current_values = group[column].values
            previous_values = shifted_group[column].values

            changes = current_values != previous_values
            meaningful_changes = changes & (
                ~pd.isna(current_values) | ~pd.isna(previous_values)
            )

            change_indices = np.where(meaningful_changes)[0]
            change_descriptions = [
                {column: (previous_values[i], current_values[i])}
                for i in change_indices
            ]

            change_count = len(change_indices)
            if change_count > 0:
                changes_for_column_tracking[column] = change_count

            for i, description in zip(change_indices, change_descriptions):
                updated = updated_values[i]
                if updated not in changes_dict:
                    changes_dict[updated] = {}
                changes_dict[updated].update(description)

        try:
            result_group = pd.DataFrame(
                {
                    '_CHANGES_DICT': [changes_dict],
                    '_COUNT_COLUMN_UPDATES': [changes_for_column_tracking],
                },
                index=pd.MultiIndex.from_tuples(
                    [group.index[0]], names=group.index.names
                ),
            )
            return result_group
        except Exception as e:
            function_name = "result_group = pd.DataFrame({"
            message = f"[ERROR]: {function_name} {e}"
            print(message)
            logger.error(message)
            raise e
    except Exception as e:
        function_name = "track_changes"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def format_datetime(dt):
    if pd.isna(dt):  # Check for NaT
        return "nan"
    if isinstance(dt, np.datetime64):
        return pd.to_datetime(dt).strftime('%Y-%m-%dT%H:%M:%S')
    return str(dt)


# Function to recursively convert dictionary keys and values to strings
def convert_keys_and_values_to_str(d):
    if isinstance(d, dict):
        return {
            format_datetime(k): (  # Format datetime keys
                convert_keys_and_values_to_str(v)
                if isinstance(v, dict)
                else (
                    format_datetime(v)
                    if isinstance(
                        v, (np.int32, np.int64, np.float32, np.float64, np.datetime64)
                    )
                    else tuple(map(format_datetime, v)) if isinstance(v, tuple) else v
                )
            )  # Convert tuple elements to strings
            for k, v in d.items()
        }
    return d


def make_uniques(df_original, principal_col='NUMBER', link_to_column='PROBLEM'):
    try:
        # Check if the specified columns exist in the DataFrame
        if (
            principal_col not in df_original.columns
            or link_to_column not in df_original.columns
        ):
            raise ValueError(
                f"Columns '{principal_col}' or '{link_to_column}' not found in DataFrame."
            )

        df = df_original.copy().pipe(convert_categorical_to_str)

        nunique_primary_col = f'_DUPLICATED_{principal_col}'
        nunique_link_col = f'_DUPLICATED_{link_to_column}'

        # Assign duplicate counts using groupby and transform
        df[nunique_primary_col] = df.groupby(principal_col)[principal_col].transform(
            'count'
        )
        df[nunique_link_col] = df.groupby(link_to_column)[link_to_column].transform(
            'count'
        )

        # Sort the DataFrame
        df = df.sort_values(
            by=[nunique_primary_col, principal_col, 'CREATED', 'UPDATED']
        ).reset_index(drop=True)

        df_uniques = df[df['_DUPLICATED_NUMBER'] == 1]
        df = df[df['_DUPLICATED_NUMBER'] > 1]

        # Initialize a list to store results
        df_uniques['_CHANGES_DICT'] = {}
        df_uniques['_COUNT_COLUMN_UPDATES'] = {}
        df_uniques['_IS_UNIQUE'] = True

        results = [df_uniques]
        iteration_count = len(df_uniques)

        df = df.replace({None: np.nan})

        # Group by 'NUMBER' and apply the track_changes function
        for (number, counter, created), group in df.groupby(
            [principal_col, '_COUNTER', 'CREATED'], dropna=False
        ):
            group = group.set_index([principal_col, '_COUNTER']).sort_values(
                by='UPDATED', ascending=True
            )
            df_base = group[group['UPDATED'] == group['UPDATED'].max()].iloc[[-1]]

            changed_columns = [
                col for col in group.columns if group[col].fillna('').nunique() != 1
            ]
            if 'UPDATED' not in changed_columns:
                changed_columns.append('UPDATED')
            group = group[changed_columns]

            if len(df_base) > 1:
                logger.info(
                    f"Processing group for '{principal_col}' {number} with {len(df_base)} rows"
                )
                logger.info(f"Group details:\n{df_base}")
                break
            try:
                changes_df = track_changes(group, link_to_column=link_to_column)

                changes_df.index = changes_df.index.rename(['NUMBER', '_COUNTER'])

                result = df_base.merge(changes_df, left_index=True, right_index=True)
                results.append(result.reset_index())
            except Exception as e:
                function_name = "make_uniques"
                message = f"[ERROR]: {function_name} {e}"
                print(message)
                logger.error(message)
                raise e
            iteration_count += 1

        logger.info(f"Total iterations: {iteration_count}")
        final_result = pd.concat(results)

        # Convert dictionaries to JSON strings using a vectorized approach
        mask = final_result['_COUNT_COLUMN_UPDATES'].apply(
            lambda x: isinstance(x, dict)
        )
        final_result.loc[mask, '_COUNT_COLUMN_UPDATES'] = final_result.loc[
            mask, '_COUNT_COLUMN_UPDATES'
        ].apply(lambda x: json.dumps(convert_keys_and_values_to_str(x)))
        mask = final_result['_CHANGES_DICT'].apply(lambda x: isinstance(x, dict))
        final_result.loc[mask, '_CHANGES_DICT'] = final_result.loc[
            mask, '_CHANGES_DICT'
        ].apply(lambda x: json.dumps(convert_keys_and_values_to_str(x)))
        # Convert other types to strings
        final_result['_COUNT_COLUMN_UPDATES'] = final_result[
            '_COUNT_COLUMN_UPDATES'
        ].astype(str)
        final_result['_CHANGES_DICT'] = final_result['_CHANGES_DICT'].astype(str)

        # tabulate_df(final_result.sort_values(by=['_COUNT_COLUMN_UPDATES'], ascending=[False]).reset_index(drop=True))

        final_result = final_result.sort_values(
            by=['_DUPLICATED_NUMBER', 'CREATED'], ascending=[False, False]
        ).reset_index(drop=True)

        # tabulate_df(final_result.sort_values(by=['_DUPLICATED_NUMBER'], ascending=[False]).reset_index(drop=True))

        return final_result
    except Exception as e:
        function_name = (
            "changes_df = track_changes(group, link_to_column=link_to_column)"
        )
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


def log_and_get_nuniques(df_original, df_name, col_pairs, log=True):
    try:
        # ok test
        # df_incidents[df_incidents._nunique_PROBLEM.isna()].PROBLEM.unique()
        df = df_original.copy()

        for index_col, value_col in col_pairs:
            df_value_counts = get_number_nuniques(
                df, index_col, value_col, df_name, log
            )
            mask = (df_value_counts[f"_nunique_{value_col}"] == 0) & (
                df_value_counts[f"_contains_null_{value_col}"] == True
            )
            df_value_counts = df_value_counts[~mask].drop(
                columns=[f"_contains_null_{value_col}"]
            )
            df = df.merge(df_value_counts, on=index_col, how='left')
        return df
    except Exception as e:
        function_name = "log_and_get_nuniques"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e


# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------


def standardize_company_name(name):
    # Convert to uppercase
    name = name.upper()
    # Remove special characters except spaces
    name = ''.join(e for e in name if e.isalnum() or e.isspace())
    # Strip extra spaces
    name = ' '.join(name.split())
    name = name.replace('S A', 'SA')
    return name


def append_sample_to_value_counts(series, df, mask):
    # Obtener el conteo de valores
    counts = series.value_counts(dropna=False)
    result_df = counts.to_frame(name='COUNT')

    # Para cada valor único en la serie, obtener una muestra de incidentes que cumplan con la máscara
    samples = {}
    for value in counts.index:
        # Aplicar la máscara y seleccionar los incidentes correspondientes al valor actual
        filtered_df = df[mask & (df['STATE'] == value)]
        sample_size = min(5, len(filtered_df))  # Ajustar el tamaño de la muestra
        sample_incidents = (
            filtered_df['NUMBER'].sample(n=sample_size, random_state=1).tolist()
        )
        samples[value] = sample_incidents

    # Agregar la muestra al DataFrame de resultados
    result_df['SAMPLE'] = result_df.index.map(samples)

    return result_df


def clean_note(note, synonyms):
    # Convertir a cadena, eliminar espacios y convertir a minúsculas
    note = str(note).strip().lower()

    # Eliminar caracteres especiales excepto el espacio
    note = re.sub(r'[^\w\s]', '', note)

    # Eliminar espacios dobles
    note = re.sub(r'\s+', ' ', note)
    note = note.strip()

    # Reemplazar sinónimos por un término común
    for key, values in synonyms.items():
        if note in values:
            return key
    return note.strip()


def clean_close_notes(notes):
    # Definir sinónimos y términos irrelevantes
    synonyms = {
        'solucionado': [
            'solucionado',
            'solved',
            'resolved',
            'resuelto',
            'fixed',
            'arreglado',
            'corregido',
            'solventado',
            'ok',
            'ya resuelto',
            'resuelrto',
            'resolevd',
            'solventamos',
            'resolver',
            'resolvemos',
            'gestionado',
            'resuelta',
            'solucionada',
            'solventada',
            'corrected',
            'gestionada',
            'restaurado',
            'resolta',
            'remedied',
            'resuelvo',
            'resulved',
            'ya funciona',
            'funciona',
            'solcuionado',
            'completado',
            'controlado',
            'finalizado',
            'fesolved',
            'resuleto',
            'restablecida',
            'caso resuelto',
            'restablecido',
            'problem solved',
            'issue solved',
            'issue resolved',
            'resolt',
            'checked',
            'todo resuelto',
            'ya solucionado',
            'solved issue',
            'solved thanks',
            'ya entregado',
            'ya gestionado',
            'ya no ocurre',
            'ya realizado',
            'ya reseteado',
            'arreglado todo',
        ],
        'cerrado': [
            'closed',
            'cerrado',
            'close',
            'si cerrados',
            'closing',
            'se cierra',
            'cierro',
            'cerrada',
            'req closed',
            'cierro ticket',
            'caso cerrado',
            'cerramos',
            'ticket cerrado',
        ],
        'actualizado': ['actualizado', 'update done', 'ci updated', 'ci retired'],
        'informado': [
            'informado',
            'informo',
            'informamos',
            'informada',
            'respondido',
            'revisado',
            'se informa',
            'enviado',
            'anotado',
            'enviados',
            'le informo',
            'repuesto',
            'user informed',
            'respuesta dada',
        ],
        'duplicado': ['duplicado', 'duplicate', 'duplicated', 'duplicada'],
        'prueba': [
            'test',
            'prueba',
            'tested',
            'ticket test',
            'test ticket',
            'end test',
        ],
        'ok': ['ok', 'o.k.', 'done', 'hecho', 'realizado'],
        'audit_fail': [
            '.',
            ',',
            ';',
            ':',
            '-',
            "'",
            '!',
            '?',
            'xxx',
            'xxxx',
            'xxxxx',
            'thanks',
            'gracias',
            'no aplica',
            '3',
            '',
            ' ',
            'a',
            'aaa',
            '2',
            '_',
            'thank you',
            'v',
            'zazwimq',
        ],
    }

    # Aplicar la función de limpieza a cada nota usando map
    cleaned_notes = list(map(lambda note: clean_note(note, synonyms), notes))

    return cleaned_notes


def clean_service_offering_column(series):
    # array = df_incidents_grouped['SERVICE_OFFERING'].apply(
    #     lambda x: x.replace(r'^\d+_', '', regex=True)
    # )
    # suffix_pattern = r'(_\d+)?$'
    # .str.replace(suffix_pattern, '', regex=True)
    return series.str.replace(r'^\d+\s*_', '', regex=True).str.replace(' ', '')


def extract_region_and_clean(service_offering):
    suffixes = [
        '_ESP',
        '_IBL',
        '_PT',
        '_BR',
        '_CO',
        '_ES',
        '_US',
        '_AZMEX',
        '_IBEROLATAM',
        '_UK',
        '_FR',
        '_DE',
        '_CH',
        '_EN',
    ]
    for suffix in suffixes:
        if service_offering.endswith(suffix):
            return service_offering.replace(suffix, ''), suffix[1:]
    return service_offering, None


def analyze_service_offerings(df, group_col, offering_col):
    # Count occurrences of each combination of group_col and offering_col
    if offering_col == 'SERVICE_OFFERING':
        new = f'_{offering_col}'
        df[new] = clean_service_offering_column(df[offering_col]).fillna("_VACIO_")
        df[[new, '_SERVICE_OFFERING_OE']] = df[new].apply(
            lambda x: pd.Series(extract_region_and_clean(x))
        )
    else:
        new = offering_col

    counts_df = (
        df[[group_col, new]].value_counts(dropna=False).reset_index(name='count')
    )

    result_df = (
        counts_df.groupby(group_col)
        .agg(
            **{
                f'_COUNT_{offering_col}': ('count', 'sum'),
                f'_NUNIQUE_{offering_col}': (new, 'nunique'),
                f'_DICT_{offering_col}': (
                    new,
                    lambda x: dict(zip(x, counts_df.loc[x.index, 'count'])),
                ),
            }
        )
        .reset_index()
    )

    # Sort the result by _COUNT_SERVICE_OFFERING in descending order
    result_df = result_df.sort_values(
        by=f'_COUNT_{offering_col}', ascending=False
    ).reset_index(drop=True)

    return result_df


def process_not_closed(df_original, date_pairs):
    df = df_original.copy()
    logger.info("Original DataFrame shape: %s", df.shape)

    # Define the comparison date for 'TODAY'
    comparison_date = pd.to_datetime('2025-09-01')

    # Function to calculate timedelta difference
    def calculate_timedelta_difference(start_date, end_date):
        if pd.isna(start_date) or pd.isna(end_date):
            return None  # Return None if either date is NaN
        difference = end_date - start_date
        # Adjust the representation for negative differences
        if difference.total_seconds() < 0:
            # If the difference is negative, represent it as negative minutes
            total_seconds = abs(difference.total_seconds())
            minutes = total_seconds // 60
            return timedelta(minutes=-minutes)
        return difference

    # Process date differences for incidents that are not closed
    for col1, col2 in date_pairs:
        if col2 == 'TODAY':
            # Calculate difference with comparison_date for non-closed incidents
            df[f'_DIFF_{col1}_{col2}'] = df.apply(
                lambda row: (
                    calculate_timedelta_difference(row[col1], comparison_date)
                    if row['STATE'] != 'Closed'
                    else None
                ),
                axis=1,
            )
        else:
            # Calculate difference between two date columns for non-closed incidents
            df[f'_DIFF_{col1}_{col2}'] = df.apply(
                lambda row: (
                    calculate_timedelta_difference(row[col1], row[col2])
                    if row['STATE'] != 'Closed'
                    else None
                ),
                axis=1,
            )

    # Process date differences for incidents that are closed
    closed_date_pairs = [pair for pair in date_pairs if 'TODAY' not in pair]
    for col1, col2 in closed_date_pairs:
        df[f'_DIFF_{col1}_{col2}'] = df.apply(
            lambda row: (
                calculate_timedelta_difference(row[col1], row[col2])
                if row['STATE'] == 'Closed'
                else None
            ),
            axis=1,
        )

    return df


from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    EasterMonday,
    GoodFriday,
)
from dateutil.relativedelta import relativedelta as rd
from pandas.tseries.offsets import Day


class SpanishHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1),
        Holiday('Epiphany', month=1, day=6),
        GoodFriday,
        EasterMonday,
        Holiday('Labor Day', month=5, day=1),
        Holiday('Assumption of Mary', month=8, day=15),
        Holiday('National Day', month=10, day=12),
        Holiday('All Saints', month=11, day=1),
        Holiday('Constitution Day', month=12, day=6),
        Holiday('Immaculate Conception', month=12, day=8),
        Holiday('Christmas Day', month=12, day=25),
    ]


def process_not_closed_business(df_original, date_pairs):
    df = df_original.copy()
    logger.info("Original DataFrame shape: %s", df.shape)

    # Define the comparison date for 'TODAY'
    comparison_date = pd.to_datetime('2025-09-01')

    def calculate_business_hours(start_date, end_date):
        if pd.isna(start_date) or pd.isna(end_date):
            return None

        # Ensure start_date is before end_date
        if start_date > end_date:
            start_date, end_date = end_date, start_date
            is_negative = True
        else:
            is_negative = False

        # Create a Spanish Holiday calendar
        cal = SpanishHolidayCalendar()
        holidays = cal.holidays(start=start_date, end=end_date)

        business_hours = timedelta()
        current_date = start_date.date()

        while current_date <= end_date.date():
            if (
                current_date.weekday() < 5 and current_date not in holidays
            ):  # Weekday and not a holiday
                day_start = datetime.combine(current_date, time(9, 0))
                day_end = datetime.combine(current_date, time(17, 0))

                if current_date == start_date.date():
                    day_start = max(day_start, start_date)
                if current_date == end_date.date():
                    day_end = min(day_end, end_date)

                if day_end > day_start:
                    business_hours += min(
                        day_end, datetime.combine(current_date, time(17, 0))
                    ) - max(day_start, datetime.combine(current_date, time(9, 0)))

            current_date += timedelta(days=1)

        hours = business_hours.total_seconds() / 3600

        return -hours if is_negative else hours

    # Process date differences for incidents that are not closed
    for col1, col2 in date_pairs:
        if col2 == 'TODAY':
            df[f'_DIFB_{col1}_{col2}'] = df.apply(
                lambda row: (
                    calculate_business_hours(row[col1], comparison_date)
                    if row['STATE'] != 'Closed'
                    else None
                ),
                axis=1,
            )
        else:
            df[f'_DIFB_{col1}_{col2}'] = df.apply(
                lambda row: (
                    calculate_business_hours(row[col1], row[col2])
                    if row['STATE'] != 'Closed'
                    else None
                ),
                axis=1,
            )

    # Process date differences for incidents that are closed
    closed_date_pairs = [pair for pair in date_pairs if 'TODAY' not in pair]
    for col1, col2 in closed_date_pairs:
        df[f'_DIFB_{col1}_{col2}'] = df.apply(
            lambda row: (
                calculate_business_hours(row[col1], row[col2])
                if row['STATE'] == 'Closed'
                else None
            ),
            axis=1,
        )

    return df


def exceed_bands_priority_CRE_RES(df, priority_col):
    # Define the thresholds for each priority level in hours
    thresholds = {'P1': 12, 'P2': 20, 'P3': 120, 'P4': 160}
    df['_SLA_THRESHOLD'] = df[priority_col].map(thresholds)
    # Create the '_SLA_OK' column by checking if the outage duration exceeds the threshold ## row['_DIFB_CREATED_RESOLVED'].total_seconds() / 3600
    df[f'_SLA_{priority_col}'] = df.apply(
        lambda row: (
            row['_DIFB_CREATED_RESOLVED']
            <= thresholds.get(row[priority_col], float('inf'))
            if row['_DIFB_CREATED_RESOLVED'] is not None
            else row['_DIFB_CREATED_TODAY']
            <= thresholds.get(row[priority_col], float('inf'))
        ),
        axis=1,
    )

    return df


def process_not_closed_business_vectorized(df_original, date_pairs):
    df = df_original.copy()
    logger.info("Original DataFrame shape: %s", df.shape)

    # Define the comparison date for 'TODAY'
    comparison_date = pd.to_datetime('2025-09-01')

    # Create a Spanish Holiday calendar once
    cal = SpanishHolidayCalendar()

    def vectorized_business_hours(start_series, end_series):
        # Create a DataFrame to hold our date pairs
        date_df = pd.DataFrame({'start': start_series, 'end': end_series})

        # Handle cases where end date is missing (use comparison_date)
        date_df['end'] = date_df['end'].fillna(comparison_date)

        # Handle cases where start date is missing
        mask_start_missing = date_df['start'].isna()
        results = pd.Series(np.nan, index=date_df.index)

        # Process only valid date pairs
        valid_pairs = date_df[~mask_start_missing].copy()

        # Ensure start_date is before end_date
        swap_needed = valid_pairs['start'] > valid_pairs['end']
        temp = valid_pairs.loc[swap_needed, 'start'].copy()
        valid_pairs.loc[swap_needed, 'start'] = valid_pairs.loc[swap_needed, 'end']
        valid_pairs.loc[swap_needed, 'end'] = temp
        sign_multiplier = pd.Series(1, index=valid_pairs.index)
        sign_multiplier[swap_needed] = -1

        # Generate business days between dates
        business_hours = []

        # Process in chunks for better performance
        for _, chunk in valid_pairs.groupby(
            valid_pairs.index // 1000
        ):  # Process in chunks of 1000
            chunk_results = []

            for idx, row in chunk.iterrows():
                start_date = row['start']
                end_date = row['end']

                # Get all holidays between these dates
                holidays = cal.holidays(start=start_date, end=end_date)

                # Generate all dates between start and end
                date_range = pd.date_range(start=start_date.date(), end=end_date.date())

                # Filter to only business days (weekdays and not holidays)
                business_days = date_range[
                    ~date_range.isin(holidays) & (date_range.dayofweek < 5)
                ]

                # Calculate hours for each business day
                total_hours = 0

                for day in business_days:
                    day_start = pd.Timestamp.combine(
                        day, pd.Timestamp('09:00:00').time()
                    )
                    day_end = pd.Timestamp.combine(day, pd.Timestamp('17:00:00').time())

                    # Adjust for first and last days
                    if day.date() == start_date.date():
                        day_start = max(day_start, start_date)
                    if day.date() == end_date.date():
                        day_end = min(day_end, end_date)

                    # Calculate hours if valid range
                    if day_end > day_start:
                        hours = (day_end - day_start).total_seconds() / 3600
                        total_hours += hours

                chunk_results.append(total_hours)

            business_hours.extend(chunk_results)

        # Apply sign multiplier and store results
        valid_indices = valid_pairs.index
        results[valid_indices] = [
            h * m for h, m in zip(business_hours, sign_multiplier)
        ]

        return results

    # Process date differences for all date pairs
    for col1, col2 in date_pairs:
        df[f'_DIFB_{col1}_{col2}'] = vectorized_business_hours(df[col1], df[col2])

    return df





def process_not_closed_business_vectorized(df_original, date_pairs):
    """
    Process business hours between date pairs efficiently.

    Args:
        df_original: DataFrame with date columns (CREATED, RESOLVED, CLOSED, etc.)
        date_pairs: List of tuples with column pairs to compare

    Returns:
        DataFrame with added columns for business hours differences (_DIFB_*)
        and business day differences (_DIFB_*_d)
    """
    df = df_original.copy()
    logger.info("Original DataFrame shape: %s", df.shape)

    # Define the comparison date for handling NaT values
    comparison_date = pd.to_datetime('2025-09-01')

    # Create a Spanish Holiday calendar once
    cal = SpanishHolidayCalendar()

    # Define business hours
    BUSINESS_START = time(9, 0)
    BUSINESS_END = time(17, 0)
    BUSINESS_HOURS_PER_DAY = 8.0

    # Pre-generate all holidays in the relevant range
    all_date_cols = set(col for pair in date_pairs for col in pair)

    # Find min and max dates across all columns
    min_date = pd.Timestamp.max
    max_date = pd.Timestamp.min

    for col in all_date_cols:
        if col in df.columns:
            non_nat_values = df[col].dropna()
            if not non_nat_values.empty:
                min_col = non_nat_values.min()
                if min_col < min_date:
                    min_date = min_col

                max_col = non_nat_values.max()
                if max_col > max_date:
                    max_date = max_col

    # Ensure we have valid min/max dates
    if max_date < min_date:
        logger.warning("No valid date range found, using default range")
        min_date = pd.Timestamp('2020-01-01')
        max_date = comparison_date

    # Add buffer to max_date
    max_date = max(max_date, comparison_date)

    logger.info(f"Calculating holidays between {min_date} and {max_date}")

    # Get all holidays in one go
    all_holidays = cal.holidays(start=min_date, end=max_date).date
    all_holidays_set = set(all_holidays)

    # Pre-compile business days for the entire date range
    date_range = pd.date_range(start=min_date.date(), end=max_date.date())
    business_days_mask = ~date_range.isin(all_holidays) & (date_range.dayofweek < 5)
    business_days = date_range[business_days_mask]
    business_days_set = set(d.date() for d in business_days)

    logger.info(f"Found {len(business_days_set)} business days in range")

    def calculate_hours_for_day(day_start, day_end):
        """Calculate hours between timestamps"""
        if day_end > day_start:
            return (day_end - day_start).total_seconds() / 3600
        return 0.0

    def calculate_business_hours_batch(start_dates, end_dates):
        """Calculate business hours for a batch of date pairs"""
        results = np.empty(len(start_dates))

        for i in range(len(start_dates)):
            start_date = start_dates.iloc[i]
            end_date = end_dates.iloc[i]

            # Handle NaT values
            if pd.isna(start_date):
                results[i] = np.nan
                continue

            if pd.isna(end_date):
                end_date = comparison_date

            # Ensure start_date is before end_date
            if start_date > end_date:
                start_date, end_date = end_date, start_date
                sign = -1
            else:
                sign = 1

            # Fast path for same day
            if start_date.date() == end_date.date():
                if start_date.date() in business_days_set:
                    day_start = pd.Timestamp.combine(start_date.date(), BUSINESS_START)
                    day_end = pd.Timestamp.combine(start_date.date(), BUSINESS_END)

                    day_start = max(day_start, start_date)
                    day_end = min(day_end, end_date)

                    total_hours = calculate_hours_for_day(day_start, day_end)
                    results[i] = total_hours * sign
                else:
                    results[i] = 0.0
                continue

            # For multi-day spans
            total_hours = 0.0

            # Handle first day
            if start_date.date() in business_days_set:
                day_start = max(
                    pd.Timestamp.combine(start_date.date(), BUSINESS_START), start_date
                )
                day_end = pd.Timestamp.combine(start_date.date(), BUSINESS_END)
                total_hours += calculate_hours_for_day(day_start, day_end)

            # Handle middle days efficiently
            # Instead of generating all dates, count days directly
            current_date = start_date.date() + pd.Timedelta(days=1)
            end_date_minus_one = end_date.date() - pd.Timedelta(days=1)

            while current_date <= end_date_minus_one:
                if current_date in business_days_set:
                    total_hours += BUSINESS_HOURS_PER_DAY
                current_date += pd.Timedelta(days=1)

            # Handle last day
            if (
                end_date.date() in business_days_set
                and end_date.date() > start_date.date()
            ):
                day_start = pd.Timestamp.combine(end_date.date(), BUSINESS_START)
                day_end = min(
                    pd.Timestamp.combine(end_date.date(), BUSINESS_END), end_date
                )
                total_hours += calculate_hours_for_day(day_start, day_end)

            results[i] = total_hours * sign

        return results

    # Process date differences for all date pairs
    for col1, col2 in date_pairs:
        logger.info(f"Processing date pair: {col1} - {col2}")

        # Check if both columns exist
        if col1 not in df.columns or col2 not in df.columns:
            logger.warning(
                f"Columns {col1} and/or {col2} not found in DataFrame, skipping"
            )
            continue

        # Process in batches
        batch_size = 10000
        result_series = pd.Series(index=df.index)

        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            logger.info(f"Processing batch {start_idx}-{end_idx} for {col1}-{col2}")

            batch_results = calculate_business_hours_batch(
                df[col1].iloc[start_idx:end_idx], df[col2].iloc[start_idx:end_idx]
            )

            result_series.iloc[start_idx:end_idx] = batch_results

        # Store hours results
        hours_col = f'_DIFB_{col1}_{col2}'
        df[hours_col] = result_series

        # Create days column by dividing hours by business hours per day
        days_col = f'_DIFB_{col1}_{col2}_d'
        df[days_col] = df[hours_col] / BUSINESS_HOURS_PER_DAY

        logger.info(f"Created hours column {hours_col} and days column {days_col}")

    return df


def exceed_bands_priority_CRE_RES_optimized(df, priority_col):
    thresholds = {'P1': 12, 'P2': 20, 'P3': 120, 'P4': 160}

    df['_SLA_THRESHOLD'] = df[priority_col].map(thresholds)

    has_resolution = ~df['_DIFB_CREATED_RESOLVED'].isna()

    df[f'_SLA_{priority_col}'] = False

    closed_mask = has_resolution
    if closed_mask.any():
        df.loc[closed_mask, f'_SLA_{priority_col}'] = (
            df.loc[closed_mask, '_DIFB_CREATED_RESOLVED']
            <= df.loc[closed_mask, '_SLA_THRESHOLD']
        )

    return df


# --


def sort_values_of_incidents(df, extra_col=None):
    if extra_col:
        return df.sort_values(
            [extra_col, 'NUMBER', 'UPDATED'], ascending=[False, True, True]
        ).reset_index(drop=True)
    return df.sort_values(['NUMBER', 'UPDATED'], ascending=[True, True]).reset_index(
        drop=True
    )


def get_inc_count_rows(df):
    """
    Count the number of rows per incident number and add as a new column.
    """
    # Pre-compute the counts once and map them directly
    counts = df['NUMBER'].value_counts()
    return (
        df.assign(_INC_COUNT_ROWS=df['NUMBER'].map(counts))
        .sort_values('_INC_COUNT_ROWS', ascending=False)
        .reset_index(drop=True)
    )


def get_COLvals_list_per_inc(df, func_col):
    """
    Create a comma-separated string of problems per incident number.
    """
    # Group outside of the mapping for better performance
    lists = (
        df.dropna(subset=[func_col])
        .groupby('NUMBER')[func_col]
        .apply(lambda x: ','.join(set(x)))
    )

    return df.assign(**{f'_{func_col}_LIST_PER_INC': df['NUMBER'].map(lists)})


def get_COLvals_count_per_inc(df, func_col):
    counts = df.dropna(subset=[func_col]).groupby('NUMBER')[func_col].size()

    # Convert to dictionary for faster mapping, with default of 0
    count_dict = counts.to_dict()

    return (
        df.assign(
            **{
                f'_{func_col}_COUNT_PER_INC': df['NUMBER'].map(
                    lambda x: count_dict.get(x, 0)
                )
            }
        )
        .sort_values(f'_{func_col}_COUNT_PER_INC', ascending=False)
        .reset_index(drop=True)
    )


def get_COLvals_countUnique_per_inc(df, func_col):
    counts = df.dropna(subset=[func_col]).groupby('NUMBER')[func_col].nunique()

    # Convert to dictionary for faster mapping, with default of 0
    count_dict = counts.to_dict()

    return (
        df.assign(
            **{
                f'_{func_col}_COUNT_UNIQUE_PER_INC': df['NUMBER'].map(
                    lambda x: count_dict.get(x, 0)
                )
            }
        )
        .sort_values(f'_{func_col}_COUNT_UNIQUE_PER_INC', ascending=False)
        .reset_index(drop=True)
    )


def get_bool_comparation(df, col, col2):
    mask = df[col] != df[col2]
    df[f'_{col}_NOT_EQUAL_{col2}'] = mask
    return df


def flag_state_change(df):
    """
    Flags improper state transitions in tickets where:
    - 'Resolved' status is followed by anything other than 'Closed' or repeated 'Resolved' status
    - 'Closed' status is followed by any other status (reopened tickets) or repeated 'Closed' status

    Returns:
        DataFrame with additional columns:
        - '_IMPROPER_STATE_CHANGE': Boolean flag for improper transitions
        - '_IMPROPER_STATE_COUNTS': Count of improper transitions per ticket
    """
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()

    # Initialize the columns
    result_df['_IMPROPER_STATE_CHANGE'] = False
    result_df['_IMPROPER_STATE_COUNTS'] = 0

    # Process each ticket (NUMBER) separately
    groups = result_df.sort_values(
        ['NUMBER', 'UPDATED'], ascending=[True, True]
    ).groupby('NUMBER')

    # Dictionary to store improper transition counts per ticket
    improper_counts = {}

    for number, group in groups:
        # Get the chronological sequence of states
        states = group['STATE'].tolist()

        count = 0  # Counter for this specific ticket
        has_improper = False

        # Check for improper transitions
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]

            # Resolved should only be followed by Closed or Resolved
            if current_state == 'Resolved' and next_state not in ['Closed', 'Resolved']:
                count += 1
                has_improper = True

            # Closed should only be followed by Closed
            elif current_state == 'Closed' and next_state != 'Closed':
                count += 1
                has_improper = True

        # Only store information if there are improper transitions
        if has_improper:
            improper_counts[number] = count

    # Apply flags and counts to the dataframe
    if improper_counts:
        # Set the improper transition flag
        result_df.loc[
            result_df['NUMBER'].isin(improper_counts.keys()), '_IMPROPER_STATE_CHANGE'
        ] = True

        # Set the counts for each ticket
        for number, count in improper_counts.items():
            result_df.loc[result_df['NUMBER'] == number, '_IMPROPER_STATE_COUNTS'] = (
                count
            )

    return result_df


def clean_ASSIGNMENT_GROUP(df, col='ASSIGNMENT_GROUP'):

    df[col] = df[col].str.strip().str.upper().str.replace(' ', '')
    return df


def clean_LAST_ASSIGNMENT_GROUP(df, col='LAST_ASSIGNMENT_GROUP'):
    df[col] = df[col].str.strip().str.upper().str.replace(' ', '')
    return df


def tweak_confluence(df):
    return (
        df.groupby('LAST_ASSIGNMENT_GROUP')
        .agg(lambda x: ', '.join(sorted(set(str(val) for val in x if pd.notna(val)))))
        .reset_index()
    )


def load_confluence():
    return (
        pd.read_excel(r"../assets/teams_confluence.xlsx")
        .pipe(clean_headers)
        .rename(
            columns={'POST_MIGRATION_ASSIGNMENT_GROUP_NAME': 'LAST_ASSIGNMENT_GROUP'}
        )
        .pipe(clean_LAST_ASSIGNMENT_GROUP)
        .add_prefix('_')
        .add_suffix('_CONFLUENCE')
        .rename(columns={'_LAST_ASSIGNMENT_GROUP_CONFLUENCE': 'LAST_ASSIGNMENT_GROUP'})
    ).pipe(tweak_confluence)


# --


def process_not_closed_business_optimized(df_original, date_pairs):
    df = df_original.copy()
    logger.info("Original DataFrame shape: %s", df.shape)

    # Define the comparison date for 'TODAY'
    comparison_date = pd.to_datetime('2025-09-01')

    # Create a Spanish Holiday calendar once for reuse
    cal = SpanishHolidayCalendar()

    # Function to calculate business hours between two dates
    def calculate_business_hours_vectorized(start_dates, end_dates):
        """Vectorized version of calculate_business_hours"""
        result = pd.Series(index=start_dates.index, dtype=float)

        # Handle NaN values
        valid_mask = ~(start_dates.isna() | end_dates.isna())
        if not valid_mask.any():
            return result

        s_dates = start_dates[valid_mask]
        e_dates = end_dates[valid_mask]

        # Determine where start > end
        swap_needed = s_dates > e_dates

        # Ensure start_date is always before end_date
        temp_s = s_dates.copy()
        s_dates = pd.Series(
            np.where(swap_needed, e_dates, s_dates), index=s_dates.index
        )
        e_dates = pd.Series(np.where(swap_needed, temp_s, e_dates), index=e_dates.index)

        # Get all unique dates in the range for holiday calculation
        min_date = s_dates.min().date()
        max_date = e_dates.max().date()
        holidays = cal.holidays(start=min_date, end=max_date)

        # Create a mapping of dates to their business day status (1 for business day, 0 for holiday/weekend)
        date_range = pd.date_range(min_date, max_date)
        business_days = {
            d.date(): (d.weekday() < 5 and d.date() not in holidays) for d in date_range
        }

        hours_list = []

        # Process each pair of dates
        for idx, (start_date, end_date) in enumerate(zip(s_dates, e_dates)):
            business_hours = timedelta()
            current_date = start_date.date()

            while current_date <= end_date.date():
                if business_days[current_date]:  # Weekday and not a holiday
                    day_start = datetime.combine(current_date, time(9, 0))
                    day_end = datetime.combine(current_date, time(17, 0))

                    if current_date == start_date.date():
                        day_start = max(day_start, start_date)
                    if current_date == end_date.date():
                        day_end = min(day_end, end_date)

                    if day_end > day_start:
                        business_hours += min(
                            day_end, datetime.combine(current_date, time(17, 0))
                        ) - max(day_start, datetime.combine(current_date, time(9, 0)))

                current_date += timedelta(days=1)

            hours = business_hours.total_seconds() / 3600
            if swap_needed.iloc[idx]:
                hours = -hours

            hours_list.append(hours)

        # Assign results back to the correct positions
        result.loc[valid_mask] = hours_list
        return result

    # Split the dataframe into closed and not closed
    closed_df = df[df['STATE'] == 'Closed']
    not_closed_df = df[df['STATE'] != 'Closed']

    # Process not closed incidents
    today_pairs = [pair for pair in date_pairs if 'TODAY' in pair]
    for col1, col2 in today_pairs:
        if col2 == 'TODAY':
            col_name = f'_DIFB_{col1}_{col2}'
            if not not_closed_df.empty:
                start_dates = not_closed_df[col1]
                end_dates = pd.Series(comparison_date, index=start_dates.index)
                not_closed_df[col_name] = calculate_business_hours_vectorized(
                    start_dates, end_dates
                )

    # Process all date pairs for closed incidents
    non_today_pairs = [pair for pair in date_pairs if 'TODAY' not in pair]
    for col1, col2 in non_today_pairs:
        col_name = f'_DIFB_{col1}_{col2}'

        # For closed tickets
        if not closed_df.empty:
            closed_df[col_name] = calculate_business_hours_vectorized(
                closed_df[col1], closed_df[col2]
            )

        # For non-closed tickets
        if not not_closed_df.empty:
            not_closed_df[col_name] = calculate_business_hours_vectorized(
                not_closed_df[col1], not_closed_df[col2]
            )

    # Recombine the dataframes
    result_df = pd.concat([closed_df, not_closed_df])

    # Sort to preserve original order
    result_df = result_df.sort_index()

    return result_df


def plot_dates_periods(df):
    print(df.select_dtypes(include="datetime64[ns]").columns)
    print()
    for col in df.select_dtypes(include="datetime64[ns]").columns:
        df_sub = df[[col]].copy()
        # Period
        df_sub[col] = pd.to_datetime(df_sub[col].dt.date)
        # hig_border = datetime.today().strftime('%Y-%m-%d')  # Adjusted to today's date
        # Define date range and convert to datetime
        low_border = pd.to_datetime("2023-01-01")
        hig_border = pd.to_datetime("2025-09-01")

        # Filter DataFrame using boolean indexing
        df_filtered = df_sub[(df_sub[col] >= low_border) & (df_sub[col] < hig_border)]
        df_sub[[col]].hist()
        for n, col2 in enumerate(df_sub.columns):
            df_count = (
                df_sub[col2]
                .value_counts(dropna=False)
                .to_frame("COUNT")
                .reset_index()
                .rename(columns={"index": col})
            )
            df_count = reset_df(df_count.sort_values(by=col2))
            if n == 0:
                window = 20
                sigma = 2
                df_count["floor"] = df_count["COUNT"].rolling(window=window).mean() - (
                    sigma * df_count["COUNT"].rolling(window=window).std()
                )
                df_count["top"] = df_count["COUNT"].rolling(window=window).mean() + (
                    sigma * df_count["COUNT"].rolling(window=window).std()
                )
                df_count["alert"] = df_count.apply(
                    lambda row: row["COUNT"] if (row["COUNT"] >= row["top"]) else 0,
                    axis=1,
                )
                mask = df_count["alert"] > 0
                alert_dates = df_count.loc[mask, col].tolist()  # Extracting alert dates
                alert_table = df_count.loc[mask, [col, "COUNT"]]
                print("ALERT TABLE: ")
                display(alert_table.reset_index(drop=True))
                print()
                df_count.plot(
                    x=col2,
                    linewidth=0.7,
                    rot=45,
                    legend=None,
                    title=f'Count {col2} across time',
                    xlabel='',
                    fontsize=10,
                    figsize=(20, 8),
                )
                plt.show()

                # Additional plot for daily lines for each year
                df_sub['DAY_OF_YEAR'] = df_sub[col].dt.dayofyear
                yearly_data = (
                    df_sub.groupby([df_sub[col].dt.year, 'DAY_OF_YEAR'])
                    .size()
                    .unstack(level=0)
                )

                # Plot configuration
                plt.figure(figsize=(25, 6))
                marker_styles = ['o', 's', '^']  # Different marker styles for each year
                year_colors = {
                    2022: 'Yellow',
                    2023: 'Greens',
                    2024: 'Blues',
                    2025: 'Reds',
                }

                # Define transparency levels
                alpha_full = 0.8  # Full visibility for Jan-Aug
                alpha_reduced = 0.4  # Reduced visibility for Sep-Dec

                for i, year in enumerate(yearly_data.columns):
                    # Add random noise to x and y positions
                    x_jitter = yearly_data.index + np.random.uniform(
                        -0.3, 0.3, size=yearly_data.index.size
                    )
                    y_jitter = yearly_data[year] + np.random.uniform(
                        -0.3, 0.3, size=yearly_data[year].size
                    )

                    # Determine transparency based on the month
                    alpha_values = np.where(
                        x_jitter <= 243, alpha_full, alpha_reduced
                    )  # Day 243 corresponds to Aug 31

                    plt.scatter(
                        x_jitter,
                        y_jitter,
                        label=str(year),
                        alpha=alpha_values,
                        marker=marker_styles[i % len(marker_styles)],
                    )

                plt.title('Daily Counts Comparison Across Years with Jitter')
                plt.xlabel('Day of Year')
                plt.ylabel('Count')
                plt.legend(title='Year')
                plt.grid(True)
                plt.show()

            elif n > 1:
                print("FREQUENCY: ")
                display(df_count)
                print()
                df_count.plot(
                    x=col2,
                    y="COUNT",
                    kind='line',
                    linewidth=0.5,
                    rot=45,
                    legend=None,
                    title=f'Count {col2} across time',
                    xlabel='',
                    fontsize=10,
                    figsize=(12, 4),
                )
                plt.show()


def find_words_in_row(row, words):
    if isinstance(words, str):
        search_words = [words.lower()]
    else:
        search_words = [str(word).lower() for word in words]
    found_details = []

    for column, entry in row.items():
        # Skip columns by name pattern
        if (
            str(column).startswith('Status')
            or str(column).endswith('_Info')
            or str(column).startswith('_word')
        ):
            continue
        # Handle missing values
        if pd.isnull(entry):
            continue
        # Handle nested dictionaries
        if isinstance(entry, dict):
            for root_key, nested_dict in entry.items():
                if isinstance(nested_dict, dict):
                    for key, value in nested_dict.items():
                        try:
                            if any(word in str(value).lower() for word in search_words):
                                found_details.append(
                                    {f"{column}.{root_key}.{key}": value}
                                )
                        except Exception:
                            continue
        # Handle lists
        elif isinstance(entry, list):
            for idx, item in enumerate(entry):
                try:
                    if any(word in str(item).lower() for word in search_words):
                        found_details.append({f"{column}[{idx}]": item})
                except Exception:
                    continue
        # Handle strings and other types
        else:
            try:
                if any(word in str(entry).lower() for word in search_words):
                    found_details.append({f"{column}": entry})
            except Exception:
                continue

    n_cases = len([value for value in found_details if pd.notnull(value)])
    return f'Cases:{str(n_cases)}\n\n' + '\n___\n'.join(
        [str(val) for val in found_details]
    ).replace('|', '\n -> ')


def get_entire_rows_with_words(df, words):
    if isinstance(words, str):
        words = [words]
    words_upper = [w.upper() for w in words]

    # Create mask for any word in any object column
    # Use numpy vectorized string operations for performance
    obj_cols = df.select_dtypes(include='object').columns
    mask = df[obj_cols].apply(
        lambda col: col.astype(str)
        .str.upper()
        .str.contains('|'.join(words_upper), na=False)
    )

    return df[mask.any(axis=1)]


def get_rows_columns_with_words(df, keywords):
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords_upper = [str(keyword).upper() for keyword in keywords]

    def contains_keyword(x):
        try:
            return any(keyword in str(x).upper() for keyword in keywords_upper)
        except Exception:
            return False

    # Convert categorical columns to strings
    df = df.apply(lambda col: col.astype(str) if col.dtype.name == 'category' else col)

    mask = df.map(contains_keyword)

    filtered_df = df.loc[mask.any(axis=1), mask.any(axis=0)]

    return filtered_df


def find_columns_value_match(
    df, ignore_columns=['NUMBER', 'CREATED_OUT_OF_INCIDENT', 'PROBLEM']
):
    # Define patterns with regex to match similar structures
    patterns = [
        r'PRB\d{7}',  # Matches 'PRB' followed by 7 digits
        r'INC\d{8}',  # Matches 'INC' followed by 8 digits
        r'PTASK\d{7}',  # Matches 'PTASK' followed by 7 digits
        r'CHG\d{7}',  # Matches 'CHG' followed by 7 digits
    ]

    # Compile each pattern into a regex object
    compiled_patterns = [re.compile(pattern) for pattern in patterns]

    # Ensure ignore_columns is a list
    if ignore_columns is None:
        ignore_columns = []

    # Function to collect all pattern matches in a row
    def collect_matches(row):
        matches = []
        for column, value in row.items():
            if column in ignore_columns:
                continue
            for pattern in compiled_patterns:
                matches.extend(pattern.findall(str(value)))
        return list(set(matches))

    # Apply the function across rows and create '_LINKED_TICKETS' column
    df['_LINKED_TICKETS'] = df.apply(collect_matches, axis=1)

    # Function to remove matches that originate from ignored columns
    def remove_ignored_matches(row):
        ignored_matches = []
        for column in ignore_columns:
            if column in df.columns:
                value = row[column]
                for pattern in compiled_patterns:
                    ignored_matches.extend(pattern.findall(str(value)))
        # Remove ignored matches from '_LINKED_TICKETS'
        return [p for p in row['_LINKED_TICKETS'] if p not in set(ignored_matches)]

    # Apply the function to remove ignored matches
    df['_LINKED_TICKETS'] = df.apply(remove_ignored_matches, axis=1)
    df['_LINKED_TICKETS_COUNT'] = df['_LINKED_TICKETS'].apply(len)
    df['_LINKED_TICKETS'] = df['_LINKED_TICKETS'].apply(lambda x: ','.join(x))
    # logger.info("\n" + tabulate(df, headers='keys', tablefmt='psql'))
    return df


def get_LINKED_TICKETS_summary(df):
    """
    Generate a summary of pattern tickets from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A summary DataFrame with the count of pattern tickets per problem.
    """
    # Group by 'NUMBER' and aggregate the data
    df['_LINKED_TICKETS'] = df['_LINKED_TICKETS'].str.split(',')
    grouped_data = (
        df.loc[df['_LINKED_TICKETS'].map(lambda x: len(x) > 0)]
        .groupby('NUMBER')
        .agg(
            _LINKED_TICKETS_COUNT=(
                '_LINKED_TICKETS',
                lambda x: sum(len(matches) for matches in x),
            ),
            _SAMPLE_TICKETS=(
                '_LINKED_TICKETS',
                lambda x: ', '.join([item for sublist in x for item in sublist][:5]),
            ),
        )
        .reset_index()
        .sort_values(by='_LINKED_TICKETS_COUNT', ascending=False)
    )

    # Log the information using tabulate
    # logger.info("\n" + tabulate(grouped_data, headers='keys', tablefmt='psql'))
    return grouped_data


def flatten_df_columns(df):
    df.columns = [
        '_'.join(tuple(item for item in col if item != '')).strip()
        for col in df.columns.values
    ]
    return df


def flatten_df_columns(df):
    df.columns = [
        '_'.join(str(item) for item in col if item != '').strip()
        for col in df.columns.values
    ]
    return df


def find_matching_columns(df1, df2):
    """
    This function takes two DataFrames as input and returns a list of columns
    that are present in both DataFrames.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.

    Returns:
    list: A list of matching column names.
    """
    # Get the columns of each DataFrame
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)

    # Find matching columns
    matching_columns = columns_df1.intersection(columns_df2)

    return list(matching_columns)


def write_df_to_excel(writer, sheet_name, df):
    # Check if the DataFrame has MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # Concatenate the levels of the MultiIndex columns using '_'
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

    # Write the DataFrame to the Excel sheet
    df.to_excel(writer, sheet_name=sheet_name, index=False)


def process_dataframe_stage(df_original, index_columns, examples=False):
    # logger.info(f"Starting the process_dataframe function. Index columns: {index_columns}")

    try:
        df = df_original.copy()
        df_pivot = (
            df.pivot_table(
                index=index_columns,
                columns='_CREATED_period',
                values='NUMBER',
                aggfunc='count',
                dropna=True,
            )
            .assign(
                TOTAL=lambda x: x[x.select_dtypes(include='number').columns].sum(axis=1)
            )
            .sort_values(by='TOTAL', ascending=False)
            .assign(PCT=lambda x: 100 * np.round(x.TOTAL / x.TOTAL.sum(), 4))
        )

        if examples:
            df_pivot = df_pivot.merge(
                df.groupby(index_columns)['NUMBER'].agg(list),
                left_index=True,
                right_index=True,
            )

            # Get max length of lists in 'NUMBER' column
            max_len = (
                df_pivot['NUMBER']
                .apply(lambda x: len(x) if isinstance(x, list) else 0)
                .max()
            )
            if max_len > 5:
                df_pivot['NUMBER'] = df_pivot['NUMBER'].apply(
                    lambda x: x[:5] if isinstance(x, list) else x
                )
                # logger.info(f"Truncated lists in 'NUMBER' examples column to max length of {5}. Original max length was {max_len}.")

    except Exception as e:
        print(f"An error occurred in process_dataframe: {e}")
        # if logger:
        #     logger.error(f"An error occurred in process_dataframe: {e}")
        raise

    # logger.info("\n" + tabulate(df_pivot, headers='keys', tablefmt='psql'))
    return df_pivot


def flag_PARENT_change(df):
    """
    Flags and categorizes PARENT_INCIDENT changes using fully vectorized operations.

    Args:
        df: DataFrame with NUMBER, UPDATED, PARENT_INCIDENT columns

    Returns:
        DataFrame with additional columns:
        - '_CHANGE_IN_ROW': Boolean indicating if this row has a different value than the next
        - '_TYPE_CHANGE': Categorization of the type of change including start status or end status
        - '_TOTAL_CHANGES': Total number of changes for each NUMBER
    """
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()

    # Fill NA values with empty string
    result_df['PARENT_INCIDENT'] = result_df['PARENT_INCIDENT'].fillna('')

    # Sort the entire dataframe by NUMBER and UPDATED
    result_df = result_df.sort_values(['NUMBER', 'UPDATED'])

    # Create helper columns to identify first and last entry of each group
    result_df['_is_first'] = ~result_df['NUMBER'].duplicated(keep='first')
    result_df['_is_last'] = ~result_df['NUMBER'].duplicated(keep='last')

    # Calculate next values using shift with a condition to only shift within same NUMBER
    result_df['_next_number'] = result_df['NUMBER'].shift(-1)
    result_df['NEXT_PARENT'] = result_df['PARENT_INCIDENT'].shift(-1)
    # When next row is from a different NUMBER, use current value (no change)
    result_df.loc[result_df['NUMBER'] != result_df['_next_number'], 'NEXT_PARENT'] = (
        result_df['PARENT_INCIDENT']
    )

    # Identify rows where current != next (where changes occur)
    result_df['_CHANGE_IN_ROW'] = (
        result_df['PARENT_INCIDENT'] != result_df['NEXT_PARENT']
    )

    # Initialize _TYPE_CHANGE column
    result_df['_TYPE_CHANGE'] = ''

    # Apply type categories using conditions
    conditions = [
        # First row without parent
        (result_df['_is_first']) & (result_df['PARENT_INCIDENT'] == ''),
        # First row with parent
        (result_df['_is_first']) & (result_df['PARENT_INCIDENT'] != ''),
        # Last row without parent
        (result_df['_is_last']) & (result_df['PARENT_INCIDENT'] == ''),
        # Last row with parent
        (result_df['_is_last']) & (result_df['PARENT_INCIDENT'] != ''),
        # From null to parent
        (result_df['_CHANGE_IN_ROW'])
        & (result_df['PARENT_INCIDENT'] == '')
        & (result_df['NEXT_PARENT'] != ''),
        # From parent to null
        (result_df['_CHANGE_IN_ROW'])
        & (result_df['PARENT_INCIDENT'] != '')
        & (result_df['NEXT_PARENT'] == ''),
        # From parent to different parent
        (result_df['_CHANGE_IN_ROW'])
        & (result_df['PARENT_INCIDENT'] != '')
        & (result_df['NEXT_PARENT'] != '')
        & (result_df['PARENT_INCIDENT'] != result_df['NEXT_PARENT']),
    ]

    choices = [
        'S_NP',  # First row without parent
        'S_WP',  # First row with parent
        'E_NP',  # Last row without parent
        'E_WP',  # Last row with parent
        'F_N_T_P',  # From null to parent
        'F_P_T_N',  # From parent to null
        'F_P_T_DP',  # From parent to different parent
    ]

    result_df['_TYPE_CHANGE'] = np.select(conditions, choices, default='')

    # Calculate total changes per NUMBER
    result_df['_TOTAL_CHANGES'] = result_df['NUMBER'].map(
        result_df.groupby('NUMBER')['_CHANGE_IN_ROW'].sum()
    )

    # Clean up temporary columns
    result_df = result_df.drop(
        columns=['_is_first', '_is_last', '_next_number', 'NEXT_PARENT']
    )

    result_df['_SET_TICKET_PARENTS'] = result_df.NUMBER.map(
        df.groupby('NUMBER')['PARENT_INCIDENT'].agg(
            lambda x: ','.join(set(val for val in x if pd.notnull(val)))
        )
    )

    return result_df

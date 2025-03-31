"""Plotting utils for the vector databases benchmarking."""
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def get_times() -> dict:
    """Get the times from the json file."""
    if not os.path.exists("times.json") or os.path.getsize("times.json") == 0:
        raise FileNotFoundError("times.json file not found or empty. Run the benchmark first.")

    with open("times.json", "r") as f:
        data = json.load(f)
    return data


def get_usages():
    """
    Reads docker stats text files from the current directory, parses the CPU usage percentages,
    and returns a list of dictionaries with the extracted data.
    
    The function expects the following files to be present:
      - docker_stats_standalone.txt
      - docker_stats_es01.txt
      - docker_stats_qdrant.txt

    Each file contains header lines in the format:
      "Stats for <operation> in <database>:"
    followed by one or more lines with container data. This function extracts the operation
    and database from the header, and for each container line extracts:
      - container_name: the container's name.
      - cpu_usage: the CPU usage percentage as a float.
      - mem_usage: the memory usage percentage as a float.

    Returns:
        List[dict]: A list of dictionaries with keys:
            - "file": The source file name.
            - "operation": The operation (e.g., "search", "insert").
            - "database": The database (e.g., "standalone", "es01", "qdrant").
            - "container_name": The container's name.
            - "cpu_usage": The CPU usage percentage as a float.
            - "mem_usage": The memory usage percentage as a float.
    """
    filenames = [
        "docker_stats_standalone.txt",
        "docker_stats_es01.txt",
        "docker_stats_qdrant.txt"
    ]
    
    usages = []
    for fname in filenames:
        if not os.path.exists(fname):
            continue  # Skip files that do not exist.
        with open(fname, "r") as f:
            lines = f.readlines()
        
        current_operation = None
        current_database = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if the line is a header indicating the operation and database.
            if line.startswith("Stats for"):
                # Expected format: "Stats for <operation> in <database>:"
                parts = line.split()
                if len(parts) >= 5:
                    current_operation = parts[2]
                    # Remove trailing ':' from database
                    current_database = parts[4].rstrip(':')
                continue
            
            # Process a data line.
            parts = line.split()
            if len(parts) < 8:
                continue
            container_name = parts[1]
            cpu_usage_str = parts[2]   # e.g. "10.37%"
            mem_usage_str = parts[6]     # e.g. "0.38%"
            try:
                cpu_usage_val = float(cpu_usage_str.strip('%'))
            except Exception:
                cpu_usage_val = None
            try:
                mem_usage_val = float(mem_usage_str.strip('%'))
            except Exception:
                mem_usage_val = None
            
            usages.append({
                "file": fname,
                "operation": current_operation,
                "database": current_database,
                "container_name": container_name,
                "cpu_usage": cpu_usage_val,
                "mem_usage": mem_usage_val
            })
    return usages


def plot_times() -> None:
    """Plot the times from the json file."""
    data = get_times()

    first_db = next(iter(data.values()))
    if "search_results" in first_db:
        del first_db["search_results"]
    actions = list(first_db.keys())
    actions.remove("create_index")
    actions.append("create_index/create_collection")

    for db, times in data.items():
        if "search_results" in times:
            del times["search_results"]
        if "create_index" in times:
            times["create_index/create_collection"] = times["create_index"]
            del times["create_index"]
        if "create_collection" in times:
            times["create_index/create_collection"] = times["create_collection"]
            del times["create_collection"]

    num_dbs = len(data)
    x = np.arange(len(actions))
    width = 0.8 / num_dbs

    fig, ax = plt.subplots()

    for idx, (db, times) in enumerate(data.items()):
        values = [times[action] for action in actions]
        ax.bar(x + idx * width, values, width, label=db)

    ax.set_xlabel("Action")
    ax.set_ylabel("Time (s)")
    ax.set_title("Time taken for each action")
    ax.set_xticks(x + width * (num_dbs - 1) / 2)
    ax.set_xticklabels(actions)
    ax.legend()
    ax.grid()
    plt.savefig("times.png")
    print("Times plotted and saved to times.png")


def plot_usages():
    """
    Retrieves docker stats usage data using get_usages(), computes the average CPU usage percentage
    for each operation and database, and plots a grouped bar plot where the x-axis represents the operations
    and each group shows the average CPU usage for each database.
    
    The plot is saved as 'usages_plot.png'.
    """
    data = get_usages()
    
    if not data:
        print("No data available to plot.")
        return

    # Create sets of unique operations and databases.
    operations = sorted({entry["operation"] for entry in data if entry["operation"]})
    databases = sorted({entry["database"] for entry in data if entry["database"]})
    
    # Compute average CPU usage for each (operation, database) pair.
    avg_usage = {op: {db: [] for db in databases} for op in operations}
    for entry in data:
        op = entry["operation"]
        db = entry["database"]
        if op and db and entry["cpu_usage"] is not None:
            avg_usage[op][db].append(entry["cpu_usage"])
    
    # Replace lists with average values.
    for op in operations:
        for db in databases:
            values = avg_usage[op][db]
            if values:
                avg_usage[op][db] = sum(values) / len(values)
            else:
                avg_usage[op][db] = 0.0  # Use 0 if no data is available.
    
    # Plot grouped bar chart.
    n_ops = len(operations)
    n_dbs = len(databases)
    bar_width = 0.8 / n_dbs  # total width for group is 0.8
    indices = np.arange(n_ops)
    
    plt.figure(figsize=(10, 6))
    for i, db in enumerate(databases):
        # Compute positions for this database's bars within each group.
        pos = indices - 0.4 + i * bar_width + bar_width / 2
        # Get average CPU usage for each operation for this database.
        avg_values = [avg_usage[op][db] for op in operations]
        plt.bar(pos, avg_values, width=bar_width, label=db)
    
    plt.xlabel("Operation")
    plt.ylabel("Average CPU Usage (%)")
    plt.title("Average CPU Usage per Operation and Database")
    plt.xticks(indices, operations)
    plt.legend(title="Database")
    plt.tight_layout()
    plt.savefig("usages.png")


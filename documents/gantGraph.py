# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:08:04 2025

@author: robbe
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Define tasks and their start/end dates
tasks = [
    ("Task 1: ", "2025-02-10", "2025-02-14"),
    ("Task 2: ", "2025-02-17", "2025-02-23"),
    ("Task 3: ", "2025-02-24", "2025-03-10"),
    ("Task 4: ", "2025-03-03", "2025-03-24"),
    ("Task 5: ", "2025-03-17", "2025-04-14"),
    ("Task 6: ", "2025-04-14", "2025-04-28"),
    ("Task 7: ", "2025-04-28", "2025-05-09"),
    ("Task 8: ", "2025-05-09", "2025-05-21"),
    ("Task 9: ", "2025-04-14", "2025-06-01")
]

# Convert string dates to datetime objects
task_dates = [(task[0], datetime.strptime(task[1], "%Y-%m-%d"), datetime.strptime(task[2], "%Y-%m-%d")) for task in tasks]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))  # Format dates on X-axis
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Show weekly ticks

# Plot each task as a horizontal bar
for i, (task, start, end) in enumerate(task_dates):
    ax.barh(task, (end - start).days, left=start, color="black", edgecolor="white", height=0.9)

# Labels and formatting
ax.set_xlabel("Timeline")
ax.set_ylabel("")
ax.set_title("Thesis Calender")
plt.xticks(rotation=45)
plt.grid(linestyle="--", linewidth=0.5, alpha=0.7)
plt.gca().invert_yaxis()  # Invert Y-axis to have tasks in order

# Show the Gantt chart
plt.tight_layout()
plt.show()

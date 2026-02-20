# generate_ieee_graph.py

import matplotlib.pyplot as plt
import numpy as np

# Data (Based on your latest 200-question massive run)
categories = ['Logical Questions', 'Non-Logical Questions', 'Overall Accuracy']
llm_baseline = [60.0, 31.3, 32.0]  # Raw Llama 3.2 accuracy
logic_guard = [100.0, 31.3, 33.0]  # After LogicGuard validation

x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

# Create Plot
fig, ax = plt.subplots(figsize=(8, 5), dpi=300) # 300 DPI is required for IEEE papers

# Draw bars
rects1 = ax.bar(x - width/2, llm_baseline, width, label='LLM Baseline (Llama 3.2)', color='#d9534f', edgecolor='black')
rects2 = ax.bar(x + width/2, logic_guard, width, label='LogicGuard (Our System)', color='#5cb85c', edgecolor='black')

# Add text, labels, title, and custom x-axis tick labels
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: LLM Baseline vs. LogicGuard', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 115) # Leave space for the 100% label at the top
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add exact numbers on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# Save the plot for the IEEE paper
plt.savefig('ieee_results_chart.png', bbox_inches='tight')
print("✅ Graph saved successfully as 'ieee_results_chart.png'. Paper mein insert kar do!")
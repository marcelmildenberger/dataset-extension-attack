import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import numpy as np

# Load data
df = pd.read_csv("formatted_results.csv")

# Extract numeric dataset size (optional)
def extract_dataset_size(name):
    match = re.search(r"(\d+)k", name)
    if match:
        return int(match.group(1)) * 1000
    elif "euro" in name:
        return 26625  # placeholder
    return None

df["DatasetSize"] = df["Dataset"].apply(extract_dataset_size)
df["Overlap"] = df["Overlap"].astype(float)

# Output directory
os.makedirs("analysis/plots", exist_ok=True)
os.makedirs("analysis/tables", exist_ok=True)

# Metrics to visualize
metrics = [
    ("TrainedPrecision", "Precision"),
    ("TrainedRecall", "Recall"),
    ("TrainedF1", "F1 Score"),
    ("ReidentificationRate", "Re-identification Rate"),
]

# Baseline metrics per dataset
baseline_metrics = {
    "fakename_1k":     {"Precision": 0.2162, "Recall": 0.2476, "F1": 0.2300},
    "fakename_2k":     {"Precision": 0.2131, "Recall": 0.2452, "F1": 0.2271},
    "fakename_5k":     {"Precision": 0.2144, "Recall": 0.2470, "F1": 0.2287},
    "fakename_10k":    {"Precision": 0.2151, "Recall": 0.2467, "F1": 0.2289},
    "fakename_20k":    {"Precision": 0.2153, "Recall": 0.2473, "F1": 0.2293},
    "fakename_50k":    {"Precision": 0.2151, "Recall": 0.2463, "F1": 0.2288},
    "titanic_full":    {"Precision": 0.2468, "Recall": 0.3770, "F1": 0.2896},
    "euro_person":     {"Precision": 0.2197, "Recall": 0.2446, "F1": 0.2306}
}

encoding_map = {
    "BloomFilter": "Bloom Filter",
    "TabMinHash": "Tabulation Minhash",
    "TwoStepHash": "Two-Step Hash"
}

# Create plots for fixed dataset and encoding combinations
# Loop through each (Dataset, Encoding) pair
for (dataset, encoding), group in df.groupby(["Dataset", "Encoding"]):
    # Create subfolder for this encoding if it doesn't exist
    plot_dir = f"analysis/plots/{encoding}"
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))

    dataset_label = dataset.replace(".tsv", "").replace("_", " ")
    encoding_label = encoding_map[encoding]
    title = f"{encoding_label} — {dataset_label}"
    fig.suptitle(title, fontsize=16)

    dataset_key = dataset.replace(".tsv", "")

    # Regular metric plots
    for ax, (metric_key, metric_label) in zip(axes.flat[:6], metrics):
        # Handle re-identification rate differently (convert to percentage)
        if metric_key == "ReidentificationRate":
            # Create a copy of the data with percentage conversion
            plot_data = group.copy()
            plot_data[metric_key] = plot_data[metric_key] * 100
            y_label = "Re-identification Rate (%)"
            # Dynamic scaling based on actual values
            max_rate = plot_data[metric_key].max()
            if max_rate > 0:
                y_lim = (0, min(100, max_rate * 1.1))  # Add 10% margin, cap at 100%
            else:
                y_lim = (0, 1)  # Default range when max_rate is 0
        else:
            plot_data = group
            y_label = metric_label
            y_lim = None

        sns.lineplot(
            data=plot_data,
            x="Overlap",
            y=metric_key,
            hue="DropFrom",
            marker="o",
            ax=ax
        )
        ax.set_title(metric_label)
        ax.set_xlabel("Overlap")
        ax.set_ylabel(y_label)
        if y_lim:
            ax.set_ylim(y_lim)
        ax.grid(True)

        # Baselines
        if dataset_key in baseline_metrics:
            if metric_key == "TrainedF1":
                ax.axhline(y=baseline_metrics[dataset_key]["F1"], linestyle="--", color="gray", label="Baseline F1")
                ax.legend()
            elif metric_key == "TrainedRecall":
                ax.axhline(y=baseline_metrics[dataset_key]["Recall"], linestyle="--", color="gray", label="Baseline Recall")
                ax.legend()
            elif metric_key == "TrainedPrecision":
                ax.axhline(y=baseline_metrics[dataset_key]["Precision"], linestyle="--", color="gray", label="Baseline Precision")
                ax.legend()

    # Two subplots for Re-ID Comparison by DropFrom
    melted = group.melt(
        id_vars=["Overlap", "DropFrom"],
        value_vars=["ReidentificationRateFuzzy", "ReidentificationRateGreedy", "ReidentificationRate"],
        var_name="Method",
        value_name="Rate"
    )
    method_map = {
        "ReidentificationRateFuzzy": "Fuzzy",
        "ReidentificationRateGreedy": "Greedy",
        "ReidentificationRate": "Combined"
    }
    melted["Method"] = melted["Method"].map(method_map)
    melted["Rate"] *= 100

    # Plot for DropFrom = Eve
    ax = axes.flat[4]
    subset_eve = melted[melted["DropFrom"] == "Eve"]
    sns.lineplot(
        data=subset_eve,
        x="Overlap",
        y="Rate",
        hue="Method",
        marker="o",
        ax=ax
    )
    ax.set_title("Re-identification Rate (DropFrom = Eve)")
    ax.set_ylabel("Re-identification Rate (%)")
    # Dynamic scaling based on actual values
    max_rate = subset_eve["Rate"].max()
    if max_rate > 0:
        ax.set_ylim(0, min(100, max_rate * 1.1))  # Add 10% margin, cap at 100%
    else:
        ax.set_ylim(0, 1)  # Default range when max_rate is 0
    ax.grid(True)

    # Plot for DropFrom = Both
    ax = axes.flat[5]
    subset_both = melted[melted["DropFrom"] == "Both"]
    sns.lineplot(
        data=subset_both,
        x="Overlap",
        y="Rate",
        hue="Method",
        marker="o",
        ax=ax
    )
    ax.set_title("Re-identification Rate (DropFrom = Both)")
    ax.set_ylabel("Re-identification Rate (%)")
    # Dynamic scaling based on actual values
    max_rate = subset_both["Rate"].max()
    if max_rate > 0:
        ax.set_ylim(0, min(100, max_rate * 1.1))  # Add 10% margin, cap at 100%
    else:
        ax.set_ylim(0, 1)  # Default range when max_rate is 0
    ax.grid(True)

    # Scatter plot for TrainedF1 vs. HypOpF1
    ax = axes.flat[6]
    sns.scatterplot(
        data=group,
        x="HypOpF1",
        y="TrainedF1",
        s=60,
        color="steelblue",
        ax=ax
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="x = y")
    ax.set_title("Trained vs. Hyperparameter Optimization F1")
    ax.set_xlabel("Hyperparameter Op. F1")
    ax.set_ylabel("Trained F1")
    # Dynamic axis limits
    x_min, x_max = group["HypOpF1"].min(), group["HypOpF1"].max()
    y_min, y_max = group["TrainedF1"].min(), group["TrainedF1"].max()
    margin = 0.05
    ax.set_xlim(max(0, x_min - margin), min(1, x_max + margin))
    ax.set_ylim(max(0, y_min - margin), min(1, y_max + margin))
    ax.grid(True)
    ax.legend(loc="lower right", fontsize="small")

    # Privacy-Utility Trade-off plot with dynamic axis limits
    ax = axes.flat[7]
    x_min, x_max = group["TrainedF1"].min(), group["TrainedF1"].max()
    y_min, y_max = (group["ReidentificationRate"] * 100).min(), (group["ReidentificationRate"] * 100).max()
    margin_x = 0.05
    margin_y = 2
    x = group["TrainedF1"]
    y = group["ReidentificationRate"] * 100
    ax.scatter(x, y, color="purple", s=60, alpha=0.7)
    # Linear regression fit
    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, color="orange", linestyle="--", label="Best Fit")
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Re-identification Rate (%)")
    ax.set_title("Re-identification Rate vs. F1 Score")
    ax.set_xlim(max(0, x_min - margin_x), min(1, x_max + margin_x))
    ax.set_ylim(max(0, y_min - margin_y), min(100, y_max + margin_y))
    ax.grid(True)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"{plot_dir}/{encoding}_{dataset_label.replace(' ', '_')}_metrics.png"
    plt.savefig(filename, dpi=300)
    plt.close()


# %%
# Select relevant columns for the summary
summary_df = df[[
    "Dataset", "Encoding", "Overlap", "DropFrom",
    "ReidentificationRateFuzzy", "ReidentificationRateGreedy", "ReidentificationRate"
]]

# Rename columns for clarity
summary_df = summary_df.rename(columns={
    "ReidentificationRateFuzzy": "Fuzzy Rate",
    "ReidentificationRateGreedy": "Greedy Rate",
    "ReidentificationRate": "Combined Rate"
})

# Sort for readability
summary_df = summary_df.sort_values(by=["Dataset", "Encoding", "Overlap", "DropFrom"])

# Export to CSV
summary_df.to_csv("analysis/tables/reidentification_summary.csv", index=False)

# Ensure time columns are numeric
time_cols = [
    "GraphMatchingAttackTime", "HyperparameterOptimizationTime",
    "ModelTrainingTime", "ApplicationtoEncodedDataTime",
    "RefinementandReconstructionTime", "TotalRuntime"
]
df[time_cols] = df[time_cols].apply(pd.to_numeric, errors="coerce")

# # 1. Overall Runtime Breakdown
# avg_runtime = df[time_cols].mean().sort_values(ascending=False)

# plt.figure(figsize=(10, 6))
# sns.barplot(x=avg_runtime.values, y=avg_runtime.index)
# plt.title("Average Runtime Breakdown (All Experiments)")
# plt.xlabel("Time in Minutes")
# plt.tight_layout()
# plt.savefig("analysis/plots/dea_runtime_overall.png", dpi=300)
# plt.close()

# # 3 . Runtime Breakdown: Average vs. Max
# avg_times = df[time_cols].mean()
# max_times = df[time_cols].max()
# runtime_df = pd.DataFrame({
#     "Average Time (m)": avg_times,
#     "Max Time (m)": max_times
# }).sort_values("Average Time (m)", ascending=False)

# # 2. Runtime by Encoding Scheme
# encoding_runtime = df.groupby("Encoding")[time_cols].mean().T

# encoding_runtime.plot(kind="bar", figsize=(12, 6))
# plt.title("Average Runtime per Step by Encoding Scheme")
# plt.ylabel("Time in Minutes")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("analysis/plots/dea_runtime_by_encoding.png", dpi=300)
# plt.close()


# Set consistent color palette and styling
sns.set(style="whitegrid")



# Get sorted list of datasets
datasets = sorted(df["Dataset"].unique())

# Create subplots: 3 columns, enough rows
n_cols = 3
n_rows = (len(datasets) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
axes = axes.flatten()

for i, dataset in enumerate(datasets):
    ax = axes[i]
    subset = df[df["Dataset"] == dataset]

    # Group by encoding and DropFrom, averaging across overlap
    grouped = (
        subset.groupby(["Encoding", "DropFrom"])["TrainedF1"]
        .mean()
        .reset_index()
    )

    # Add a new column for the encoding label
    grouped["EncodingLabel"] = grouped["Encoding"].map(encoding_map)

    sns.barplot(
        data=grouped,
        x="EncodingLabel",
        y="TrainedF1",
        hue="DropFrom",
        ax=ax,
        errorbar="sd"
    )
    ax.set_title(dataset.replace(".tsv", "").replace("_", " "))
    ax.set_ylabel("F1 / Dice Score")
    ax.set_xlabel("Encoding")
    ax.set_ylim(0, 1)
    ax.legend(title="DropFromtegy", loc="upper right", fontsize="small")

# Remove empty axes if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("DEA Performance by Encoding, Aggregated over Overlap", fontsize=16, y=1.02)
plt.savefig("analysis/plots/dea_encoding_comparison_all_datasets.png", dpi=300, bbox_inches="tight")
plt.close()


# DEA Encoding Comparison: Re-identification Rate instead of F1
n_cols = 3
n_rows = (len(datasets) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
axes = axes.flatten()

for i, dataset in enumerate(datasets):
    ax = axes[i]
    subset = df[df["Dataset"] == dataset]

    # Group by encoding and DropFrom, averaging across overlap
    grouped = (
        subset.groupby(["Encoding", "DropFrom"])["ReidentificationRate"]
        .mean()
        .reset_index()
    )
    grouped["EncodingLabel"] = grouped["Encoding"].map(encoding_map)
    grouped["ReidentificationRate"] = grouped["ReidentificationRate"] * 100

    sns.barplot(
        data=grouped,
        x="EncodingLabel",
        y="ReidentificationRate",
        hue="DropFrom",
        ax=ax,
        errorbar="sd"
    )
    ax.set_title(dataset.replace(".tsv", "").replace("_", " "))
    ax.set_ylabel("Re-identification Rate (%)")
    ax.set_xlabel("Encoding")
    ax.set_ylim(0, 100)
    ax.legend(title="DropFromtegy", loc="upper right", fontsize="small")

# Remove empty axes if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("DEA Re-identification Rate by Encoding, Aggregated over Overlap", fontsize=16, y=1.02)
plt.savefig("analysis/plots/dea_encoding_reidrate_comparison_all_datasets.png", dpi=300, bbox_inches="tight")
plt.close()


# Map encoding names for readability
df["EncodingLabel"] = df["Encoding"].map(encoding_map)

# If Overlap is a float (e.g., 0.2), convert to percentage string for axis labels
df["OverlapLabel"] = (df["Overlap"] * 100).astype(int).astype(str) + "%"

# Create line charts for re-identification rate and F1 score by encoding scheme
# Only consider specific overlaps
desired_overlaps = [0.2, 0.4, 0.6, 0.8]
line_data = df[df["Overlap"].isin(desired_overlaps)].groupby(["EncodingLabel", "Overlap"])[["ReidentificationRate", "TrainedF1"]].mean().reset_index()

# Convert re-identification rate to percentage
line_data["ReidentificationRate"] = line_data["ReidentificationRate"] * 100

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Re-identification Rate vs Overlap
for encoding in line_data["EncodingLabel"].unique():
    subset = line_data[line_data["EncodingLabel"] == encoding]
    ax1.plot(subset["Overlap"], subset["ReidentificationRate"],
             marker="o", linewidth=2, markersize=6, label=encoding)

ax1.set_xlabel("Overlap")
ax1.set_ylabel("Re-identification Rate (%)")
ax1.set_title("Re-identification Rate by Encoding Scheme", fontsize=14, weight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 1)

# Plot 2: F1 Score vs Overlap
for encoding in line_data["EncodingLabel"].unique():
    subset = line_data[line_data["EncodingLabel"] == encoding]
    ax2.plot(subset["Overlap"], subset["TrainedF1"],
             marker="s", linewidth=2, markersize=6, label=encoding)

ax2.set_xlabel("Overlap")
ax2.set_ylabel("F1 Score")
ax2.set_title("F1 Score by Encoding Scheme", fontsize=14, weight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.savefig("analysis/plots/dea_encoding_comparison_line_charts.png", dpi=300, bbox_inches="tight")
plt.close()


# Executive Summary for DEA

summary_lines = []
summary_lines.append("DATASET EXTENSION ATTACK (DEA) - EXECUTIVE SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append("")

# General stats
total_experiments = len(df)
encoding_schemes_tested = df["Encoding"].nunique()
datasets_analyzed = df["Dataset"].nunique()
avg_reid = df["ReidentificationRate"].mean() * 100
max_reid = df["ReidentificationRate"].max() * 100
avg_f1 = df["TrainedF1"].mean()
max_f1 = df["TrainedF1"].max()
avg_runtime = df["TotalRuntime"].mean()

summary_lines.append(f"Total Experiments: {total_experiments}")
summary_lines.append(f"Encoding Schemes Tested: {encoding_schemes_tested}")
summary_lines.append(f"Datasets Analyzed: {datasets_analyzed}")
summary_lines.append(f"Average Re-identification Rate: {avg_reid:.2f}%")
summary_lines.append(f"Maximum Re-identification Rate: {max_reid:.2f}%")
summary_lines.append(f"Average F1 Score: {avg_f1:.3f}")
summary_lines.append(f"Maximum F1 Score: {max_f1:.3f}")
summary_lines.append("")

# Per-encoding stats
summary_lines.append("ENCODING SCHEME PERFORMANCE:")
summary_lines.append("-" * 30)

for encoding, label in encoding_map.items():
    subset = df[df["Encoding"] == encoding]
    avg_reid_enc = subset["ReidentificationRate"].mean() * 100
    avg_f1_enc = subset["TrainedF1"].mean()
    n_exp = len(subset)
    summary_lines.append(f"{label}:")
    summary_lines.append(f"  - Avg Re-ID Rate: {avg_reid_enc:.2f}%")
    summary_lines.append(f"  - Avg F1 Score: {avg_f1_enc:.3f}")
    summary_lines.append(f"  - Experiments: {n_exp}")
    summary_lines.append("")

# Write to file
summary_path = "analysis/dea_executive_summary.txt"
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

print(f"Executive summary written to {summary_path}")


# ================= Neural Network Architecture Parameter Distributions =====================
# For each encoding scheme, generate a single PNG/PDF with subplots showing the distribution of all architecture parameters used.

arch_cols = [
    "Encoding", "EncodingLabel", "HypOpNumLayers", "HypOpHiddenSize", "HypOpDropout", "HypOpActivation", "HypOpOptimizer", "HypOpThreshold", "HypOpLRScheduler", "HypOpBatchSize", "HypOpEpochs"
]
arch_df = df[arch_cols].copy()
arch_df = arch_df.dropna(subset=["HypOpNumLayers", "HypOpHiddenSize", "HypOpDropout", "HypOpActivation", "HypOpOptimizer", "HypOpThreshold", "HypOpLRScheduler", "HypOpBatchSize", "HypOpEpochs"])
arch_df["HypOpNumLayers"] = pd.to_numeric(arch_df["HypOpNumLayers"], errors="coerce")
arch_df["HypOpHiddenSize"] = pd.to_numeric(arch_df["HypOpHiddenSize"], errors="coerce")
arch_df["HypOpDropout"] = pd.to_numeric(arch_df["HypOpDropout"], errors="coerce")
arch_df["HypOpThreshold"] = pd.to_numeric(arch_df["HypOpThreshold"], errors="coerce")
arch_df["HypOpBatchSize"] = pd.to_numeric(arch_df["HypOpBatchSize"], errors="coerce")
arch_df["HypOpEpochs"] = pd.to_numeric(arch_df["HypOpEpochs"], errors="coerce")

# Clean optimizer and LR scheduler name to just the name (e.g., AdamW from "{'name': 'AdamW', ...}")
def extract_name_from_dict(val):
    if isinstance(val, str):
        match = re.search(r"'name': ?'([^']+)'", val)
        if match:
            return match.group(1)
        # fallback: if just a name
        return val.strip().split()[0].replace('"','').replace("'","")
    return val
arch_df["OptimizerName"] = arch_df["HypOpOptimizer"].apply(extract_name_from_dict)
arch_df["LRSchedulerName"] = arch_df["HypOpLRScheduler"].apply(extract_name_from_dict)

for encoding, group in arch_df.groupby(["Encoding", "EncodingLabel"]):
    enc, enc_label = encoding
    # Create subfolder for this encoding if it doesn't exist
    plot_dir = f"analysis/plots/{enc}"
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(22, 14))
    axes = axes.flatten()
    # NumLayers: bar plot of counts
    ax = axes[0]
    counts = group["HypOpNumLayers"].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    ax.set_title("# Layers")
    ax.set_xlabel("Num Layers")
    ax.set_ylabel("Count")
    # HiddenSize: bar plot of counts
    ax = axes[1]
    counts = group["HypOpHiddenSize"].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color="lightgreen", edgecolor="black")
    ax.set_title("Hidden Size")
    ax.set_xlabel("Hidden Size")
    ax.set_ylabel("Count")
    # Dropout: histogram with mean/std (if not all integer)
    ax = axes[2]
    vals = group["HypOpDropout"].dropna()
    if (vals % 1 != 0).any():
        mean = vals.mean()
        std = vals.std()
        ax.hist(vals, bins=10, color="plum", edgecolor="black")
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
        ax.axvline(mean+std, color="orange", linestyle=":", label=f"Std: {std:.2f}")
        ax.axvline(mean-std, color="orange", linestyle=":")
        ax.legend()
    else:
        counts = vals.value_counts().sort_index()
        counts.plot(kind="bar", ax=ax, color="plum", edgecolor="black")
    ax.set_title("Dropout")
    ax.set_xlabel("Dropout")
    ax.set_ylabel("Count")
    # Categorical: Activation
    ax = axes[3]
    act_counts = group["HypOpActivation"].value_counts()
    act_counts.plot(kind="bar", ax=ax, color="cornflowerblue", edgecolor="black")
    ax.set_title("Activation Function")
    ax.set_xlabel("Activation")
    ax.set_ylabel("Count")
    # Categorical: Optimizer (just name)
    ax = axes[4]
    opt_counts = group["OptimizerName"].value_counts()
    opt_counts.plot(kind="bar", ax=ax, color="salmon", edgecolor="black")
    ax.set_title("Optimizer")
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Count")
    # Threshold: histogram with mean/std (if not all integer)
    ax = axes[5]
    vals = group["HypOpThreshold"].dropna()
    if (vals % 1 != 0).any():
        mean = vals.mean()
        std = vals.std()
        ax.hist(vals, bins=10, color="gold", edgecolor="black")
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
        ax.axvline(mean+std, color="orange", linestyle=":", label=f"Std: {std:.2f}")
        ax.axvline(mean-std, color="orange", linestyle=":")
        ax.legend()
    else:
        counts = vals.value_counts().sort_index()
        counts.plot(kind="bar", ax=ax, color="gold", edgecolor="black")
    ax.set_title("Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")
    # Categorical: LR Scheduler (just name)
    ax = axes[6]
    lr_counts = group["LRSchedulerName"].value_counts()
    lr_counts.plot(kind="bar", ax=ax, color="mediumorchid", edgecolor="black")
    ax.set_title("LR Scheduler")
    ax.set_xlabel("LR Scheduler")
    ax.set_ylabel("Count")
    # Batch Size: bar plot of counts (no mean/std)
    ax = axes[7]
    counts = group["HypOpBatchSize"].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color="deepskyblue", edgecolor="black")
    ax.set_title("Batch Size")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Count")
    # Epochs: bar plot of counts with mean/std lines
    ax = axes[8]
    counts = group["HypOpEpochs"].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color="gray", edgecolor="black")
    mean = group["HypOpEpochs"].mean()
    std = group["HypOpEpochs"].std()
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
    ax.axvline(mean+std, color="orange", linestyle=":", label=f"Std: {std:.2f}")
    ax.axvline(mean-std, color="orange", linestyle=":")
    ax.set_title("Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Count")
    ax.legend()
    fig.suptitle(f"Neural Network Architecture Parameters — {enc_label}", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{plot_dir}/{enc}_architecture.png", dpi=300)
    plt.close()


# ================= Find Duplicate Experiment Settings =====================
# Find (Encoding, Dataset, DropFrom, Overlap) settings that occur more than once and store the experiment folder names for manual inspection.
duplicate_groups = (
    df.groupby(["Encoding", "Dataset", "DropFrom", "Overlap"])
    .filter(lambda g: len(g) > 1)
)
if not duplicate_groups.empty:
    dupe_summary = (
        duplicate_groups.groupby(["Encoding", "Dataset", "DropFrom", "Overlap"])
        .agg({"ExperimentFolder": lambda x: ";".join(x)})
        .reset_index()
        .rename(columns={"ExperimentFolder": "ExperimentFolders"})
    )
    os.makedirs("analysis/tables", exist_ok=True)
    dupe_summary.to_csv("analysis/tables/duplicate_experiments.csv", index=False)
else:
    # If no duplicates, write an empty file with header
    pd.DataFrame(columns=["Encoding", "Dataset", "DropFrom", "Overlap", "ExperimentFolders"]).to_csv("analysis/tables/duplicate_experiments.csv", index=False)


# ================= Overlap vs. Re-identification Rate and F1 Score per Encoding (Aggregated over DropFrom) =====================
desired_overlaps = [0.2, 0.4, 0.6, 0.8]
for encoding, enc_label in encoding_map.items():
    plot_dir = f"analysis/plots/{encoding}"
    os.makedirs(plot_dir, exist_ok=True)
    subset = df[(df["Encoding"] == encoding) & (df["Overlap"].isin(desired_overlaps))]
    # Aggregate over DropFrom
    agg = subset.groupby(["Dataset", "Overlap"]).agg({
        "ReidentificationRate": "mean",
        "TrainedF1": "mean"
    }).reset_index()
    plt.figure(figsize=(14, 6))
    # Re-identification Rate plot
    plt.subplot(1, 2, 1)
    for dataset in agg["Dataset"].unique():
        data = agg[agg["Dataset"] == dataset]
        plt.plot(data["Overlap"], data["ReidentificationRate"] * 100, marker="o", label=dataset.replace(".tsv", "").replace("_", " "))
    plt.xlabel("Overlap")
    plt.ylabel("Re-identification Rate (%)")
    plt.title(f"{enc_label}: Overlap vs. Re-identification Rate (mean over DropFrom)")
    # Only add legend if there are datasets to show
    if len(agg["Dataset"].unique()) > 0:
        plt.legend(title="Dataset", fontsize="small")
    plt.grid(True)
    # F1 Score plot
    plt.subplot(1, 2, 2)
    for dataset in agg["Dataset"].unique():
        data = agg[agg["Dataset"] == dataset]
        plt.plot(data["Overlap"], data["TrainedF1"], marker="o", label=dataset.replace(".tsv", "").replace("_", " "))
    plt.xlabel("Overlap")
    plt.ylabel("F1 Score")
    plt.title(f"{enc_label}: Overlap vs. F1 Score (mean over DropFrom)")
    # Only add legend if there are datasets to show
    if len(agg["Dataset"].unique()) > 0:
        plt.legend(title="Dataset", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{encoding}_overlap_summary.png", dpi=300)
    plt.close()


# ================= Check for Missing Experiment Combinations =====================
print("\n" + "="*80)
print("CHECKING FOR MISSING EXPERIMENT COMBINATIONS")
print("="*80)

# Define expected combinations based on experiment_setup.py
expected_encodings = ["TabMinHash", "TwoStepHash", "BloomFilter"]
expected_datasets = ["titanic_full.tsv", "fakename_1k.tsv", "fakename_2k.tsv", "fakename_5k.tsv", "fakename_10k.tsv", "fakename_20k.tsv", "fakename_50k.tsv", "euro_person.tsv"]
expected_drop_from = ["Eve", "Both"]
expected_overlaps = [0.2, 0.4, 0.6, 0.8]

# Create all expected combinations
expected_combinations = []
for dataset in expected_datasets:
    for encoding in expected_encodings:
        for drop_from in expected_drop_from:
            for overlap in expected_overlaps:
                # Special case: BloomFilter has EveAlgo = encoding, others have EveAlgo = "None"
                if encoding == "BloomFilter":
                    expected_combinations.append((dataset, encoding, drop_from, overlap))
                else:
                    expected_combinations.append((dataset, encoding, drop_from, overlap))

# Get actual combinations from results
actual_combinations = []
for _, row in df.iterrows():
    actual_combinations.append((row["Dataset"], row["Encoding"], row["DropFrom"], row["Overlap"]))

# Convert to sets for comparison
expected_set = set(expected_combinations)
actual_set = set(actual_combinations)

# Find missing combinations
missing_combinations = expected_set - actual_set
extra_combinations = actual_set - expected_set

# Create summary report
summary_lines = []
summary_lines.append("EXPERIMENT COVERAGE ANALYSIS")
summary_lines.append("=" * 50)
summary_lines.append(f"Expected combinations: {len(expected_combinations)}")
summary_lines.append(f"Actual combinations: {len(actual_combinations)}")
summary_lines.append(f"Missing combinations: {len(missing_combinations)}")
summary_lines.append(f"Extra combinations: {len(extra_combinations)}")
summary_lines.append(f"Coverage: {len(actual_combinations)/len(expected_combinations)*100:.1f}%")
summary_lines.append("")

if missing_combinations:
    summary_lines.append("MISSING COMBINATIONS:")
    summary_lines.append("-" * 25)
    for dataset, encoding, drop_from, overlap in sorted(missing_combinations):
        summary_lines.append(f"  {dataset} | {encoding} | {drop_from} | {overlap}")
    summary_lines.append("")
else:
    summary_lines.append("✅ ALL EXPECTED COMBINATIONS COMPLETED!")
    summary_lines.append("")

if extra_combinations:
    summary_lines.append("EXTRA COMBINATIONS (not in original plan):")
    summary_lines.append("-" * 40)
    for dataset, encoding, drop_from, overlap in sorted(extra_combinations):
        summary_lines.append(f"  {dataset} | {encoding} | {drop_from} | {overlap}")
    summary_lines.append("")

# Detailed breakdown by encoding
summary_lines.append("BREAKDOWN BY ENCODING:")
summary_lines.append("-" * 25)
for encoding in expected_encodings:
    expected_for_encoding = len([c for c in expected_combinations if c[1] == encoding])
    actual_for_encoding = len([c for c in actual_combinations if c[1] == encoding])
    missing_for_encoding = len([c for c in missing_combinations if c[1] == encoding])
    coverage = actual_for_encoding / expected_for_encoding * 100
    summary_lines.append(f"  {encoding}: {actual_for_encoding}/{expected_for_encoding} ({coverage:.1f}%) - {missing_for_encoding} missing")

summary_lines.append("")

# Detailed breakdown by dataset
summary_lines.append("BREAKDOWN BY DATASET:")
summary_lines.append("-" * 25)
for dataset in expected_datasets:
    expected_for_dataset = len([c for c in expected_combinations if c[0] == dataset])
    actual_for_dataset = len([c for c in actual_combinations if c[0] == dataset])
    missing_for_dataset = len([c for c in missing_combinations if c[0] == dataset])
    coverage = actual_for_dataset / expected_for_dataset * 100
    summary_lines.append(f"  {dataset}: {actual_for_dataset}/{expected_for_dataset} ({coverage:.1f}%) - {missing_for_dataset} missing")

summary_lines.append("")

# Detailed breakdown by overlap
summary_lines.append("BREAKDOWN BY OVERLAP:")
summary_lines.append("-" * 25)
for overlap in expected_overlaps:
    expected_for_overlap = len([c for c in expected_combinations if c[3] == overlap])
    actual_for_overlap = len([c for c in actual_combinations if c[3] == overlap])
    missing_for_overlap = len([c for c in missing_combinations if c[3] == overlap])
    coverage = actual_for_overlap / expected_for_overlap * 100
    summary_lines.append(f"  {overlap}: {actual_for_overlap}/{expected_for_overlap} ({coverage:.1f}%) - {missing_for_overlap} missing")

# Write detailed report to file
os.makedirs("analysis/tables", exist_ok=True)
coverage_report_path = "analysis/tables/experiment_coverage_report.txt"
with open(coverage_report_path, "w") as f:
    f.write("\n".join(summary_lines))

# Also print to console
print("\n".join(summary_lines))
print(f"\nDetailed coverage report saved to: {coverage_report_path}")

# Create a CSV with missing combinations for easy reference
if missing_combinations:
    missing_df = pd.DataFrame(list(missing_combinations),
                             columns=["Dataset", "Encoding", "DropFrom", "Overlap"])
    missing_df.to_csv("analysis/tables/missing_experiments.csv", index=False)
    print(f"Missing experiments list saved to: analysis/tables/missing_experiments.csv")

# Create a CSV with all expected vs actual combinations
coverage_df = pd.DataFrame({
    "Dataset": [c[0] for c in expected_combinations],
    "Encoding": [c[1] for c in expected_combinations],
    "DropFrom": [c[2] for c in expected_combinations],
    "Overlap": [c[3] for c in expected_combinations],
    "Status": ["Completed" if c in actual_set else "Missing" for c in expected_combinations]
})
coverage_df.to_csv("analysis/tables/experiment_coverage_matrix.csv", index=False)
print(f"Complete coverage matrix saved to: analysis/tables/experiment_coverage_matrix.csv")

print("\n" + "="*80)
print("EXPERIMENT COVERAGE ANALYSIS COMPLETE")
print("="*80)



import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

def make_graphs(evaluation, var_name):

    names_alg = list(evaluation.keys())
    all_keys = list(evaluation[names_alg[0]].keys())
    num_folds = len(all_keys)

    # Prepare data for grouped bar chart
    bar_width = 0.05
    index = np.arange(num_folds) + 0.5

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, alg in enumerate(names_alg):
        means = [np.mean(evaluation[alg][fold]) for fold in all_keys]
        std_devs = [np.std(evaluation[alg][fold]) for fold in all_keys]

        bars = ax.bar(index + i * bar_width, means, bar_width, label=alg, yerr=std_devs)


    ax.set_xlabel('Folds')
    ax.set_ylabel(var_name)
    ax.set_title(str(var_name)+' for different recommder systems')
    ax.set_xticks(index + ((len(names_alg) - 1) / 2) * bar_width)
    ax.set_xticklabels(all_keys)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()

    plt.savefig("/Users/elisestijger/Desktop/"+str(var_name) +"_implicit.png")


    return plt

def make_bars(data, metrics , data2):

    # Extracting fold values and their corresponding precision values for each category
    fold_labels = list(data[str(metrics) +' before active learning'].keys())
    precision_before = list(data[str(metrics) +' before active learning'].values())
    precision_after = list(data[str(metrics) +' after active learning'].values())
    precision_after2 = list(data2[str(metrics) +' after active learning'].values())

    all_values = []  # List to collect all values for determining the min and max

    for key in data[str(metrics)+' before active learning']:
        all_values.append(data[str(metrics)+' before active learning'][key])
    for key in data[str(metrics)+' after active learning']:
        all_values.append(data[str(metrics)+' after active learning'][key])
    for key in data2[str(metrics)+' after active learning']:
        all_values.append(data2[str(metrics)+' after active learning'][key])

    min_value = min(all_values)
    max_value = max(all_values)
    buffer = 0.1  # You can adjust this value

    data_range = max_value - min_value
    proportional_buffer = buffer * data_range
    min_value -= proportional_buffer
    max_value += proportional_buffer
    bar_width = 0.30       

    # and this was good:
    # x = range(len(fold_labels))

    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size

    x = np.arange(len(fold_labels))  # Generate array of values for the x-axis

    bar1 = ax.bar(x - bar_width, precision_before, width=bar_width, label='Before active learning')
    bar2 = ax.bar(x, precision_after, width=bar_width, label='After sampling active learning')
    bar3 = ax.bar(x + bar_width, precision_after2, width=bar_width, label='After sampling active learning in 4 batches')

    ax.set_xlabel('Folds')
    ax.set_ylabel(str(metrics))
    ax.set_title(str(metrics) + ' before and after active learning with sampling and sampling in batches')
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.legend()

    # Set the y-axis limits to emphasize the differences
    # ax.set_ylim(0.189, 0.198)
    ax.set_ylim(min_value, max_value)

    for i, rect in enumerate(bar1 + bar2 + bar3):
        height = rect.get_height()
        if(metrics != 'correct counts'):
            ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom')
        else:
            ax.text(rect.get_x() + rect.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()

    plt.savefig("/Users/elisestijger/Desktop/"+str(metrics) +"_batches_sampling.png")

    return plt


def make_boxplots (evaluation):

    # CHANGE:
    labels = list(evaluation['ndcg before sampling active learning'].keys())
    precision_before = list(evaluation['ndcg before sampling active learning'].values())
    precision_after = list(evaluation['ndcg after sampling active learning'].values())

    # Create the box plots
    fig, ax = plt.subplots(figsize=(8, 10))

    box_data = [precision_before, precision_after]
    ax.boxplot(box_data, labels=['Before Sampling', 'After Sampling'])

    ax.set_xlabel('Active learning')
    ax.set_ylabel('Ndcg')
    ax.set_title('Precision Before and After Sampling (Active Learning)')

    plt.show()

    return plt 


def make_graphs_together(evaluations):
    all_keys = []
    all_values = []
    all_total_values = []
    names_alg = []

    # Loop through each dictionary in the list of evaluations
    for evaluation in evaluations:
        for outer_key, inner_dict in evaluation.items():
            names_alg.append(outer_key)
            for value in inner_dict.values():
                all_values.append(value)
            all_total_values.append(all_values)
            all_values = []
            for key in inner_dict.keys():
                all_keys.append(key)

    x_values = [f"Fold {i}" for i in range(1, len(all_total_values[0]) + 1)]
    fig = plt.figure(1)
    for i, sublist in enumerate(all_total_values, start=1):
        plt.plot(x_values, sublist, label=names_alg[i - 1])

    # Label the axes
    plt.xlabel('Folds')
    plt.ylabel('count')

    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
# END 

    return plt


def learningcurve(evaluation, metrics):

    fig = plt.figure()

    for fold, counts in evaluation[str(metrics)+' before active learning'].items():
        x_values = [evaluation['trainsize'][0] + i for i in range(len(evaluation[str(metrics)+' after active learning'][fold]) + 1)]
        y_values = [counts] + evaluation[str(metrics)+' after active learning'][fold]

        plt.plot(x_values, y_values, label=f' {fold}')

    # Label the axes
    plt.xlabel('Training Size')     
    plt.ylabel(str(metrics))

    plt.title(str(metrics)+' per added item with random active learning')
    # Add a legend
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0))

    plt.tight_layout()

    plt.savefig('/Users/elisestijger/Desktop/final plots active leaning/learning curve/random/random_'+str(metrics)+'.png')

    return plt

def learningcurve_together(evaluations, metrics):
    fig = plt.figure()

    labels = ["Sampling", "Sampling in batches"]

    i = 0
    for evaluation in evaluations:
        for fold, counts in evaluation[str(metrics)+' before active learning'].items():
            x_values = [evaluation['trainsize'][0] + i for i in range(len(evaluation[str(metrics)+' after active learning'][fold]) + 1)]
            y_values = [counts] + evaluation[str(metrics)+' after active learning'][fold]
            plt.plot(x_values, y_values, label=f'{labels[i]} - {fold}')
        i = i + 1


    # Label the axes

    plt.xlabel('Training Size')     
    plt.ylabel(str(metrics))


    # Add a legend
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0))

    plt.tight_layout()
    plt.savefig('/Users/elisestijger/Desktop/sample_batches_'+str(metrics)+'.png')

    return plt



def boxplots2(data):

    # Calculate the mean values for 'ndcg after sampling active learning'
    mean_ndcg_after = [sum(data['ndcg after sampling active learning'][fold]) / len(data['ndcg after sampling active learning'][fold]) for fold in data['ndcg after sampling active learning']]

    # Create bar plots
    fig, ax = plt.subplots(figsize=(10, 6))

    # X-axis labels (fold names)
    x_labels = list(data['ndcg before sampling active learning'].keys())

    # Y-axis data
    y_ndcg_before = list(data['ndcg before sampling active learning'].values())
    y_ndcg_after = mean_ndcg_after  # Use the calculated mean values

    # Bar width
    bar_width = 0.35

    # X-axis positions
    x_positions = range(len(x_labels))

    # Create bars for 'ndcg before sampling active learning'
    ax.bar([x - bar_width/2 for x in x_positions], y_ndcg_before, bar_width, label='ndcg Before Sampling')

    # Create bars for 'ndcg after sampling active learning'
    ax.bar([x + bar_width/2 for x in x_positions], y_ndcg_after, bar_width, label='ndcg After Sampling')

    # Set X-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)

    # Set labels and title
    ax.set_xlabel('Folds')
    ax.set_ylabel('ndcg')
    ax.set_title('ndcg Before and After Sampling (Active Learning)')

    # Add a legend
    ax.legend()

    plt.show()

    return plt
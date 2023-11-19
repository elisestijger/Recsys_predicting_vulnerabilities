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
    bar2 = ax.bar(x, precision_after, width=bar_width, label='SingleBatch-20')
    bar3 = ax.bar(x + bar_width, precision_after2, width=bar_width, label='10Batch-20')

    ax.set_xlabel('Folds')
    ax.set_ylabel(str(metrics))
    # ax.set_title(str(metrics) + ' before and after random sampling active learning and single-batch active learning, adding 20 items')
    ax.set_title('\n'.join([str(metrics) + ' before and after active learning with single-batch and 10-batch sampling adding 20 items']), fontsize=10)
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

    plt.savefig("/Users/elisestijger/Desktop/graphs/"+str(metrics) +"_10batch.png")

    return plt

def make_bars2(data, metrics , data1, data2, data3, data4, data5, data0):

    # Extracting fold values and their corresponding precision values for each category
    fold_labels = list(data[str(metrics) +' before active learning'].keys())
    precision_before = list(data[str(metrics) +' before active learning'].values())
    precision_after = list(data[str(metrics) +' after active learning'].values())
    precision_after1 = list(data1[str(metrics) +' after active learning'].values())
    precision_after2 = list(data2[str(metrics) +' after active learning'].values())
    precision_after3 = list(data3[str(metrics) +' after active learning'].values())
    precision_after4 = list(data4[str(metrics) +' after active learning'].values())
    precision_after5 = list(data5[str(metrics) +' after active learning'].values())
    precision_after0 = list(data0[str(metrics) +' after active learning'].values())


    all_values = []  # List to collect all values for determining the min and max

    for key in data[str(metrics)+' before active learning']:
        all_values.append(data[str(metrics)+' before active learning'][key])
    for key in data[str(metrics)+' after active learning']:
        all_values.append(data[str(metrics)+' after active learning'][key])
    for key in data1[str(metrics)+' after active learning']:
        all_values.append(data1[str(metrics)+' after active learning'][key])
    for key in data2[str(metrics)+' after active learning']:
        all_values.append(data2[str(metrics)+' after active learning'][key])
    for key in data3[str(metrics)+' after active learning']:
            all_values.append(data3[str(metrics)+' after active learning'][key])
    for key in data4[str(metrics)+' after active learning']:
            all_values.append(data4[str(metrics)+' after active learning'][key])
    for key in data5[str(metrics)+' after active learning']:
            all_values.append(data5[str(metrics)+' after active learning'][key])
    for key in data0[str(metrics)+' after active learning']:
            all_values.append(data0[str(metrics)+' after active learning'][key])


    min_value = min(all_values)
    max_value = max(all_values)
    buffer = 0.1  # You can adjust this value

    data_range = max_value - min_value
    proportional_buffer = buffer * data_range
    min_value -= proportional_buffer
    max_value += proportional_buffer
    bar_width = 0.10      

    # and this was good:
    # x = range(len(fold_labels))

    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size

    x = np.arange(len(fold_labels))  # Generate array of values for the x-axis


    bar1 = ax.bar(x - 4 *  bar_width, precision_before, width=bar_width, label='Before active learning')
    bar0 = ax.bar(x - 3 *  bar_width, precision_after0, width=bar_width, label='Random-20')
    bar2 = ax.bar(x - 2 * bar_width, precision_after, width=bar_width, label='SingleBatch-20')
    bar3 = ax.bar(x - bar_width, precision_after1, width=bar_width, label='SingleBatch-40')
    bar4 = ax.bar(x , precision_after2, width=bar_width, label='4Batch-20')
    bar5 = ax.bar(x +  bar_width, precision_after3, width=bar_width, label='4Batch-40')
    bar6 = ax.bar(x + 2 * bar_width, precision_after4, width=bar_width, label='10Batch-20')
    bar7 = ax.bar(x +  3 * bar_width, precision_after5, width=bar_width, label='10Batch-40')



    ax.set_xlabel('Folds')
    ax.set_ylabel(str(metrics))
    # ax.set_title(str(metrics) + ' before and after random sampling active learning and single-batch active learning, adding 20 items')
    ax.set_title('\n'.join([str(metrics) + ' before and after active learning with single-batch, 4-batch or 10-batch sampling, adding 20 or 40 items']), fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.legend()

    # Set the y-axis limits to emphasize the difference
    # ax.set_ylim(0.189, 0.198)
    ax.set_ylim(min_value, max_value)

    # for i, rect in enumerate(bar1 + bar2 + bar3 + bar4 + bar5 + bar6 + bar7):
    #     height = rect.get_height()
    #     if(metrics != 'correct counts'):
    #         ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom')
    #     else:
    #         ax.text(rect.get_x() + rect.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()

    plt.savefig("/Users/elisestijger/Desktop/graphs/"+str(metrics) +"_ALL_RANDOM.png")

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

    labels = ["SingleBatch-20", "10Batch-20"]

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
    plt.savefig('/Users/elisestijger/Desktop/graphs/10batch'+str(metrics)+'.png')

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
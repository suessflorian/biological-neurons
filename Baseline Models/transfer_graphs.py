import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Baseline Models/csvs/transfer.csv')



fig, axs = plt.subplots(3,1)

legend_locations = ['lower right', 'lower center', 'lower center']

for i, dataset in enumerate(['MNIST', 'FASHION', 'SVHN']):
    mnist = data[data['dataset'] == dataset].drop('dataset', axis = 1)
    relevant_rows = mnist[mnist.attack == 'LInfFMNAttack'] # just because rows are duplicated
    relevant_rows = relevant_rows[['model', 'extraction_layer', 'train_accuracy', 'test_accuracy', 'baseline_train_accuracy', 'baseline_test_accuracy']].drop_duplicates()
    unique_models = relevant_rows.model.unique()
    for model, colour in zip(unique_models, ['blue', 'orange']):
        subset = relevant_rows[relevant_rows.model == model]
        axs[i].plot(subset.extraction_layer, subset.train_accuracy, color=colour, linestyle='solid')
        axs[i].plot(subset.extraction_layer, subset.test_accuracy, color=colour, linestyle='dashed')
    axs[i].plot(subset.extraction_layer, subset.baseline_train_accuracy, color='red', linestyle='solid')
    axs[i].plot(subset.extraction_layer, subset.baseline_test_accuracy, color='red', linestyle='dashed')
    axs[i].legend([
        'ParaLIF (Train)',
        'ParaLIF (Test)',
        'LIF (Train)',
        'LIF (Test)',
        'LeNet Baseline (Train)',
        'LeNet Baseline (Test)'
    ])
    axs[i].set_title(f'{dataset}')
    axs[i].set_xticks(subset.extraction_layer)
    axs[i].grid(axis='x')
    axs[i].set_xlabel('Extraction Layer')
    axs[i].set_ylabel('Accuracy')

plt.subplots_adjust(hspace=0.35)
plt.show()


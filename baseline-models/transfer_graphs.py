import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12.8, 9.6))

data = pd.read_csv('Baseline Models/csvs/transfer.csv')



### OPTIONS ###
dataset = 'MNIST' # [SVHN, FASHION, MNIST]
attack = 'LInfFMNAttack' # [LInfFMNAttack, LinfDeepFoolAttack, LinfFastGradientAttack, SquareAttack]
epsilon = 0.01 # [0.01, 0.05, 0.1]

print_attack = False



dataset_specific = data[data['dataset'] == dataset].drop('dataset', axis = 1)


# ACCURACY PLOT
if print_attack:
    relevant_rows = dataset_specific[dataset_specific.attack == 'LInfFMNAttack']
    relevant_rows = relevant_rows[['model', 'extraction_layer', 'train_accuracy', 'test_accuracy', 'baseline_train_accuracy', 'baseline_test_accuracy']].drop_duplicates()
    unique_models = relevant_rows.model.unique()

    for model, colour in zip(unique_models, ['blue', 'orange']):
        subset = relevant_rows[relevant_rows.model == model]
        plt.plot(subset.extraction_layer, subset.train_accuracy, color=colour, linestyle='solid')
        plt.plot(subset.extraction_layer, subset.test_accuracy, color=colour, linestyle='dashed')
    plt.plot(subset.extraction_layer, subset.baseline_train_accuracy, color='red', linestyle='solid')
    plt.plot(subset.extraction_layer, subset.baseline_test_accuracy, color='red', linestyle='dashed')
    plt.legend([
        'ParaLIF (Train)',
        'ParaLIF (Test)',
        'LIF (Train)',
        'LIF (Test)',
        'LeNet Baseline (Train)',
        'LeNet Baseline (Test)'
    ])
    plt.title(f'Train Test {dataset}')
    plt.xticks(subset.extraction_layer)
    plt.grid(axis='x')
    plt.xlabel('Extraction Layer')
    plt.ylabel('Accuracy')
    plt.show()



# Attacks

dataset_attack = dataset_specific[(dataset_specific.epsilon == epsilon) & (dataset_specific.attack == attack)]

unique_models = dataset_attack.model.unique()

for model, colour in zip(unique_models, ['blue', 'orange']):
    subset = dataset_attack[dataset_attack.model == model]
    plt.plot(subset.extraction_layer, subset.susceptibility_rate, color=colour)
plt.plot(subset.extraction_layer, subset.baseline_susceptibility_rate, color='red', linestyle='dashed')
plt.legend(unique_models.tolist() + ['LeNet Baseline'])
plt.xlabel('Layer')
plt.ylabel('Attack Susceptibility')
plt.title(f'FGSM, epsilon={epsilon}')
plt.xticks(subset.extraction_layer)
plt.grid(axis='x')
plt.show()


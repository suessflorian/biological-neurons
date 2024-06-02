import pandas as pd
import matplotlib.pyplot as plt

running_times = pd.read_csv('Baseline Models/csvs/training_time_comp.csv')

fig, axs = plt.subplots(2,1, sharey=True, dpi=200)

linestyles = ['solid', 'dashed']
colours = ['red', 'green', 'blue', 'orange']

for i, dataset in enumerate(['MNIST', 'FASHION']):
    baseline_subset = running_times[['iteration', 'baseline', 'baseline_time']].drop_duplicates()['baseline_time'].agg(['mean', 'std'])
    
    legend = []
    subset = running_times[running_times.dataset == dataset].drop('dataset', axis=1)
    for j, model in enumerate(subset.model.unique()):
        subset_model = subset[subset.model == model].drop('model', axis=1)
        values = subset_model.groupby(['n_steps', 'n_hidden_layers'])['time'].agg(['mean', 'std']).reset_index().sort_values(['n_hidden_layers', 'n_steps'])
        for n in range(4):
            final = values[values.n_hidden_layers == n]
            
            axs[i].plot(final['n_steps'], final['mean'], color = colours[n], linestyle = linestyles[j])
            legend += [f'{model} w/ {n} hidden layers']
    
    axs[i].plot([0, 20, 40, 60, 100], baseline_subset.repeat(5)['mean'], color = 'black')
    axs[i].legend(legend, fontsize='xx-small')
    axs[i].set_title(dataset)

axs[i].set_xlabel('Number of Steps (T)')
fig.text(0.04, 0.5, 'Mean Training Time (seconds)', va='center', rotation='vertical')

# fig.suptitle('Training Time per 1 Epoch')
fig.subplots_adjust(hspace=0.35)
plt.show()
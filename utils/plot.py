### this file comes from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw2

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""

def plot_data(data, value="AverageReturn", x="Iteration"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, y=value, x=x, hue="Condition") #, units="Unit", estimator=None)
    plt.legend(loc='best').set_draggable(True)
    plt.show()


def get_datasets(fpath, condition=None, x='Iteration'):
    unit = 0
    datasets = []
    if 'log.txt' in os.listdir(fpath):
        param_path = open(os.path.join(fpath,'params.json'))
        params = json.load(param_path)
        exp_name = params['exp_name']
        
        log_path = os.path.join(fpath,'log.txt')
        experiment_data_all = pd.read_table(log_path)

        # quick hack TODO
        n = len(experiment_data_all) // len(experiment_data_all.loc[experiment_data_all['Iteration'] == experiment_data_all['Iteration'].min()])

        for i in range(0, len(experiment_data_all), n):
          experiment_data = experiment_data_all[i:i + n]

          if x == 'Episodes':
            df = pd.DataFrame()
            for ep in np.arange(0, experiment_data['Episodes'].max(), 50):
              index = abs(experiment_data['Episodes'] - ep).idxmin()
              experiment_data_all.loc[index,'Episodes'] = ep
              df=df.append(experiment_data_all.iloc[[index]])
            experiment_data = df

          experiment_data.insert(
              len(experiment_data.columns),
              'Unit',
              unit
              )        
          experiment_data.insert(
              len(experiment_data.columns),
              'Condition',
              condition or exp_name
              )

          datasets.append(experiment_data)
        unit += 1

    return datasets


def plot(logdir, legend=None, value=None, x="Iteration"):
    use_legend = False
    if legend is not None:
        assert len(legend) == len(logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(logdir, legend):
            data += get_datasets("experiments/" + logdir, legend_title, x=x)
    else:
        for logdir in logdir:
            data += get_datasets("experiments/" + logdir, x=x)

    if isinstance(value, list):
        values = value
    else:
        values = [value]
    if isinstance(x, list):
        xs = x
    else:
        xs = [x]*len(values)
    for value, x in zip(values, xs):
        plot_data(data, value=value, x=x)


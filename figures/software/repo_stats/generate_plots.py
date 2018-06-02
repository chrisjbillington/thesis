import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

### Text ###
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Minion Pro']
rcParams['font.size'] = 8.5
rcParams['text.usetex'] = True


### Layout ###
rcParams.update({'figure.autolayout': True})

### Axes ###
rcParams['axes.labelsize'] = 9.24994 # memoir \small for default 10pt, to match caption text
rcParams['axes.titlesize'] = 9.24994
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.labelsize'] = 9.24994
rcParams['ytick.labelsize'] = 9.24994
rcParams['lines.markersize'] = 4
rcParams['xtick.major.size'] = 2
rcParams['xtick.minor.size'] = 0
rcParams['ytick.major.size'] = 2
rcParams['ytick.minor.size'] = 0

### Legends ###
rcParams['legend.fontsize'] = 8.5 # memoir \scriptsize for default 10pt
rcParams['legend.borderpad'] = 0
rcParams['legend.handlelength'] = 1.5 # Big enough for multiple dashes or dots
rcParams['legend.handletextpad'] = 0.3
rcParams['legend.labelspacing'] = 0.3
rcParams['legend.frameon'] = False
rcParams['legend.numpoints'] = 1

with open('aliases.txt') as f:
    aliases = {}
    for line in f.readlines():
        if line:
            alias, name = line.split('=')
            alias = alias.strip()
            name = name.strip()
            aliases[alias] = name

# this_dir = os.getcwd()
# for name in os.listdir('.'):
#     if os.path.isdir(name):
#         os.chdir(name)
#         os.system('hg pull')
#         os.system('hg update')
#         os.system('hg churn ' +
#                   '--include "**.py" ' +
#                   '--include "**.pyw" ' +
#                   '--include "**.c" ' +
#                   '--include "**.pyx" ' +
#                   '--aliases ../aliases.txt '
#                   # '--diffstat ' +
#                   f' > ../{name}_churn.txt')
#         os.system('hg blame -u ' +
#                   '--include "**.py" ' +
#                   '--include "**.pyw" ' +
#                   '--include "**.c" ' +
#                   '--include "**.pyx" * ' +
#                   f' > ../{name}_blame.txt')
#         os.chdir(this_dir)


def get_counts_churn():
    counts = defaultdict(lambda: defaultdict(int))
    for filename in os.listdir('.'):
        if 'churn' in filename:
            project_name = filename.replace('_churn.txt', '')
            with open(filename) as f:
                for line in f:
                    name = line.split('  ')[0]
                    if not name:
                        name = 'Unknown'
                    lines = int(line.split(' ')[-2])
                    counts[name][project_name] = lines
    return counts

def get_counts_blame():
    counts = defaultdict(lambda: defaultdict(int))
    for filename in os.listdir('.'):
        if 'blame' in filename:
            project_name = filename.replace('_blame.txt', '')
            with open(filename) as f:
                for line in f:
                    name = line.split(':')[0].strip()
                    if name in aliases:
                        name = aliases[name]
                    if not name:
                        name = 'Unknown'
                    counts[name][project_name] += 1
    return counts

projects = None

FIG_WIDTH = 4.5
FIG_HEIGHT = 2.25

def make_plot(counts, filename):

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    totals_by_project = defaultdict(int)
    totals_by_name = defaultdict(int)

    for name, name_counts in counts.items():
        for project_name, lines in name_counts.items():
            totals_by_project[project_name] += lines
            totals_by_name[name] += lines

    global projects
    if projects is None:
        projects = list(reversed(sorted(totals_by_project.keys(),
                                   key=totals_by_project.__getitem__)))
    names = list(sorted(totals_by_name.keys(),
                            key=totals_by_name.__getitem__))

    # Only plot the top 8
    N = 8 #len(names)
    ind = np.arange(N)
    width = 0.35

    left = np.zeros(N)

    names = names[-N:]

    legend_bars = []
    legend_labels = []

    for project in projects:
        lines_by_name = [counts[name][project] for name in names]
        p = plt.barh(ind, lines_by_name, left=left)
        left += lines_by_name
        legend_bars.append(p)
        legend_labels.append(r'\texttt{' + project.replace('_', r'\_') + '}')

    plt.gca().set_yticks(ind)
    plt.gca().set_yticklabels(names)
    plt.legend(legend_bars, legend_labels, loc='lower right', ncol=2)

    if filename == 'churn':
        plt.xlabel('Historical lines changed/added')
    elif filename == 'blame':
        plt.xlabel('Current lines authored')

    plt.savefig(f'../{filename}.pdf')

# import numpy as np
# import matplotlib.pyplot as plt
make_plot(get_counts_blame(), 'blame')
make_plot(get_counts_churn(), 'churn')
plt.show()
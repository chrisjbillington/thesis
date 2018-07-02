import os
from subprocess import run, PIPE
import datetime
from matplotlib.dates import date2num
import numpy as np

os.chdir(os.getenv('HOME') + '/thesis')

wordcounts = []
dates = []

chapters = {'introduction.tex': 1,
            'atomic_physics.tex': 2,
            'numerics.tex': 3,
            'software.tex': 4,
            'velocimetry.tex': 5,
            'hidden_variables.tex': 6}

wordcounts = {chapter: [] for chapter in chapters.values()}

HIDDEN_REVISIONS = [177, 178]
i = 7
while True:
    if i in HIDDEN_REVISIONS:
        i += 1
        continue
    print(i)
    result = run(['hg', 'cat', '-r', str(i), 'wc.txt'], stdout=PIPE, stderr=PIPE)
    if 'unknown revision' in result.stderr.decode('utf8'):
        break
    elif 'no such file' in result.stderr.decode('utf8'):
        for chapter_number in chapters.values():
            wordcounts[chapter_number].append(np.nan)
    elif not result.stdout.decode('utf8'):
        # no wc.txt
        i += 1
        continue
    else:
        lines = result.stdout.decode('utf8').splitlines()
        chapters_counted = set()
        for line in lines:
            for chapter, chapter_number in chapters.items():
                if chapter in line:
                    subcounts = line.split()[0]
                    counts = sum([int(n) for n in subcounts.split('+')])
                    wordcounts[chapter_number].append(counts)
                    chapters_counted.add(chapter_number)
        assert chapters_counted == set(chapters.values())
    result = run(['hg', 'log', '-r', str(i), '--template', "'{date}'"],
                 stdout=PIPE)
    date = result.stdout.decode('utf8').split('-')[0].strip('\'')
    dt = datetime.datetime.fromtimestamp(float(date))
    dates.append(date2num(dt))
    i += 1

    # if i == 30:
    #     break

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())

# plt.step(dates, wordcounts, where='post')
# plt.step(dates, wordcounts, 'k-', markersize=1, where='post', label='wordcount')

plt.stackplot(
    dates, *wordcounts.values(), labels=[f'Chapter {d}' for d in chapters.values()]
)

# plt.row_stack

plt.axvline(date2num(datetime.datetime(2018, 7, 1)), linestyle='--', color='k', label='due date')
# plt.grid(True)
plt.title('Thesis')
plt.ylabel('wordcount')
plt.legend()
plt.gcf().autofmt_xdate()
plt.gca().set_ylim(ymin=0)

plt.savefig('wordcount_vs_time.pdf')
# plt.show()

import os
from subprocess import run, PIPE
import datetime
from matplotlib.dates import date2num

os.chdir('/home/cjb7/thesis')

wordcounts = []
dates = []

for i in range(7, 57):
    print(i)
    result = run(['hg', 'cat', '-r', str(i), 'wc.txt'], stdout=PIPE)
    try:
        lines = result.stdout.decode('utf8').splitlines()
        words_in_text = int(lines[1].split(': ')[1])
        words_in_headers = int(lines[2].split(': ')[1])
        words_outside_text = int(lines[3].split(': ')[1])
        math_inlines = int(lines[6].split(': ')[1])
        math_display = int(lines[7].split(': ')[1])

        words = (words_in_text + words_in_headers + words_outside_text + 
                 math_inlines + math_display)
    except Exception:
        pass
    else:
        result = run(['hg', 'log', '-r', str(i), '--template', "'{date}'"],
                     stdout=PIPE)
        date = result.stdout.decode('utf8').split('-')[0].strip('\'')
        dt = datetime.datetime.fromtimestamp(float(date))
        dates.append(date2num(dt))
        wordcounts.append(words)


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())

plt.step(dates, wordcounts, where='post')
plt.grid(True)
plt.ylabel('wordcount')

plt.gcf().autofmt_xdate()


plt.show()
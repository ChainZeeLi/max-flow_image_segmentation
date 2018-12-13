import sys
import time

def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

spinner = spinning_cursor()


def spin(text):
	sys.stdout.write(text)
	for _ in range(50):
   # sys.stdout.write(next(spinner))
	    sys.stdout.write(next(spinner))
	    sys.stdout.flush()
	    time.sleep(0.1)
	    sys.stdout.write('\b')




import progressbar
from time import sleep
def bar(l):
    bar = progressbar.ProgressBar(maxval=l, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in xrange(l):
        bar.update(i+1)
        sleep(0.1)
    bar.finish()


bar(50000)


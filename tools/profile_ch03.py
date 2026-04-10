import os
import sys
import cProfile
import pstats

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from experiments.ch03_structured.experiment import run_ch03
print('starting profile run_ch03')
pr = cProfile.Profile()
pr.enable()
run_ch03(N=512, p=0.1, beta=0.5, n_trials=2, n_seeds=2)
pr.disable()
print('profile complete, printing stats')
ps = pstats.Stats(pr).sort_stats('cumtime')
ps.print_stats(40)
print('done')
import os
import sys
import cProfile
import pstats

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from experiments.ch03_structured.experiment import run_ch03

print('starting profile run_ch03')
pr = cProfile.Profile()
pr.enable()
run_ch03(N=512, p=0.1, beta=0.5, n_trials=2, n_seeds=2)
pr.disable()
print('profile complete, printing stats')
ps = pstats.Stats(pr).sort_stats('cumtime')
ps.print_stats(40)
print('done')

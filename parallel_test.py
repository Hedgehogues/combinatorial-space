import multiprocessing
import time

def a():
    time.sleep(2)
    return 'a'
def b():
    time.sleep(2)
    return 'b'
def c():
    time.sleep(1)
    return 'c'

params_mapping  = {
  'a':a,
  'b':b,
  'c':c
}
def func(param):
    return params_mapping[param]()

p = multiprocessing.Pool()

results = p.map(func,['a','b','c'])
print(results)
p.close()
p.join()
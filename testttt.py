import numpy as np 

a = [
    501.5,500.7,492.0,504.7,483.0,512.8,504.0,490.3, 486.0, 520.0
]

mu = 500 

_ = 0.0
for i in a:
    _ += (i - mu)**2
print(_)
xerion is the data management tool for PMSP96:

- URL:
    The original data will be found at: 
    http://www.cnbc.cmu.edu/~plaut/xerion/xerion-3.1-nets-share.tar.gz

- The data format is as following:
    - onset ['Y', 'S', 'P', 'T', 'K', 'Q', 'C', 'B', 'D', 'G', 'F', 'V', 'J', 'Z', 'L', 'M', 'N', 'R', 'W', 'H', 'CH', 'GH', 'GN', 'PH', 'PS', 'RH', 'SH', 'TH', 'TS', 'WH']
    - vowel ['E', 'I', 'O', 'U', 'A', 'Y', 'AI', 'AU', 'AW', 'AY', 'EA', 'EE', 'EI', 'EU', 'EW', 'EY', 'IE', 'OA', 'OE', 'OI', 'OO', 'OU', 'OW', 'OY', 'UE', 'UI', 'UY']
    - coda ['H', 'R', 'L', 'M', 'N', 'B', 'D', 'G', 'C', 'X', 'F', 'V', '∫', 'S', 'Z', 'P', 'T', 'K', 'Q', 'BB', 'CH', 'CK', 'DD', 'DG', 'FF', 'GG', 'GH', 'GN', 'KS', 'LL', 'NG', 'NN', 'PH', 'PP', 'PS', 'RR', 'SH', 'SL', 'SS', 'TCH', 'TH', 'TS', 'TT', 'ZZ', 'U', 'E', 'ES', 'ED']

    - onset ['s', 'S', 'C', 'z', 'Z', 'j', 'f', 'v', 'T', 'D', 'p', 'b', 't', 'd', 'k', 'g', 'm', 'n', 'h', 'I', 'r', 'w', 'y']
    - vowel ['a', 'e', 'i', 'o', 'u', '@', '^', 'A', 'E', 'I', 'O', 'U', 'W', 'Y']
    - coda ['r', 'I', 'm', 'n', 'N', 'b', 'g', 'd', 'ps', 'ks', 'ts', 's', 'z', 'f', 'v', 'p', 'k', 't', 'S', 'Z', 'T', 'D', 'C', 'j']

Referene:

- Plaut, D., McClelland, J. L., Seidenberg, M. S., & Patterson, K. (1996). Understanding normal and impaired word reading: Computational principles in quasi-regular domains. Psychological Review, 103, 56–115.
- Seidenberg, M. S., & McClelland, J. L. (1989). A distributed, developmenta model of word recognition and naming. Psychological Review, 96, 523–568.

Usage:
```python
import numpy
import xerion

#from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

data = xerion.xerion()
X = np.asarray(data.input, dtype=np.float32)
y = np.asarray(data.output, dtype=np.float32)

model = MLPRegressor()
model.fit(X,y)
```

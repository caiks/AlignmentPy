# AlignmentPy

The AlignmentPy repository is a literal Python implementation of some of the set-theoretic functions and structures described in the paper *The Theory and Practice of Induction by Alignment* at https://greenlake.co.uk/. The AlignmentPy repository is a  translation of the orginal Haskell [Alignment repository](https://github.com/caiks/Alignment). 
<!--
The AlignmentPy repository is designed with the goal of theoretical correctness rather performance. A fast implementation of *practicable inducers* is in the [AlignmentRepaPy repository](https://github.com/caiks/AlignmentRepaPy).
-->

## Documentation

Some of the sections of the Overview of the paper have been illustrated with a [Python commentary](https://greenlake.co.uk/pages/overview_python). The comments provide both (a) code examples for the paper and (b) documentation for the code. 
<!--
For programmers who are interested in implementing *inducers*, some of the sections of the paper have been expanded in a [Python commentary](https://greenlake.co.uk/pages/inducer_python) with links to documentation of the code in the repository. The code documentation is gathered together in [Python code](https://greenlake.co.uk/pages/inducer_python_implementation). 
-->

## Installation

The `Alignment` module requires the [Python 3 platform](https://www.python.org/downloads/) to be installed.

For example in Ubuntu,
```
sudo apt-get update
sudo apt-get install python3.5
sudo apt install python3-pip
```
Then use the Python installer tool `pip` to install [Sorted Containers](http://www.grantjenks.com/docs/sortedcontainers) and [NumPy and SciPy](https://www.scipy.org/), 
```
python3.5 -m pip install --user numpy
python3.5 -m pip install --user scipy
python3.5 -m pip install --user sortedcontainers
```
Then download the zip file or use git to get the repository -
```
cd
git clone https://github.com/caiks/AlignmentPy.git
```


## Usage

The Alignment modules are not optimised for performance and are mainly intended to allow experimentation in the Python interpreter. Load `AlignmentDev` to import the modules and define various useful abbreviated functions,
```
cd AlignmentPy
python3
```
```py
from AlignmentDev import *

```
The Alignment types implement a `__str__` method which approximates to a set-theoretic representation of the structure. 

For example, to create a *regular cartesion histogram* of *dimension* 2 and *valency* 3 and display the result,
```py
regcart(3,2)
# {({(1, 1), (2, 1)}, 1 % 1), ({(1, 1), (2, 2)}, 1 % 1), ({(1, 1), (2, 3)}, 1 % 1), ({(1, 2), (2, 1)}, 1 % 1), ({(1, 2), (2, 2)}, 1 % 1), ({(1, 2), (2, 3)}, 1 % 1), ({(1, 3), (2, 1)}
, 1 % 1), ({(1, 3), (2, 2)}, 1 % 1), ({(1, 3), (2, 3)}, 1 % 1)}
```
Larger structures can be converted to a list and displayed over more than one line,
```py
rpln(aall(regcart(3,2)))
# ({(1, 1), (2, 1)}, 1 % 1)
# ({(1, 1), (2, 2)}, 1 % 1)
# ({(1, 1), (2, 3)}, 1 % 1)
# ({(1, 2), (2, 1)}, 1 % 1)
# ({(1, 2), (2, 2)}, 1 % 1)
# ({(1, 2), (2, 3)}, 1 % 1)
# ({(1, 3), (2, 1)}, 1 % 1)
# ({(1, 3), (2, 2)}, 1 % 1)
# ({(1, 3), (2, 3)}, 1 % 1)
```
Here is a simple example that calculates the *alignment* of a *regular cartesian*, a *regular diagonal* and the *resized sum*,
```py
cc = resize(100,(regcart(3,2)))

rpln(aall(cc))
# ({(1, 1), (2, 1)}, 100 % 9)
# ({(1, 1), (2, 2)}, 100 % 9)
# ({(1, 1), (2, 3)}, 100 % 9)
# ({(1, 2), (2, 1)}, 100 % 9)
# ({(1, 2), (2, 2)}, 100 % 9)
# ({(1, 2), (2, 3)}, 100 % 9)
# ({(1, 3), (2, 1)}, 100 % 9)
# ({(1, 3), (2, 2)}, 100 % 9)
# ({(1, 3), (2, 3)}, 100 % 9)

dd = resize(100,(regdiag(3,2)))

rpln(aall(dd))
# ({(1, 1), (2, 1)}, 100 % 3)
# ({(1, 2), (2, 2)}, 100 % 3)
# ({(1, 3), (2, 3)}, 100 % 3)

aa = resize(100,(add(cc,dd)))

rpln(aall(aa))
# ({(1, 1), (2, 1)}, 200 % 9)
# ({(1, 1), (2, 2)}, 50 % 9)
# ({(1, 1), (2, 3)}, 50 % 9)
# ({(1, 2), (2, 1)}, 50 % 9)
# ({(1, 2), (2, 2)}, 200 % 9)
# ({(1, 2), (2, 3)}, 50 % 9)
# ({(1, 3), (2, 1)}, 50 % 9)
# ({(1, 3), (2, 2)}, 50 % 9)
# ({(1, 3), (2, 3)}, 200 % 9)

ind(dd) == cc
# True

ind(aa) == cc
# True

algn(cc)
# 0.0

algn(aa)
# 22.098856350828214

algn(dd)
# 98.71169723276279
```


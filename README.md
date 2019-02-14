# AlignmentPy

The AlignmentPy repository is a literal Python implementation of some of the set-theoretic functions and structures described in the paper *The Theory and Practice of Induction by Alignment* at https://greenlake.co.uk/. The AlignmentPy repository is a literal translation of the orginal Haskell [Alignment repository](https://github.com/caiks/Alignment). 

The AlignmentPy repository is designed with the goal of theoretical correctness rather performance. A fast implementation of *practicable inducers* is in the [AlignmentRepaPy repository](https://github.com/caiks/AlignmentRepaPy).

## Documentation

Some of the sections of the Overview of the paper have been illustrated with a [Python commentary](https://greenlake.co.uk/pages/overview_python). The comments provide both (a) code examples for the paper and (b) documentation for the code. 

For programmers who are interested in implementing *inducers*, some of the sections of the paper have been expanded in a [Python commentary](https://greenlake.co.uk/pages/inducer_python) with links to documentation of the code in the repository. The code documentation is gathered together in [Python code](https://greenlake.co.uk/pages/inducer_python_implementation). 

## Installation

The `Alignment` module requires the [Python 3 platform](https://www.python.org/downloads/) to be installed.

For example in Ubuntu,
```
sudo apt-get update
sudo apt-get install python3.7
```
Then download the zip file or use git to get the repository -
```
cd
git clone https://github.com/caiks/AlignmentPy.git
```
Then use the Python installer tool `pip` to install [Sorted Containers](http://www.grantjenks.com/docs/sortedcontainers) and [NumPy and SciPy](https://www.scipy.org/), 
```
pip install numpy-1.15.2-cp37-none-win32.whl

pip install scipy-1.1.0-cp37-none-win32.whl

pip install sortedcontainers-2.0.5-py2.py3-none-any.whl
````

## Usage

The Alignment modules are not optimised for performance and are mainly intended to allow experimentation in the Python interpreter. Load `AlignmentDev` to import the modules and define various useful abbreviated functions,
```
cd AlignmentPy
python
```
```py
from AlignmentDev import *

```
The Alignment types implement the class type `Represent` defined in `AlignmentUtil` which requires them to implement the `represent` function,
```py
represent :: Show a => a -> String
```
The `represent` function returns a `String` that approximates to a set-theoretic representation of the structure. `AlignmentDev` defines the abbreviation `rp`.

For example, to create a *regular cartesion histogram* of *dimension* 2 and *valency* 3 and display the result,
```py
rp $ regcart 3 2
"{({(1,1),(2,1)},1 % 1),({(1,1),(2,2)},1 % 1),({(1,1),(2,3)},1 % 1),({(1,2),(2,1)},1 % 1),({(1,2),(2,2)},1 % 1),({(1,2),(2,3)},1 % 1),({(1,3),(2,1)},1 % 1),({(1,3),(2,2)},1 % 1),({(1,3),(2,3)},1 % 1)}"
```
Larger structures can be converted to a list and displayed over more than one line,
```py
rpln $ aall $ regcart 3 2
"({(1,1),(2,1)},1 % 1)"
"({(1,1),(2,2)},1 % 1)"
"({(1,1),(2,3)},1 % 1)"
"({(1,2),(2,1)},1 % 1)"
"({(1,2),(2,2)},1 % 1)"
"({(1,2),(2,3)},1 % 1)"
"({(1,3),(2,1)},1 % 1)"
"({(1,3),(2,2)},1 % 1)"
"({(1,3),(2,3)},1 % 1)"
```
Here is a simple example that calculates the *alignment* of a *regular cartesian*, a *regular diagonal* and the *resized sum*,
```py
let cc = resize 100 $ regcart 3 2

rpln $ aall cc
"({(1,1),(2,1)},100 % 9)"
"({(1,1),(2,2)},100 % 9)"
"({(1,1),(2,3)},100 % 9)"
"({(1,2),(2,1)},100 % 9)"
"({(1,2),(2,2)},100 % 9)"
"({(1,2),(2,3)},100 % 9)"
"({(1,3),(2,1)},100 % 9)"
"({(1,3),(2,2)},100 % 9)"
"({(1,3),(2,3)},100 % 9)"

let dd = resize 100 $ regdiag 3 2

rpln $ aall dd
"({(1,1),(2,1)},100 % 3)"
"({(1,2),(2,2)},100 % 3)"
"({(1,3),(2,3)},100 % 3)"

let aa = resize 100 $ cc `add` dd

rpln $ aall aa
"({(1,1),(2,1)},200 % 9)"
"({(1,1),(2,2)},50 % 9)"
"({(1,1),(2,3)},50 % 9)"
"({(1,2),(2,1)},50 % 9)"
"({(1,2),(2,2)},200 % 9)"
"({(1,2),(2,3)},50 % 9)"
"({(1,3),(2,1)},50 % 9)"
"({(1,3),(2,2)},50 % 9)"
"({(1,3),(2,3)},200 % 9)"

ind dd == cc
True

ind aa == cc
True

algn cc
0.0

algn aa
22.09885634287619

algn dd
98.71169723276279
```


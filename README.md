# AlignmentPy

The Alignment repository is a literal Haskell implementation of some of the set-theoretic functions and structures described in the paper *The Theory and Practice of Induction by Alignment* at https://greenlake.co.uk/. 

The Alignment repository is designed with the goal of theoretical correctness rather performance. A fast implementation of *practicable inducers* is in the [AlignmentRepa repository](https://github.com/caiks/AlignmentRepaPy).

## Documentation

Some of the sections of the Overview of the paper have been illustrated with a [Haskell commentary](https://greenlake.co.uk/pages/overview_haskell). The comments provide both (a) code examples for the paper and (b) documentation for the code. 

For programmers who are interested in implementing *inducers*, some of the sections of the paper have been expanded in a [Haskell commentary](https://greenlake.co.uk/pages/inducer_haskell) with links to documentation of the code in the repository. The code documentation is gathered together in [Haskell code](https://greenlake.co.uk/pages/inducer_haskell_implementation). 

## Installation

The `Alignment` module requires the [Haskell platform](https://www.haskell.org/downloads#platform) to be installed.

For example in Ubuntu,
```
sudo apt-get update
sudo apt-get install haskell-platform
```
Then download the zip file or use git to get the repository -
```
cd
git clone https://github.com/caiks/Alignment.git
```

## Usage

The Alignment modules are not optimised for performance and are mainly intended to allow experimentation in the Haskell interpreter. Load `AlignmentDev` to import the modules and define various useful abbreviated functions,
```
cd Alignment
ghci
```
```hs
:set +m
:l AlignmentDev
```
The Alignment types implement the class type `Represent` defined in `AlignmentUtil` which requires them to implement the `represent` function,
```hs
represent :: Show a => a -> String
```
The `represent` function returns a `String` that approximates to a set-theoretic representation of the structure. `AlignmentDev` defines the abbreviation `rp`.

For example, to create a *regular cartesion histogram* of *dimension* 2 and *valency* 3 and display the result,
```hs
rp $ regcart 3 2
"{({(1,1),(2,1)},1 % 1),({(1,1),(2,2)},1 % 1),({(1,1),(2,3)},1 % 1),({(1,2),(2,1)},1 % 1),({(1,2),(2,2)},1 % 1),({(1,2),(2,3)},1 % 1),({(1,3),(2,1)},1 % 1),({(1,3),(2,2)},1 % 1),({(1,3),(2,3)},1 % 1)}"
```
Larger structures can be converted to a list and displayed over more than one line,
```hs
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
```hs
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


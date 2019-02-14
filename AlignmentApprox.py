from Alignment import *
from math import *
from scipy.special import *

# histogramsEntropy :: Histogram -> Double

def histogramsEntropy(aa):
    def norm(aa):
        return histogramsResize(1,aa)
    trim = histogramsTrim
    aall = histogramsList
    return -sum([a * log(a) for (ss,a) in aall(trim(norm(aa)))])

# histogramsMultinomialLog :: Histogram -> Double

def histogramsMultinomialLog(aa):
    aall = histogramsList
    size = histogramsSize
    def facln(x):
        return gammaln(float(x) + 1)
    return facln(size(aa)) - sum([facln(c) for (ss,c) in aall(aa)])

# histogramsAlignment :: Histogram -> Double

def histogramsAlignment(aa):
    aall = histogramsList
    ind = histogramsIndependent
    def facln(x):
        return gammaln(float(x) + 1)
    return sum([facln(c) for (ss,c) in aall(aa)]) - sum([facln(c) for (ss,c) in aall(ind(aa))])

# transformsHistogramsEntropyComponent :: Transform -> Histogram -> Double

def transformsHistogramsEntropyComponent(tt,aa):
    def norm(aa):
        return histogramsResize(1,aa)
    size = histogramsSize
    mul = pairHistogramsMultiply
    def sunit(ss):
        return setStatesHistogramUnit(sset([ss]))
    entropy = histogramsEntropy
    def inv(tt):
        return list(transformsInverse(tt).items())
    def tmul(aa,tt):
        return transformsHistogramsApply(tt,aa)
    aa1 = tmul(norm(aa),tt)
    return sum([size(mul(aa1,sunit(rr))) * entropy(mul(aa,cc)) for (rr,cc) in inv(tt)])

# histogramsHistogramsEntropyCross :: Histogram -> Histogram -> Double

def histogramsHistogramsEntropyCross(aa,bb):
    def norm(aa):
        return histogramsResize(1,aa)
    size = histogramsSize
    mul = pairHistogramsMultiply
    def aall(aa):
        return histogramsList(histogramsTrim(aa))
    def sunit(ss):
        return setStatesHistogramUnit(sset([ss]))
    return -sum([size(mul(norm(aa),sunit(ss))) * log(b) for (ss,b) in aall(norm(bb))])

# setVarsTransformsHistogramsEntropyLabel :: Set.Set Variable -> Transform -> Histogram -> Double

def setVarsTransformsHistogramsEntropyLabel(kk,tt,aa):
    def red(aa,vv):
        return setVarsHistogramsReduce(vv,aa)
    size = histogramsSize
    mul = pairHistogramsMultiply
    def sunit(ss):
        return setStatesHistogramUnit(sset([ss]))
    entropy = histogramsEntropy
    def inv(tt):
        return list(transformsInverse(tt).items())
    def tmul(aa,tt):
        return transformsHistogramsApply(tt,aa)
    vars = histogramsSetVar
    vk = vars(aa) - kk
    aa1 = tmul(aa,tt)
    return sum([size(mul(aa1,sunit(rr))) * entropy(red(mul(aa,cc),vk)) for (rr,cc) in inv(tt)])





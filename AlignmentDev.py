from Alignment import *
from AlignmentSubstrate import *
from AlignmentApprox import *
from AlignmentRandom import *
from AlignmentPracticable import *
from AlignmentAeson import *
from AlignmentAesonPretty import *

ssll = statesList
llss = listsState
sunion = pairStatesUnionLeft
ssplit = setVarsSetStatesSplit 

cart = systemsSetVarsSetStateCartesian
sysreg = systemRegular
uunion = pairSystemsUnion
uvars = systemsVars
vol = systemsSetVarsVolume

aat = histogramsStatesCount
aall = histogramsList
llaa = listsHistogram
def aarr(aa):
    return [(ss,float(q)) for (ss,q) in aall(aa)]
unit = setStatesHistogramUnit
add = pairHistogramsAdd
sub = pairHistogramsSubtract
mul = pairHistogramsMultiply
divide = pairHistogramsDivide
apply = setVarsSetVarsSetHistogramsHistogramsApply
leq = pairHistogramsLeq
size = histogramsSize
resize = histogramsResize
def norm(aa):
    return histogramsResize(1,aa)
vars = histogramsSetVar
states = histogramsStates
dim = histogramsDimension
recip = histogramsReciprocal
def red(aa,vv):
    return setVarsHistogramsReduce(vv,aa)
def ared(aa,vv):
    return setVarsHistogramsReduce(vv,aa)
scalar = histogramScalar
single = histogramSingleton
trim = histogramsTrim
eff = histogramsEffective
ind = histogramsIndependent
empty = histogramEmpty
sys = histogramsSystemImplied 
regsing = histogramRegularUnitSingleton
regcart = histogramRegularCartesian
regdiag = histogramRegularUnitDiagonal
regpivot = histogramRegularUnitPivot
def reframe(aa,mm):
    return histogramsMapVarsFrame(aa,sdict(mm))
def cdtp(aa,ll):
    return reframe(aa, zip(list(vars(aa)), map(VarInt,ll)))
def cdaa(ll):
    return llaa([(llss([(VarInt(i), ValInt(j)) for i,j in enumerate(ss,1)]),1) for ss in ll])

llhh = listsHistory  
hhll = historiesList
hvars = historiesSetVar
hsize = historiesSize
def hred(hh,vv):
    return setVarsHistoriesReduce(vv,hh)
hadd = pairHistoriesAdd
hmul = pairHistoriesMultiply
aahh = histogramsHistory
hhaa = historiesHistogram
hshuffle = historiesShuffle
def ashuffle(aa,r):
    return hhaa(hshuffle(aahh(aa),r))

und = transformsUnderlying
der = transformsDerived
tvars = transformsVars
ttaa = transformsHistogram
def tmul(aa,tt):
    return transformsHistogramsApply(tt,aa)
def inv(tt):
    return list(transformsInverse(tt).items())
trans = histogramsSetVarsTransform

def cdtt(pp,ll):
    return trans(cdtp(cdaa(ll),pp), sset([VarInt(pp[-1])]))

qqff = setTransformsFud
ffqq = fudsSetTransform
def llff(ll):
    return setTransformsFud(sset(ll))
def ffll(ff):
    return list(fudsSetTransform(ff))
fvars = fudsSetVar
fder = fudsDerived
fund = fudsUnderlying
fhis = fudsSetHistogram
fftt = fudsTransform
def funion(ff,gg):
    return qqff(ffqq(ff) | ffqq(gg))
def fapply(aa,ff):
    return fudsHistogramsApply(ff,aa)
def fmul(aa,ff):
    return fudsHistogramsMultiply(ff,aa)
def layer(ff):
    return fudsSetVarsLayer(ff,fder(ff))
fdep = fudsSetVarsDepends
fsys = fudsSystemImplied

zzdf = treePairStateFudsDecompFud
dfzz = decompFudsTreePairStateFud
dfff = decompFudsFud
dfund = decompFudsUnderlying
def dfapply(aa,df):
    return decompFudsHistogramsApply(df,aa)
def dfmul(aa,df):
    return decompFudsHistogramsMultiply(df,aa)
dfnul = systemsDecompFudsNullablePracticable

ent = histogramsEntropy 
def lent(aa,ww,vvl):
    return ent(red(aa,ww|vvl)) - ent(red(aa,ww))
def rent(aa,bb):
    a = size(aa)
    b = size(bb)
    return (a+b) * ent(add(aa,bb)) - a * ent(aa) - b * ent(bb)
algn = histogramsAlignment

def rpln(xx):
    if isinstance(xx,dict):
        ll = list(xx.items())
    else:
        ll = list(xx)
    for x in ll:
       print(x)

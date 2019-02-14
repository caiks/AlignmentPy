from Alignment import *
import re

# stringsVariable :: String -> Variable

def stringsVariable(s):
    def start(s):
        return ','.join(s.split(',')[:-1])[1:]
    def end(s):
        return s.split(',')[-1][:-1]
    if s.isdigit():
        return VarInt(int(s))
    if len(s) >= 2 and s[0] == '<' and s[-1] == '>' and ',' in s:
        return VarPair((stringsVariable(start(s)),stringsVariable(end(s))))
    return VarStr(s)

# stringsValue :: String -> Value

def stringsValue(s):
    def isDouble(s):
        if s.lower() == "inf" or s.lower() == "-inf" or s.lower() == "infinity" or s.lower() == "-infinity": 
            return True
        if '.' not in s:
            return False  
        try:
            float(s)
            return True
        except:
            return False
    if isDouble(s):
        return ValDouble(float(s))
    if s.isdigit():
        return ValInt(int(s))
    return ValStr(s)

# stringsRational :: String -> Rational

def stringsRational(s):
    def isRational(s):
        try:
            Fraction(s)
            return True
        except:
            return False
    if '%' in s:
        s = re.sub('[\s+]','',s)
    if '%' in s and isRational('/'.join(s.split('%'))):
        return ratio(Fraction('/'.join(s.split('%'))))
    if isRational(s):
        return ratio(Fraction(s))
    return ratio()

# rationalsString :: Rational -> String

def rationalsString(r):
    return str(r)


# data VariablePersistent = VariablePersistent { var :: String, values :: [String] } deriving (Show,Generic) 


# data SystemPersistent = SystemPersistent [VariablePersistent] deriving (Show,Generic) 

# systemsPersistent :: System -> SystemPersistent

def systemsPersistent(uu):
    def vars(x):
        if isinstance(x, VarPair):
            (v,w) = x._rep
            return "<" + vars(v) + "," + vars(w) + ">"
        return str(x)
    def vals(x):
        return str(x)
    uull = systemsList
    return [{"var": vars(v), "values": [vals(w) for w in ww]} for (v,ww) in uull(uu)]

# persistentsSystem :: SystemPersistent -> Maybe System

def persistentsSystem(uu1):
    svar = stringsVariable
    sval = stringsValue        
    lluu = listsSystem
    return lluu([(svar(v1["var"]),sset([sval(w1) for w1 in v1["values"]])) for v1 in uu1])


# data HistoryPersistent = HistoryPersistent { hsystem :: SystemPersistent, hstates :: [[Int]] } deriving (Show,Generic) 

# historiesPersistent :: History -> HistoryPersistent

def historiesPersistent(hh):
    uull = systemsList
    sat = statesVarsValue
    def hsys(hh):
        return histogramsSystemImplied(historiesHistogram(hh))
    hhll = historiesList
    hvars = historiesSetVar
    uupp = systemsPersistent
    uu = hsys(hh)
    mm = dict([(v, dict([(w,i) for (i,w) in enumerate(ww)])) for (v,ww) in uull(uu)])
    vv = hvars(hh)
    return {"hsystem": uupp(uu), "hstates": [[mm[v][sat(ss,v)] for v in vv] for (_,ss) in hhll(hh)]}

# persistentsHistory :: HistoryPersistent -> Maybe History

def persistentsHistory(hh):
    ppuu = persistentsSystem
    llss = listsState
    llhh = listsHistory
    svar = stringsVariable
    sval = stringsValue        
    uu1 = ppuu(hh["hsystem"])
    if uu1 is None:
        return None
    ll = [svar(x["var"]) for x in hh["hsystem"]]
    mm = dict([(svar(x["var"]), [sval(w) for w in x["values"]]) for x in hh["hsystem"]])
    nn = []
    for (i,ss) in enumerate(hh["hstates"]):
        pp = []
        for (j,k) in enumerate(ss):
            v = ll[j]
            ww = mm[v]
            if k < len(ww):
                w = ww[k]
                pp.append((v,w))
        nn.append((IdInt(i+1), llss(pp)))
    return llhh(nn)


# data HistogramPersistent = HistogramPersistent { asystem :: SystemPersistent, astates :: [([Int],String)] } deriving (Show,Generic) 

# histogramsPersistent :: Histogram -> HistogramPersistent

def histogramsPersistent(aa):
    uull = systemsList
    sat = statesVarsValue
    sys = histogramsSystemImplied
    aall = histogramsList
    vars = histogramsSetVar
    uupp = systemsPersistent
    rs = rationalsString
    uu = sys(aa)
    mm = dict([(v, dict([(w,i) for (i,w) in enumerate(ww)])) for (v,ww) in uull(uu)])
    vv = vars(aa)
    return {"asystem": uupp(uu), "astates": [([mm[v][sat(ss,v)] for v in vv],rs(c)) for (ss,c) in aall(aa)]}

# persistentsHistogram :: HistogramPersistent -> Maybe Histogram

def persistentsHistogram(aa):
    ppuu = persistentsSystem
    llss = listsState
    llaa = listsHistogram
    svar = stringsVariable
    sval = stringsValue        
    sr = stringsRational
    uu1 = ppuu(aa["asystem"])
    if uu1 is None:
        return None
    ll = [svar(x["var"]) for x in aa["asystem"]]
    mm = dict([(svar(x["var"]), [sval(w) for w in x["values"]]) for x in aa["asystem"]])
    nn = []
    for (ss,r) in aa["astates"]:
        pp = []
        for (j,k) in enumerate(ss):
            v = ll[j]
            ww = mm[v]
            if k < len(ww):
                w = ww[k]
                pp.append((v,w))
        nn.append((llss(pp),sr(r)))
    return llaa(nn)


# data TransformPersistent = TransformPersistent { history :: HistoryPersistent, derived :: [String] } deriving (Show,Generic) 

# transformsPersistent :: Transform -> TransformPersistent

def transformsPersistent(tt):
    unit = histogramsUnit
    def aahh(aa):
        return histogramsHistory(unit(aa))
    his = transformsHistogram
    der = transformsDerived
    hhpp = historiesPersistent
    def vars(x):
        if isinstance(x, VarPair):
            (v,w) = x._rep
            return "<" + vars(v) + "," + vars(w) + ">"
        return str(x)
    return {"history": hhpp(aahh(his(tt))), "derived": [vars(w) for w in der(tt)]} 

# persistentsTransform :: TransformPersistent -> Maybe Transform

def persistentsTransform(tt):
    unit = histogramsUnit
    hhaa = historiesHistogram
    trans = histogramsSetVarsTransform
    svar = stringsVariable
    pphh = persistentsHistory
    hh1 = pphh(tt["history"])
    return trans(unit(hhaa(hh1)),sset([svar(w) for w in tt["derived"]]))


# data FudPersistent = FudPersistent [TransformPersistent] deriving (Show,Generic) 

# fudsPersistent :: Fud -> FudPersistent

def fudsPersistent(ff):
    ffqq = fudsSetTransform
    ttpp = transformsPersistent
    return [ttpp(tt) for tt in ffqq(ff)]

# persistentsFud :: FudPersistent -> Maybe Fud

def persistentsFud(ll):
    qqff = setTransformsFud
    pptt = persistentsTransform
    return qqff(sset([pptt(pp) for pp in ll]))


# data DecompFudPersistent = DecompFudPersistent { nodes :: [(HistoryPersistent,FudPersistent)], paths :: [[Int]] } 
#                                                                                                     deriving (Show,Generic) 

# decompFudsPersistent :: DecompFud -> DecompFudPersistent

def decompFudsPersistent(df):
    def flip(a,b):
        return (b,a)
    def fst(x):
        (a,b) = x
        return a
    unit = setStatesHistogramUnit
    aahh = histogramsHistory
    def sshh(ss):
        return aahh(unit(sset([ss])))
    dfzz = decompFudsTreePairStateFud
    ffpp = fudsPersistent
    hhpp = historiesPersistent
    zz = funcsListsTreesTraversePreOrder(flip,range(1000000000),dfzz(df))[0]
    nn = [(hhpp(sshh(ss)),ffpp(ff)) for (_,(ss,ff)) in treesElements(zz)]
    pp = treesPaths(funcsTreesMap(fst,zz))
    return {"nodes": nn, "paths": pp} 

# persistentsDecompFud :: DecompFudPersistent -> Maybe DecompFud

def persistentsDecompFud(df):
    def qqmin(qq):
        if len(qq) > 0:
            return list(qq)[0]
        return stateEmpty()
    hhqq = historiesSetState
    zzdf = treePairStateFudsDecompFud 
    pphh = persistentsHistory
    ppff = persistentsFud
    nn = [(qqmin(hhqq(pphh(hh))),ppff(ff)) for (hh,ff) in df["nodes"]]
    def at(i):
        return nn[i]
    zz = funcsTreesMap(at,pathsTree(df["paths"]))
    return zzdf(zz)

# persistentsDecompFud_u :: DecompFudPersistent -> DecompFud

def persistentsDecompFud_u(df):
    def qqmin(qq):
        if len(qq) > 0:
            return list(qq)[0]
        return stateEmpty()
    hhqq = historiesSetState
    def zzdf(zz):
        return zz
    pphh = persistentsHistory
    ppff = persistentsFud
    nn = [(qqmin(hhqq(pphh(hh))),ppff(ff)) for (hh,ff) in df["nodes"]]
    def at(i):
        return nn[i]
    zz = funcsTreesMap(at,pathsTree(df["paths"]))
    return zzdf(zz)



from AlignmentUtil import *

# data Variable = VarStr String | VarInt Integer | VarPartition Partition | VarIndex Int | VarPair (Variable,Variable) 

class Variable(object):
    def __init__(self, rep):
        self._rep = rep
        self._hash = 0
        self._cl = 0
    def __str__(self):
        return str(self._rep)
    def __repr__(self):
        return str(self)
    def __lt__(self,other):
        if other._cl < self._cl:
            return False
        elif other._cl > self._cl:
            return True
        else:
            return self._rep < other._rep
    def __eq__(self,other):
        return self._cl == other._cl and self._rep == other._rep
    def __le__(self,other):
        return self < other or self == other
    def __hash__(self):
        if self._hash == 0:
            self._hash = hash(self._rep)*6+self._cl
        return self._hash

class VarStr(Variable):
    def __init__(self, rep):
        self._rep = rep
        self._hash = 0
        self._cl = 1

class VarInt(Variable):
    def __init__(self, rep):
        self._rep = rep
        self._hash = 0
        self._cl = 2

class VarPartition(Variable):
    def __init__(self, rep):
        self._rep = rep
        self._hash = 0
        self._cl = 3

class VarIndex(Variable):
    def __init__(self, rep):
        self._rep = rep
        self._hash = 0
        self._cl = 4

class VarPair(Variable):
    def __init__(self, rep):
        self._rep = rep
        self._hash = 0
        self._cl = 5
    def __str__(self):
        (v,w) = self._rep
        return "<" + str(v) + "," + str(w) + ">"


# data Value = ValStr String | ValInt Integer | ValDouble Double | ValComponent Component | ValIndex Int

class Value(object):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 0
    def __str__(self):
        return str(self._rep)
    def __repr__(self):
        return str(self)
    def __lt__(self,other):
        if other._cl < self._cl:
            return False
        elif other._cl > self._cl:
            return True
        else:
            return self._rep < other._rep
    def __eq__(self,other):
        return self._cl == other._cl and self._rep == other._rep
    def __le__(self,other):
        return self < other or self == other
    def __hash__(self):
        return hash((self._cl,self._rep))

class ValStr(Value):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 1

class ValInt(Value):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 2

class ValDouble(Value):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 3

class ValComponent(Value):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 4

class ValIndex(Value):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 5


# data Id = IdStr String | IdInt Integer | IdStateInteger (State, Integer) | IdListId [Id] | IdIntId (Integer,Id) | IdPair (Id,Id) | IdNull

class Id(object):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 0
    def __str__(self):
        return str(self._rep)
    def __repr__(self):
        return str(self)
    def __lt__(self,other):
        if other._cl < self._cl:
            return False
        elif other._cl > self._cl:
            return True
        else:
            return self._rep < other._rep
    def __eq__(self,other):
        return self._cl == other._cl and self._rep == other._rep
    def __le__(self,other):
        return self < other or self == other
    def __hash__(self):
        return hash((self._cl,self._rep))

class IdStr(Id):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 1

class IdInt(Id):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 2

class IdStateInteger(Id):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 3

class IdListId(Id):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 4

class IdIntId(Id):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 5

class IdPair(Id):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 6
    def __str__(self):
        (v,w) = self._rep
        return "(" + str(v) + "," + str(w) + ")"

class IdNull(Id):
    def __init__(self, rep):
        self._rep = rep
        self._cl = 7
    def __str__(self):
        return "_"


# newtype System = System (Map.Map Variable (Set.Set Value)) 

# systemEmpty :: System

def systemEmpty():
    return sdict()

# listsSystem :: [(Variable, Set.Set Value)] -> Maybe System

def listsSystem(ll):
    ok = True
    uu = sdict(ll)
    for ww in uu.values():
       if len(ww)==0:
           ok = False
           break
    if ok:
        return uu
    return None

# listsSystem_u :: [(Variable, Set.Set Value)] -> System

def listsSystem_u(ll):
    return sdict(ll)

# systemsList :: System -> [(Variable, Set.Set Value)]

def systemsList(uu):
    return list(uu.items())

# pairSystemsUnion :: System -> System -> System

def pairSystemsUnion(uu,xx):
    yy = uu.copy()
    for (v,ww) in xx.items():
        if v in uu:
            yy[v] = uu[v] | ww
        else:
            yy[v] = ww
    return yy

# systemsVars :: System -> Set.Set Variable

def systemsSetVar(uu):
    return sset(uu.keys())

systemsVars = systemsSetVar

# systemsVarsSetValue :: System -> Variable -> Maybe (Set.Set Value)

def systemsVarsSetValue(uu,u):
    if u in uu:
        return uu[u]
    return None

# systemsSetVarsVolume :: System -> Set.Set Variable -> Maybe Integer

def systemsSetVarsVolume(uu,vv):
    v = 1
    for u in vv:
        if u in uu:
            v = v * len(uu[u])
        else:
            return None
    return v

# systemsSetVarsVolume_u :: System -> Set.Set Variable -> Maybe Integer

def systemsSetVarsVolume_u(uu,vv):
    v = 1
    for u in vv:
        v = v * len(uu[u])
    return v

# systemRegular :: Integer -> Integer -> Maybe System

def systemRegular(d,n):
    if d >= 1 and n >= 1:
        uu = listsSystem([(VarInt(i), sset([ValInt(j) for j in range(1,d+1)])) 
                for i in range(1,n+1)])
        return uu
    return None


# newtype State = State (Map.Map Variable Value)

# listsState :: [(Variable, Value)] -> State

def listsState(ll):
    return sdict(ll)

# statesList :: State -> [(Variable, Value)]

def statesList(ss):
    return list(ss.items())

# statesSetVar :: State -> Set.Set Variable

def statesSetVar(ss):
    return sset(ss.keys())

# statesVarsValue :: State -> Variable -> Maybe Value

def statesVarsValue(ss,u):
    if u in ss:
        return ss[u]
    return None

# stateEmpty :: State

def stateEmpty():
    return sdict()

# stateSingleton :: Variable -> Value -> State

def stateSingleton(v,w):
    return sdict([(v,w)])

# systemsStatesIs :: System -> State -> Bool

def systemsStatesIs(uu,ss):
    ok = statesSetVar(ss).issubset(systemsSetVar(uu))
    for (u,w) in statesList(ss):
        ok = ok and w in systemsVarsSetValue(uu,u)
    return ok

# systemsSetVarsSetStateCartesian :: System -> Set.Set Variable -> Maybe (Set.Set State)

def systemsSetVarsSetStateCartesian(uu,vv):
    if not (vv.issubset(systemsSetVar(uu))):
        return None
    qq = sset([stateEmpty()])
    if len(vv)==0:
        return qq
    for u in vv:
        ww = systemsVarsSetValue(uu,u)
        qq1 = sset()
        for ss in qq:
            for w in ww:
                qq1.add(listsState(statesList(ss) + [(u,w)]))
        qq = qq1
    return qq

# systemsSetVarsSetStateCartesian_u1 :: System -> Set.Set Variable -> Maybe (Set.Set State)

def systemsSetVarsSetStateCartesian_u1(uu,vv):
    qq = sset([stateEmpty()])
    if len(vv)==0:
        return qq
    for u in vv:
        ww = systemsVarsSetValue(uu,u)
        qq1 = sset()
        for ss in qq:
            for w in ww:
                qq1.add(listsState(statesList(ss) + [(u,w)]))
        qq = qq1
    return qq

# systemsSetVarsSetStateCartesian_u :: System -> Set.Set Variable -> Maybe (Set.Set State)

def systemsSetVarsSetStateCartesian_u(uu,vv):
    if len(vv)==0:
        return sset([stateEmpty()])
    qq = [[]]
    for u in vv:
        ww = systemsVarsSetValue(uu,u)
        qq1 = []
        for ss in qq:
            for w in ww:
                qq1.append(ss + [(u,w)])
        qq = qq1
    return sset([listsState(ss) for ss in qq])

# setVarsStatesStateFiltered :: Set.Set Variable -> State -> State 

def setVarsStatesStateFiltered(vv,ss):
    return listsState([(u,w) for (u,w) in statesList(ss) if u in vv])

# setVarsSetStatesSplit :: Set.Set Variable -> Set.Set State -> Set.Set (State,State)

def setVarsSetStatesSplit(vv,qq):
    sred = setVarsStatesStateFiltered
    svars = statesSetVar
    ll = list()
    for rr in qq:
        ss = sred(vv,rr)
        tt = sred(svars(rr) - vv, rr)
        ll.append((ss,tt))
    return sset(ll)

# pairStatesIntersection :: State -> State -> State

def pairStatesIntersection(ss,tt):
    llss = listsState
    ssll = statesList
    return llss(list(sset(ssll(ss)) & sset(ssll(tt))))

# pairStatesUnionLeft :: State -> State -> State

def pairStatesUnionLeft(ss,tt):
    rr = tt.copy()
    for (u,w) in statesList(ss):
        rr[u] = w
    return rr

# pairStatesUnionRight :: State -> State -> State

def pairStatesUnionRight(ss,tt):
    return pairStatesUnionLeft(tt,ss)

# pairStatesIsJoin :: State -> State -> Bool

def pairStatesIsJoin(ss,tt):
    return pairStatesUnionLeft(ss,tt) == pairStatesUnionRight(ss,tt)

# pairStatesIsSubstate :: State -> State -> Bool

def pairStatesIsSubstate(ss,tt):
    return sset(ss.items()).issubset(sset(tt.items()))

# newtype History = History (Map.Map Id State) 

# historyEmpty :: History

def historyEmpty():
    return sdict()

# listsHistory :: [(Id, State)] -> Maybe History

def listsHistory(ll):
    svars = statesSetVar
    if len(ll)==0:
        return historyEmpty()
    (j,rr) = ll[0]
    vv = svars(rr)
    for (i, ss) in ll:
        if svars(ss) != vv:
            return None
    return sdict(ll)

# listsHistory_u :: [(Id, State)] -> History

def listsHistory_u(ll):
    return sdict(ll)

# historyToList :: History -> [(Id, State)]

def historiesList(hh):
    return list(hh.items())

# historiesSetVar :: History -> Set.Set Variable

def historiesSetVar(hh):
    svars = statesSetVar
    if len(hh)==0:
        return sset()
    for ss in hh.values():
        return svars(ss)
    return None

# historiesSetState :: History -> Set.Set State

def historiesSetState(hh):
    return sset(hh.values())

# historiesSize :: History -> Integer

def historiesSize(hh):
    return len(hh)

# pairHistoriesJoin :: History -> History -> History

def pairHistoriesJoin(hh,gg):
    llhh = listsHistory
    hhll = historiesList
    isJoin = pairStatesIsJoin
    sunion = pairStatesUnionLeft
    ll = []
    for (i,ss) in hhll(hh):
        if i in gg and isJoin(ss,gg[i]):
            ll.append((i,sunion(ss,gg[i])))
    return llhh(ll)

# pairHistoriesAdd :: History -> History -> Maybe History

def pairHistoriesAdd(hh,gg):
    llhh = listsHistory  
    hhll = historiesList
    vars = historiesSetVar
    if len(hh)==0:
        return gg
    elif len(gg)==0:
        return hh
    elif vars(hh)==vars(gg):
        ff = llhh([(IdPair((x,IdNull(None))), ss) for (x,ss) in hhll(hh)] +
            [(IdPair((IdNull(None),x)), ss) for (x,ss) in hhll(gg)])
        return ff
    return None

# pairHistoriesMultiply :: History -> History -> History

def pairHistoriesMultiply(hh,gg):
    llhh = listsHistory  
    hhll = historiesList
    sjoin = pairStatesUnionLeft
    isjoin = pairStatesIsJoin  
    ff = llhh([(IdPair((x,y)), sjoin(ss,tt)) for (x,ss) in hhll(hh)
            for (y,tt) in hhll(gg) if isjoin(ss,tt)])
    return ff


# newtype Classification = Classification (Map.Map State (Set.Set Id)) 

# historiesClassification :: History -> Classification

def historiesClassification(hh):
    gg = sdict()
    for (i,ss) in hh.items():
        if ss not in gg:
            gg[ss] = sset()
        gg[ss].add(i)
    return gg

# setVarsHistoriesReduce :: Set.Set Variable -> History -> History 

def setVarsHistoriesReduce(vv,hh):
    filt = setVarsStatesStateFiltered
    llhh = listsHistory
    hhll = historiesList
    return llhh([(i, filt(vv,ss)) for (i,ss) in hhll(hh) ])


# classificationsHistory :: Classification -> History

def classificationsHistory(gg):
    return listsHistory([(i,ss) for (ss,ii) in gg.items() for i in ii])

# classificationsList :: Classification -> [(State, (Set.Set Id))]

def classificationsList(gg):
    return list(gg.items())


# newtype Histogram = Histogram (Map.Map State Rational)

# listsHistogram :: [(State, Rational)] -> Maybe Histogram

def listsHistogram(ll):
    svars = statesSetVar
    if len(ll)>0:
        (rr,q) = ll[0]
        vv = svars(rr)
        for (ss, q) in ll:
            if svars(ss) != vv or q < 0:
                return None
    aa = sdict()
    for (ss, q) in ll:
        if ss in aa:
            aa[ss] = ratio(aa[ss] + q)
        else:
            aa[ss] = ratio(q)
    return aa
    
# listsHistogram_u :: [(State, Rational)] -> Histogram

def listsHistogram_u(ll):
    aa = sdict()
    for (ss, q) in ll:
        if ss in aa:
            aa[ss] = ratio(aa[ss] + q)
        else:
            aa[ss] = ratio(q)
    return aa

# histogramsList :: Histogram -> [(State, Rational)]

def histogramsList(aa):
    return list(aa.items())

# histogramEmpty :: Histogram

def histogramEmpty():
    return sdict()

# histogramsCardinality :: Histogram -> Integer

histogramsCardinality = len

# histogramsSetVar :: Histogram -> Set.Set Variable

def histogramsSetVar(aa):
    svars = statesSetVar
    if len(aa)==0:
        return sset()
    for ss in aa:
        return svars(ss)
    return None

histogramsVars = histogramsSetVar

# histogramsDimension :: Histogram -> Integer

def histogramsDimension(aa):
    vars = histogramsSetVar
    return len(vars(aa))

# histogramsMapVarsFrame :: Histogram -> Map.Map Variable Variable -> Maybe Histogram

def histogramsMapVarsFrame(aa,nn):
    ssll = statesList
    llss = listsState
    llaa = listsHistogram
    aall = histogramsList
    def srepl(ss):
        tt = sdict()
        for (v,w) in ssll(ss):
            if v in nn:
                tt[nn[v]] = w
            else:
                tt[v] = w
        return tt
    vv = histogramsSetVar(aa)
    ww = sset(nn.keys())
    xx = sset(nn.values())
    isBi = len(ww) == len(xx)
    isDisjoint = len(xx & (vv - ww)) == 0
    if isBi and isDisjoint:
        bb = llaa([(srepl(ss), q) for (ss,q) in aall(aa)])
        return bb
    return None

# histogramsStates :: Histogram -> Set.Set State

def histogramsStates(aa):
    return sset(aa.keys())

def histogramsSetState(aa):
    return histogramsStates(aa)

# histogramsStatesCount :: Histogram -> State -> Maybe Rational

def histogramsStatesCount(aa,ss):
    if ss in aa:
        return aa[ss]
    else:
        return None

# histogramsSize :: Histogram -> Rational

def histogramsSize(aa):
    p = ratio(0,1)
    for (ss,q) in aa.items():
        p = p + q
    return ratio(p)

# histogramsResize :: Rational -> Histogram -> Maybe Histogram

def histogramsResize(z,aa):
    size = histogramsSize
    aall = histogramsList
    y = size(aa)
    bb = sdict()
    if z >= 0 and y > 0:
        for (ss,q) in aall(aa):
            bb[ss] = ratio(q * z / y)
        return bb
    return None

# histogramScalar :: Rational -> Maybe Histogram

def histogramScalar(q):
    llaa = listsHistogram
    return llaa([(sdict(),q)])

# histogramsTrim :: Histogram -> Histogram

def histogramsTrim(aa):
    aall = histogramsList
    bb = sdict()
    for (ss,q) in aall(aa):
        if q > 0:
            bb[ss] = q
    return bb

# histogramsIsSingleton :: Histogram -> Bool

def histogramsIsSingleton(aa):
    trim = histogramsTrim
    return len(trim(aa)) == 1

# histogramSingleton :: State -> Rational -> Maybe Histogram

def histogramSingleton(ss,q):
    llaa = listsHistogram
    return llaa([(ss,q)])

# histogramsIsUniform :: Histogram -> Bool

def histogramsIsUniform(aa):
    return len(aa) == 0 or len(sset(aa.values())) == 1

# histogramsIsIntegral :: Histogram -> Bool

def histogramsIsIntegral(aa):
    aall = histogramsList
    return all([q.denominator == 1 for (ss,q) in aall(aa)])

# histogramsUnit :: Histogram -> Histogram

def histogramsUnit(aa):
    llaa = listsHistogram_u
    aall = histogramsList
    return llaa([(ss,1) for (ss,q) in aall(aa)])

# histogramsIsUnit :: Histogram -> Bool

def histogramsIsUnit(aa):
    unit = histogramsUnit
    return unit(aa) == aa

# setStatesHistogramUnit :: Set.Set State -> Maybe Histogram

def setStatesHistogramUnit(qq):
    llaa = listsHistogram
    return llaa([(ss,1) for ss in qq])

# histogramsEffective :: Histogram -> Histogram

def histogramsEffective(aa):
    unit = histogramsUnit
    trim = histogramsTrim
    return unit(trim(aa))

# histogramsSystemImplied :: Histogram -> System

def histogramsSystemImplied(aa):
    vars = histogramsSetVar
    lluu = listsSystem_u
    val = statesVarsValue
    states = histogramsStates
    qq = states(aa)
    return lluu([(v,sset([val(ss,v) for ss in qq])) for v in vars(aa)])

# histogramsSystemImplied_1 :: Histogram -> System

def histogramsSystemImplied_1(aa):
    vars = histogramsSetVar
    lluu = listsSystem_u
    val = statesVarsValue
    states = histogramsStates
    return lluu([(v,sset([val(ss,v) for ss in states(aa)])) for v in vars(aa)])


# type Component = Set.Set State

# newtype Partition = Partition (Set.Set Component)

# setComponentsPartition :: Set.Set Component -> Maybe Partition

def setComponentsPartition(qq):
    ispart = setsIsPartition
    svars = statesSetVar
    def okvars(qq):
        if len(qq) == 0:
            return False
        vv = svars(list(list(qq)[0])[0])
        for pp in qq:
            for ss in pp:
                if svars(ss) != vv:
                    return False
        return True
    if len(qq) == 0:
        return qq
    if ispart(qq) and okvars(qq):
        return qq
    return None

# partitionsSetComponent :: Partition -> Set.Set Component

def partitionsSetComponent(qq):
    return qq

# systemsSetVarsPartitionUnary :: System -> Set.Set Variable -> Maybe Partition

def systemsSetVarsPartitionUnary(uu,vv):
    cart = systemsSetVarsSetStateCartesian
    qq = cart(uu,vv)
    if qq != None:
        return sset([qq])
    return None

# systemsSetVarsPartitionSelf :: System -> Set.Set Variable -> Maybe Partition

def systemsSetVarsPartitionSelf(uu,vv):
    cart = systemsSetVarsSetStateCartesian
    qq = cart(uu,vv)
    if qq != None:
        return sset([sset([x]) for x in qq])
    return None


# histogramRegularCartesian :: Integer -> Integer -> Maybe Histogram

def histogramRegularCartesian(d,n):
    unit = setStatesHistogramUnit
    sysreg = systemRegular
    cart = systemsSetVarsSetStateCartesian
    uvars = systemsSetVar
    if d >= 1 and n >= 1:
        uu = sysreg(d,n)
        vv = uvars(uu)
        return unit(cart(uu,vv))
    return None

# histogramRegularUnitSingleton :: Integer -> Integer -> Maybe Histogram

def histogramRegularUnitSingleton(d,n):
    llss = listsState
    llaa = listsHistogram
    if d >= 1 and n >= 1:
        return llaa([(llss([(VarInt(j), ValInt(1)) for j in range(1,n+1)]),1)])
    return None

# histogramRegularUnitDiagonal :: Integer -> Integer -> Maybe Histogram

def histogramRegularUnitDiagonal(d,n):
    llss = listsState
    llaa = listsHistogram
    if d >= 1 and n >= 1:
        return llaa([(llss([(VarInt(j), ValInt(i)) for j in range(1,n+1)]),1) for i in range(1,d+1)])
    return None

# histogramRegularUnitPivot :: Integer -> Integer -> Maybe Histogram

def histogramRegularUnitPivot(d,n):
    uvars = systemsSetVar
    lluu = listsSystem
    single = histogramRegularUnitSingleton
    cart = systemsSetVarsSetStateCartesian
    unit = setStatesHistogramUnit
    add = pairHistogramsAdd
    if d == 1 and n >= 1:
        return single(d,n)
    bb = single(d,n)
    uu = lluu([(VarInt(i), sset([ValInt(j) for j in range(2,d+1)])) for i in range(1,n+1)])
    if d >= 1 and n >= 1:
        return add(single(d,n),unit(cart(uu,uvars(uu))))
    return None

# historiesHistogram :: History -> Histogram

def historiesHistogram(hh):
    llaa = listsHistogram
    return llaa([(ss,1) for ss in hh.values()])

# histogramsHistory :: Histogram -> Maybe History

def histogramsHistory(aa):
    isint = histogramsIsIntegral
    llhh = listsHistory  
    aall = histogramsList
    if isint(aa):
        return llhh([(IdStateInteger((ss,i)),ss) for (ss,q) in aall(aa) for i in range(1,q.numerator+1)])
    return None
        
# pairHistogramsLeq :: Histogram -> Histogram -> Bool

def pairHistogramsLeq(aa,bb):
    at = histogramsStatesCount
    states = histogramsStates
    trim = histogramsTrim
    aall = histogramsList
    aat = trim(aa)
    if not states(aat).issubset(states(bb)):
        return False
    for ss,q in aall(aat):
        if q > at(bb,ss):
            return False
    return True

# pairHistogramsAdd :: Histogram -> Histogram -> Maybe Histogram

def pairHistogramsAdd(aa,bb): 
    llaa = listsHistogram_u
    aall = histogramsList
    vars = histogramsSetVar
    if len(aa)==0:
        return bb
    if len(bb)==0:
        return aa
    if vars(aa)==vars(bb):
        return llaa(aall(aa) + aall(bb))
    return None

# pairHistogramsSubtract :: Histogram -> Histogram -> Maybe Histogram

def pairHistogramsSubtract(aa,bb): 
    aall = histogramsList
    vars = histogramsSetVar
    if len(bb)==0:
        return aa
    if vars(aa)==vars(bb):
        cc = aa.copy()
        for (ss, q) in aall(bb):
            if ss in cc:
                p = cc[ss] - q
                if p >= 0:
                    cc[ss] = ratio(p)
                else:
                    cc[ss] = ratio(0)
            else:
                cc[ss] = ratio(0)
        return cc
    return None

# pairHistogramsMultiply :: Histogram -> Histogram -> Histogram

def pairHistogramsMultiply(aa,bb): 
    llaa = listsHistogram_u
    aall = histogramsList
    sjoin = pairStatesUnionLeft
    def isjoin(ss,tt):
        ll = dict(ss)
        ll.update(tt.items())
        rr = dict(tt)
        rr.update(ss.items())
        return ll == rr
    jj = aall(aa)
    kk = aall(bb)
    return llaa([(sjoin(ss,tt),q*r) for ss,q in jj for tt,r in kk if isjoin(ss,tt)])

# pairHistogramsMultiply_1 :: Histogram -> Histogram -> Histogram

def pairHistogramsMultiply_1(aa,bb): 
    llaa = listsHistogram_u
    aall = histogramsList
    sjoin = pairStatesUnionLeft
    isjoin = pairStatesIsJoin
    return llaa([(sjoin(ss,tt),q*r) for ss,q in aall(aa) for tt,r in aall(bb) if isjoin(ss,tt)])

# pairHistogramsMultiply_2 :: Histogram -> Histogram -> Histogram

def pairHistogramsMultiply_2(aa,bb): 
    llaa = listsHistogram_u
    aall = histogramsList
    sjoin = pairStatesUnionLeft
    isjoin = pairStatesIsJoin
    jj = aall(aa)
    kk = aall(bb)
    return llaa([(sjoin(ss,tt),q*r) for ss,q in jj for tt,r in kk if isjoin(ss,tt)])


# histogramsReciprocal :: Histogram -> Histogram

def histogramsReciprocal(aa):
    llaa = listsHistogram_u
    aall = histogramsList
    trim = histogramsTrim
    return llaa([(ss,1/q) for ss,q in aall(trim(aa))])

# pairHistogramsDivide :: Histogram -> Histogram -> Histogram

def pairHistogramsDivide(aa,bb): 
    recip = histogramsReciprocal
    mul = pairHistogramsMultiply
    return mul(aa,recip(bb))

# setVarsHistogramsReduce :: Set.Set Variable -> Histogram -> Histogram 

def setVarsHistogramsReduce(vv,aa):
    red = setVarsStatesStateFiltered
    llaa = listsHistogram_u
    aall = histogramsList
    return llaa([(red(vv,ss),q) for ss,q in aall(aa)])

# histogramsIsCausal :: Histogram -> Bool

def histogramsIsCausal(aa):
    isfunc = relationsIsFunc
    split = setVarsSetStatesSplit
    def states(aa):
        return histogramsSetState(histogramsTrim(aa))
    dim = histogramsDimension
    vars = histogramsSetVar    
    power = setsPowerset
    if dim(aa) < 2:
        return False
    vv = vars(aa)
    for kk in power(vv):
        if kk != vv and isfunc(split(kk,states(aa))):
            return True            
    return False

# histogramsIsDiagonal :: Histogram -> Bool

def histogramsIsDiagonal(aa):
    def states(aa):
        return histogramsSetState(histogramsTrim(aa))
    intersect = pairStatesIntersection
    dim = histogramsDimension
    if dim(aa) < 2:
        return False
    vv = vars(aa)
    for ss in states(aa):
        for tt in states(aa):
            if ss != tt and len(intersect(ss,tt)) != 0:
                return False            
    return True

# histogramsIndependent :: Histogram -> Histogram

def histogramsIndependent(aa):
    vars = histogramsSetVar
    def red(aa,v):
        return setVarsHistogramsReduce(sset([v]),aa)
    mul = pairHistogramsMultiply
    scalar = histogramScalar
    size = histogramsSize
    dim = histogramsDimension
    empty = histogramEmpty()
    z = size(aa)
    d = dim(aa)
    if aa == empty:
        return empty
    if d == 0:
        return aa
    if z == 0:
        bb = scalar(0)
    else:
        bb = scalar(ratio(z,z**d))
    for v in vars(aa):
        bb = mul(bb,red(aa,v))
    return bb

# setVarsSetVarsSetHistogramsHistogramsApply :: Set.Set Variable -> Set.Set Variable -> Set.Set Histogram -> Histogram -> Histogram

def setVarsSetVarsSetHistogramsHistogramsApply(vv,ww,mm,aa):
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    vars = histogramsVars
    card = histogramsCardinality
    def red(aa,vv):
        if vars(aa).issubset(vv):
            return aa
        return setVarsHistogramsReduce(vv,aa)
    mul = pairHistogramsMultiply
    def qvars(qq):
        yy = sset()
        for bb in qq:
            yy |= vars(bb)
        return yy
    if len(mm) == 0:
        return red(aa,ww)       
    xx = []
    for dd in mm:
        if len(vars(dd)&(vars(aa)|vv)) > 0:
            qq = mm.copy()
            qq.remove(dd)
            bb = red(mul(aa,dd),ww|qvars(qq))
            xx.append((card(bb),bb,qq))
    if len(xx) == 0:
        return red(aa,ww)         
    (_,cc,nn) = list(sset(xx))[0]
    return apply(vv,ww,nn,cc)

# setVarsHistogramsSlices :: Set.Set Variable -> Histogram -> Map.Map State Histogram

def setVarsHistogramsSlices(kk,aa):
    def single(ss):
        return histogramSingleton(ss,1)
    def red(aa,vv):
        return setVarsHistogramsReduce(vv,aa)
    states =  histogramsSetState
    mul = pairHistogramsMultiply
    return sdict([(rr,mul(aa,single(rr))) for rr in states(red(aa,kk))])

# setVarsHistogramsSliceModal :: Set.Set Variable -> Histogram -> Rational

def setVarsHistogramsSliceModal(kk,aa):
    slices = setVarsHistogramsSlices
    size = histogramsSize
    aall = histogramsList
    def aamax(aa):
        if size(aa) > 0:
            return list(sset([c for (_,c) in aall(aa)]))[-1]
        return 0
    return ratio(sum([aamax(cc) for (rr,cc) in slices(kk,aa).items()]))
    
# newtype Transform = Transform (Histogram, (Set.Set Variable))

# histogramsSetVarsTransform :: Histogram -> Set.Set Variable -> Maybe Transform

def histogramsSetVarsTransform(xx,ww):
    vars = histogramsSetVar
    if ww.issubset(vars(xx)):
        return (xx,ww)
    return None

# histogramsSetVarsTransform_u :: Histogram -> Set.Set Variable -> Transform

def histogramsSetVarsTransform_u(xx,ww):
    return (xx,ww)

# transformEmpty :: Transform

def transformEmpty():
    return (histogramEmpty(),sset())

# histogramsTransformDisjoint :: Histogram -> Transform

def histogramsTransformDisjoint(xx):
    vars = histogramsSetVar
    return (xx,vars(xx))

# histogramsTransformNull :: Histogram -> Transform

def histogramsTransformNull(xx):
    return (xx,sset())

# transformsHistogram :: Transform -> Histogram

def transformsHistogram(tt):
    (xx,ww) = tt
    return xx

# transformsDerived :: Transform -> Set.Set Variable

def transformsDerived(tt):
    (xx,ww) = tt
    return ww

# transformsUnderlying :: Transform -> Set.Set Variable

def transformsUnderlying(tt):
    vars = histogramsSetVar
    (xx,ww) = tt
    return vars(xx) - ww

# transformsVars :: Transform -> Set.Set Variable

def transformsVars(tt):
    vars = histogramsSetVar
    (xx,ww) = tt
    return vars(xx) 

# transformsIsFunc :: Transform -> Bool

def transformsIsFunc(tt):
    filt = setVarsStatesStateFiltered
    und = transformsUnderlying
    der = transformsDerived
    ttaa = transformsHistogram
    trim = histogramsTrim
    states = histogramsStates
    ww = der(tt)
    yy = und(tt)
    xx = trim(ttaa(tt))
    return relationsIsFunc(sset([(filt(yy,ss), filt(ww,ss)) for ss in states(xx)]))

# transformsHistogramsApply :: Transform -> Histogram -> Histogram

def transformsHistogramsApply(tt,aa):
    mul = pairHistogramsMultiply
    red = setVarsHistogramsReduce
    (xx,ww) = tt
    return red(ww,mul(aa,xx))

# transformsInverse :: Transform -> Map.Map State Histogram

def transformsInverse(tt):
    und = transformsUnderlying
    der = transformsDerived
    ttaa = transformsHistogram
    substate = pairStatesIsSubstate
    filt = setVarsStatesStateFiltered
    states = histogramsStates
    red = setVarsHistogramsReduce
    llaa = listsHistogram_u
    aall = histogramsList
    ww = der(tt)
    yy = und(tt)
    xx = ttaa(tt)
    ii = sdict()
    for rr in states(red(ww,xx)):
        ii[rr] = llaa([(filt(yy,ss),q) for (ss,q) in aall(xx) if substate(rr,ss)])
    return ii

# transformsHistoriesApply :: Transform -> History -> History

def transformsHistoriesApply(tt,hh):
    llhh = listsHistory  
    hhll = historiesList
    def sunit(ss):
        return histogramSingleton(ss,1)
    states = histogramsStates
    eff = histogramsEffective
    def tmul(aa,tt):
        return transformsHistogramsApply(tt,aa)
    ll = list()
    for (i,ss) in hhll(hh):
        rrs = list(states(eff(tmul(sunit(ss),tt))))
        if len(rrs) == 1:
            ll = ll + [(i,rrs[0])]
    return llhh(ll)

# partitionsTransformVarPartition ::  Partition -> Transform

def partitionsTransformVarPartition(pp):
    unit = setStatesHistogramUnit
    ppqq = partitionsSetComponent
    sunion = pairStatesUnionLeft
    single = stateSingleton
    trans = histogramsSetVarsTransform_u
    w = VarPartition(pp)
    ww = sset([w])
    xx = unit([sunion(ss,single(w,ValComponent(cc))) for cc in ppqq(pp) for ss in cc])
    return trans(xx,ww)

# transformsPartition :: Transform -> Partition

def transformsPartition(tt):
    return sset([histogramsSetState(cc) for (_,cc) in transformsInverse(tt).items()])

# transformsConverseNatural :: Transform -> Transform

def transformsConverseNatural(tt):
    trans = histogramsSetVarsTransform_u
    eff = histogramsEffective
    def red(aa,vv):
        return setVarsHistogramsReduce(vv,aa)
    div = pairHistogramsDivide
    und = transformsUnderlying
    der = transformsDerived
    ttaa = transformsHistogram
    ww = der(tt)
    yy = und(tt)
    xx = ttaa(tt)
    return trans(div(eff(xx),red(xx,ww)),yy)

# histogramsTransformsConverseActual :: Histogram -> Transform -> Maybe Transform

def histogramsTransformsConverseActual(bb,tt):
    isfunc = transformsIsFunc 
    trans = histogramsSetVarsTransform_u
    def norm(aa):
        return histogramsResize(1,aa)
    size = histogramsSize
    def inv(tt):
        return list(transformsInverse(tt).items())
    add = pairHistogramsAdd
    mul = pairHistogramsMultiply
    def unit(rr): 
        return setStatesHistogramUnit(sset([rr]))
    empty = histogramEmpty
    und = transformsUnderlying
    vars = histogramsSetVar
    yy = und(tt)
    if isfunc(tt) and vars(bb) == yy:
        ee = empty()
        for (rr,cc) in inv(tt):
            dd = mul(bb,cc)
            if size(dd) > 0:
                ee = add(ee,mul(norm(dd),unit(rr)))
        return trans(ee,yy)
    return None

# histogramsTransformsConverseIndependent :: Histogram -> Transform -> Maybe Transform

def histogramsTransformsConverseIndependent(bb,tt):
    isfunc = transformsIsFunc 
    trans = histogramsSetVarsTransform_u
    def norm(aa):
        return histogramsResize(1,aa)
    size = histogramsSize
    def inv(tt):
        return list(transformsInverse(tt).items())
    add = pairHistogramsAdd
    mul = pairHistogramsMultiply
    def unit(rr): 
        return setStatesHistogramUnit(sset([rr]))
    empty = histogramEmpty
    und = transformsUnderlying
    vars = histogramsSetVar
    ind = histogramsIndependent
    yy = und(tt)
    if isfunc(tt) and vars(bb) == yy:
        ee = empty()
        for (rr,cc) in inv(tt):
            dd = mul(bb,cc)
            if size(dd) > 0:
                ee = add(ee,mul(norm(ind(dd)),unit(rr)))
        return trans(ee,yy)
    return None


# newtype Fud = Fud (Set.Set Transform)

# setTransformsFud :: Set.Set Transform -> Maybe Fud

def setTransformsFud(qq):
    for (aa,ww) in qq:
        for (bb,xx) in qq:
            if (aa,ww) != (bb,xx) and len(ww & xx) > 0:
                return None
    return qq

# setTransformsFud_u :: Set.Set Transform -> Fud

def setTransformsFud_u(qq):
    return qq

# fudsSetTransform :: Fud -> Set.Set Transform

def fudsSetTransform(ff):
    return ff

# fudsSetHistogram :: Fud -> Set.Set Histogram 

def fudsSetHistogram(ff):
    return sset([aa for (aa,ww) in ff])

# fudEmpty :: Fud

def fudEmpty():
    return sset()

# fudsVars :: Fud -> Set.Set Variable

def fudsVars(ff):
    vars = histogramsSetVar
    vv = set()
    for (aa,ww) in ff:
        vv |= vars(aa)
    return sset(vv)

# fudsVars_1 :: Fud -> Set.Set Variable

def fudsVars_1(ff):
    vars = histogramsSetVar
    vv = sset()
    for (aa,ww) in ff:
        vv = vv | vars(aa)
    return vv

fudsSetVar = fudsVars

# fudsDerived :: Fud -> Set.Set Variable

def fudsDerived(ff):
    und = transformsUnderlying
    vv = set()
    for (aa,ww) in ff:
        vv |= ww
    for tt in ff:
        vv -= und(tt)
    return sset(vv)

# fudsDerived_1 :: Fud -> Set.Set Variable

def fudsDerived_1(ff):
    und = transformsUnderlying
    vv = sset()
    for (aa,ww) in ff:
        vv = vv | ww
    for tt in ff:
        vv = vv - und(tt)
    return vv

# fudsSystemImplied :: Fud -> System

def fudsSystemImplied(ff):
    sys = histogramsSystemImplied
    his = fudsSetHistogram
    empty = systemEmpty
    uunion = pairSystemsUnion
    uu = empty()
    for aa in his(ff):
        uu = uunion(uu,sys(aa))
    return uu

# fudsUnderlying :: Fud -> Set.Set Variable

def fudsUnderlying(ff):
    und = transformsUnderlying
    vv = set()
    for tt in ff:
        vv |= und(tt)
    for (aa,ww) in ff:
        vv -= ww
    return sset(vv)

# fudsUnderlying_1 :: Fud -> Set.Set Variable

def fudsUnderlying_1(ff):
    und = transformsUnderlying
    vv = sset()
    for tt in ff:
        vv = vv | und(tt)
    for (aa,ww) in ff:
        vv = vv - ww
    return vv

# fudsTransform :: Fud -> Transform

def fudsTransform(ff):
    fder = fudsDerived
    fund = fudsUnderlying
    mul = pairHistogramsMultiply
    def red(aa,vv):
        return setVarsHistogramsReduce(vv,aa)
    fhis = fudsSetHistogram
    scalar = histogramScalar
    trans = histogramsSetVarsTransform_u
    if len(ff) == 0:
        return transformEmpty()
    aa = scalar(1)
    for bb in fhis(ff):
        aa = mul(aa,bb)
    aa = red(aa, fder(ff) | fund(ff))
    return trans(aa,fder(ff))

# fudsDefinitions :: Fud -> Map.Map Variable Transform

def fudsDefinitions(ff):
    ffqq = fudsSetTransform
    der = transformsDerived
    return sdict([(v,tt) for tt in ffqq(ff) for v in der(tt)])

# fudsVarsDepends :: Fud -> Set.Set Variable -> Fud

def fudsVarsDepends(ff,ww):
    und = transformsUnderlying
    dd = fudsDefinitions(ff)
    yy = sset(dd.keys())
    def deps(uu,xx):
        ff = sset()
        for w in uu & yy - xx:
            tt = dd[w]
            ff.add(tt)
            zz = xx.copy()
            zz.add(w)
            ff |= deps(und(tt),zz)
        return ff
    return deps(ww,sset())

fudsSetVarsDepends = fudsVarsDepends

# fudsSetVarsLayer :: Fud -> Set.Set Variable -> Integer

def fudsSetVarsLayer(ff,ww):
    und = transformsUnderlying
    fund = fudsUnderlying
    dd = fudsDefinitions(ff)
    yy = sset(dd.keys())
    def layer(uu,xx):
        i = 0
        for w in (uu & yy - xx):
            tt = dd[w]
            zz = xx.copy()
            zz.add(w)
            j = layer(und(tt),zz) + 1
            if j > i:
                i = j
        return i
    return layer(ww,fund(ff))

# fudsOverlap :: Fud -> Bool

def fudsOverlap(ff):
    fvars = fudsVars
    fder = fudsDerived
    def dep(ff,v):
        return fudsVarsDepends(ff,sset([v]))
    for v in fder(ff):
        for w in fder(ff):
            if w != v and len(fvars(dep(ff,v)) & fvars(dep(ff,w))) > 0:
                return True
    return False

# setVarsFudHistogramsApply :: Set.Set Variable -> Fud -> Histogram -> Maybe Histogram

def setVarsFudHistogramsApply(ww,ff,aa):
    apply = setVarsFudHistogramsApply
    fvars = fudsVars
    und = transformsUnderlying
    his = transformsHistogram
    mul = pairHistogramsMultiply
    vars = histogramsVars
    def red(aa,vv):
        if vars(aa).issubset(vv):
            return aa
        return setVarsHistogramsReduce(vv,aa)
    if len(ff) == 0:
        return red(aa,ww)
    vv = vars(aa)
    xx = sset([tt for tt in ff if und(tt).issubset(vv)])
    if len(xx) == 0:
        return None
    tt = xx[0]
    gg = ff.copy()
    gg.remove(tt)
    cc = red(mul(aa,his(tt)),ww|fvars(gg))
    return apply(ww,gg,cc)

# fudsHistogramsApply :: Fud -> Histogram -> Histogram

def fudsHistogramsApply(ff,aa):
    fder = fudsDerived
    fund = fudsUnderlying
    ffqq = fudsSetHistogram
    vars = histogramsVars
    applyFud = setVarsFudHistogramsApply
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    if fund(ff).issubset(vars(aa)):
        return applyFud(fder(ff),ff,aa)
    return apply(vars(aa)|fund(ff),fder(ff),ffqq(ff),aa)

# fudsHistogramsMultiply :: Fud -> Histogram -> Histogram

def fudsHistogramsMultiply(ff,aa):
    fder = fudsDerived
    fund = fudsUnderlying
    ffqq = fudsSetHistogram
    vars = histogramsVars
    applyFud = setVarsFudHistogramsApply
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    if fund(ff).issubset(vars(aa)):
        return applyFud(vars(aa)|fder(ff),ff,aa)
    return apply(vars(aa)|fund(ff),vars(aa)|fder(ff),ffqq(ff),aa)


# newtype DecompFud = DecompFud (Tree (State,Fud))

# decompFudsSetFud :: DecompFud -> Set.Set Fud 

def decompFudsSetFud(zz):
    ran = relationsRange
    elem = treesElements
    return ran(elem(zz))

decompFudsFuds = decompFudsSetFud

# decompFudsFud :: DecompFud -> Fud 

def decompFudsFud(dd):
    fuds = decompFudsFuds
    bigcup = setSetsUnion
    return bigcup(fuds(dd))

# decompFudsUnderlying :: DecompFud -> Set.Set Variable

def decompFudsUnderlying(dd):
    ddff = decompFudsFud
    fund = fudsUnderlying
    return fund(ddff(dd))

# decompFudEmpty :: DecompFud

def decompFudEmpty():
    return sdict([((stateEmpty(),fudEmpty()),emptyTree())])

# treePairStateFudsDecompFud :: Tree (State,Fud) -> Maybe DecompFud

def treePairStateFudsDecompFud(zz):
    fvars = fudsVars
    fder = fudsDerived
    fund = fudsUnderlying
    fsys = fudsSystemImplied
    cart = systemsSetVarsSetStateCartesian
    def std(ff):
        return cart(fsys(ff),fder(ff))
    dom = relationsDomain
    ran = relationsRange
    elem = treesElements
    roots = treesRoots
    steps = treesSteps
    def okFuds(ff1,ff2):
        return all([len(ww & xx) == 0 
            for (aa,ww) in ff1 for (bb,xx) in ff2 if (aa,ww) != (bb,xx)])
    def okVars(zz):
        qq = ran(elem(zz))
        return all([len((fvars(ff1) - fund(ff1)) & fund(ff2)) == 0 and okFuds(ff1,ff2)
            for ff1 in qq for ff2 in qq if ff1 != ff2])
    def okRoots(zz):
        return dom(roots(zz)) == sset([stateEmpty()])
    def okStates(zz):
        return all([ss in std(ff) for ((_,ff),(ss,_)) in steps(zz)])
    if okVars(zz) and okRoots(zz) and okStates(zz):
        return zz
    return None

# decompFudsTreePairStateFud :: DecompFud -> Tree (State,Fud)

def decompFudsTreePairStateFud(zz):
    return zz

# decompFudsHistogramsApply :: DecompFud -> Histogram -> Tree (State,Histogram)

def decompFudsHistogramsApply(zz,aa):
    fder = fudsDerived
    fund = fudsUnderlying      
    ffqq = fudsSetHistogram
    def red(aa,vv):
        return setVarsHistogramsReduce(vv,aa)
    vars = histogramsSetVar
    mul = pairHistogramsMultiply
    single = histogramSingleton
    size = histogramsSize
    empty = histogramEmpty
    applyFud = setVarsFudHistogramsApply
    applyHis = setVarsSetVarsSetHistogramsHistogramsApply
    def apply(zz,vv,aa):
        ll = []
        for ((ss,ff),yy) in zz.items():
            aa1 = mul(aa,single(ss,1))
            ww = fder(ff)
            uu = fund(ff)
            bb = empty()
            if size(aa1) > 0:
                if uu.issubset(vv):
                    bb = applyFud(vv|ww,ff,aa1)
                else:
                    bb = applyHis(vv|uu,vv|ww,ffqq(ff),aa1)
            ll.append(((ss,red(bb,ww)),apply(yy,vv,bb)))
        return sdict(ll)
    return apply(zz,vars(aa),aa)

# decompFudsHistogramsMultiply :: DecompFud -> Histogram -> Tree (State,Histogram)

def decompFudsHistogramsMultiply(zz,aa):
    fder = fudsDerived
    fund = fudsUnderlying      
    ffqq = fudsSetHistogram
    vars = histogramsSetVar
    mul = pairHistogramsMultiply
    single = histogramSingleton
    size = histogramsSize
    empty = histogramEmpty
    applyFud = setVarsFudHistogramsApply
    applyHis = setVarsSetVarsSetHistogramsHistogramsApply
    def apply(zz,vv,aa):
        ll = []
        for ((ss,ff),yy) in zz.items():
            aa1 = mul(aa,single(ss,1))
            ww = fder(ff)
            uu = fund(ff)
            bb = empty()
            if size(aa1) > 0:
                if uu.issubset(vv):
                    bb = applyFud(vv|ww,ff,aa1)
                else:
                    bb = applyHis(vv|uu,vv|ww,ffqq(ff),aa1)
            ll.append(((ss,bb),apply(yy,vv,bb)))
        return sdict(ll)
    return apply(zz,vars(aa),aa)

# decompFudsHistogramsHistogramsQuery :: DecompFud -> Histogram -> Histogram -> Tree (State,Histogram)

def decompFudsHistogramsHistogramsQuery(zz,aa,qq):
    fder = fudsDerived
    fund = fudsUnderlying      
    ffqq = fudsSetHistogram
    vars = histogramsSetVar
    mul = pairHistogramsMultiply
    def red(aa,vv):
        return setVarsHistogramsReduce(vv,aa)
    single = histogramSingleton
    size = histogramsSize
    empty = histogramEmpty
    applyFud = setVarsFudHistogramsApply
    applyHis = setVarsSetVarsSetHistogramsHistogramsApply
    vv = vars(aa)
    kk = vars(qq)
    def query(zz,aa,qq):
        ll = []
        for ((ss,ff),yy) in zz.items():
            ww = fder(ff)
            uu = fund(ff)
            qq1 = mul(qq,single(ss,1))
            if size(qq1) > 0:
                rr = empty()
                if uu.issubset(kk):
                    rr = applyFud(vv|ww,ff,qq1)
                else:
                    rr = applyHis(vv|kk,vv|ww,ffqq(ff),qq1)
                aa1 = mul(aa,single(ss,1))
                bb = empty()
                if size(aa1) > 0:
                    if uu.issubset(vv):
                        bb = mul(applyFud(vv|ww,ff,aa1),red(rr,ww))
                    else:
                        bb = mul(applyHis(vv|uu,vv|ww,ffqq(ff),aa1),red(rr,ww))
                ll.append(((ss,bb),query(yy,bb,rr)))
        return sdict(ll)
    return query(zz,aa,qq)


# newtype RollValue = RollValue (Set.Set Variable, Variable, Value, Value)

# setVariablesVariablesValuesValuesRollValue :: (Set.Set Variable, Variable, Value, Value) -> Maybe RollValue

def setVariablesVariablesValuesValuesRollValue(r):
    (vv,v,s,t) = r
    if v in vv:
        return r
    return None

# rollValuesSetVariableVariableValueValue :: RollValue -> (Set.Set Variable, Variable, Value, Value)

def rollValuesSetVariableVariableValueValue(r):
    return r


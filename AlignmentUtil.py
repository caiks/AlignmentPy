from sortedcontainers import *
from fractions import Fraction
from math import *

class hlist(list):
    def __hash__(self):
        return 0

# cached hash collections are semi-frozen - ought to raise error on modification after hashing

class slist(list):
    def __init__(self, rep=None):
        list.__init__(self, rep)
        self._hash = 0
    def __lt__(self,other):
        for (a,b) in zip(self,other):
            if a < b:
                return True
            elif a > b:
                return False
        if len(self) < len(other):
            return True
        return False
    def __gt__(self,other):
        for (a,b) in zip(self,other):
            if a > b:
                return True
            elif a < b:
                return False
        if len(self) > len(other):
            return True
        return False
    def __eq__(self,other):
        if len(self) == len(other):
            for (a,b) in zip(self,other):
                if not a == b:
                    return False
            return True
        return False
    def __le__(self,other):
        return not self > other
    def __ge__(self,other):
        return not self < other
    def __hash__(self):
        if self._hash == 0:
            self._hash = hash(tuple(self))
        return self._hash

class sset(SortedSet):
    def __init__(self, iterable=None, key=None):
        SortedSet.__init__(self, iterable, key)
        self._hash = 0
    def __str__(self):
        return "{" + str(list(self))[1:-1] + "}"
    def __repr__(self):
        return str(self)
    def __lt__(self,other):
        for (a,b) in zip(self,other):
            if a < b:
                return True
            elif a > b:
                return False
        if len(self) < len(other):
            return True
        return False
    def __gt__(self,other):
        for (a,b) in zip(self,other):
            if a > b:
                return True
            elif a < b:
                return False
        if len(self) > len(other):
            return True
        return False
    def __eq__(self,other):
        if len(self) == len(other):
            for (a,b) in zip(self,other):
                if not a == b:
                    return False
            return True
        return False
    def __le__(self,other):
        return not self > other
    def __ge__(self,other):
        return not self < other
    def __hash__(self):
        if self._hash == 0:
            self._hash = hash(frozenset(self))
        return self._hash

class sdict(SortedDict):
    def __init__(self, *args, **kwargs):
        SortedDict.__init__(self, *args, **kwargs)
        self._hash = 0
    def __str__(self):
        return "{" + str(list(self.items()))[1:-1] + "}"
    def __repr__(self):
        return str(self)
    def __lt__(self,other):
        for (a,b) in zip(self.items(),other.items()):
            if a < b:
                return True
            elif a > b:
                return False
        if len(self) < len(other):
            return True
        return False
    def __gt__(self,other):
        for (a,b) in zip(self.items(),other.items()):
            if a > b:
                return True
            elif a < b:
                return False
        if len(self) > len(other):
            return True
        return False
    def __eq__(self,other):
        if len(self) == len(other):
            for (a,b) in zip(self.items(),other.items()):
                if not a == b:
                    return False
            return True
        return False
    def __le__(self,other):
        return not self > other
    def __ge__(self,other):
        return not self < other
    def __hash__(self):
        if self._hash == 0:
            self._hash = hash(frozenset(self.items()))
        return self._hash

class ratio(Fraction):
    def __str__(self):
        return str(self.numerator) + " % " + str(self.denominator)
    def __repr__(self):
        return str(self)


# setSetsUnion :: Ord a => Set.Set (Set.Set a) -> Set.Set a

def setSetsUnion(qq):
    rr = sset()
    for pp in qq:
        rr |= pp
    return rr

# listSetsProduct :: Ord a => [Set.Set a] -> [[a]]

def listSetsProduct(ll):
    xx = [[]]
    for qq in ll:
        xx = [jj + [x] for jj in xx for x in qq]
    return xx       

# setsIsPartition :: Ord a => Set.Set (Set.Set a) -> Bool

def setsIsPartition(qq):
    if len(qq) == 0 or sset() in qq:
        return False
    l = 0
    rr = sset()
    for pp in qq:
        l = l + len(pp)
        rr |= pp
    if len(rr) != l:
        return False
    return True

# setsSetPartition :: Ord a => Set.Set a -> Set.Set (Set.Set (Set.Set a))

def setsSetPartition(ss):
    def sgl(x):
        return sset([x])
    if len(ss) == 0:
        return sset()
    ss2 = ss.copy()
    ss2.remove(ss[0])
    qq = sgl(sgl(sgl(ss[0])))
    for x in ss2:
        qq2 = sset()
        for pp in qq:
            pp2 = pp.copy()   
            pp2.add(sgl(x))
            qq2.add(pp2)
            for cc in pp:
                pp2 = pp.copy()
                pp2.remove(cc)
                cc2 = cc.copy()
                cc2.add(x)
                pp2.add(cc2)
                qq2.add(pp2)
        qq = qq2
    return qq
                
# setsSetPartitionLimited :: Ord a => Set.Set a -> Integer -> Set.Set (Set.Set (Set.Set a))

def setsSetPartitionLimited(ss,k):
    def sgl(x):
        return sset([x])
    if len(ss) == 0 or k <= 0:
        return sset()
    if k >= len(ss):
        return setsSetPartition(ss)
    ss2 = ss.copy()
    ss2.remove(ss[0])
    qq = sgl(sgl(sgl(ss[0])))
    for x in ss2:
        qq2 = sset()
        for pp in qq:
            if len(pp) < k:
                pp2 = pp.copy()   
                pp2.add(sgl(x))
                qq2.add(pp2)
            for cc in pp:
                pp2 = pp.copy()
                pp2.remove(cc)
                cc2 = cc.copy()
                cc2.add(x)
                pp2.add(cc2)
                qq2.add(pp2)
        qq = qq2
    return qq
   
# setsSetPartitionFixed :: Ord a => Set.Set a -> Integer -> Set.Set (Set.Set (Set.Set a))

def setsSetPartitionFixed(ss,k):
    if len(ss) == 0 or k <= 0:
        return sset()
    if k > len(ss):
        return sset()
    return setsSetPartitionLimited(ss,k) - setsSetPartitionLimited(ss,k-1)

# setsPowerset :: Ord a => Set.Set a -> Set.Set (Set.Set a)

def setsPowerset(ss):
    pp = sset([sset()])
    for x in ss:
        qq = pp.copy()
        for xx in qq:
            yy = xx.copy()
            yy.add(x)
            pp.add(yy)
    return pp

# relationsIsFunc :: (Ord a) => Set.Set (a,b) -> Bool

def relationsIsFunc(ss):
    return len(sdict(list(ss))) == len(ss)

# relationsDomain :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set a

def relationsDomain(ss):
    return sset([x for (x,y) in ss])

# relationsRange :: (Ord a, Ord b) => Set.Set (a,b) -> Set.Set b

def relationsRange(ss):
    return sset([y for (x,y) in ss])

# relationsSum :: (Ord a, Ord b, Num b) => Set.Set (a,b) -> b

def relationsSum(qq):
    return sum([y for (_,y) in qq])

# relationsNormalise :: (Ord a, Ord b, Fractional b) => Set.Set (a,b) -> Set.Set (a,b)

def relationsNormalise(qq):
    s = relationsSum(qq)
    return [(x,y/s) for (x,y) in qq]

# functionsInverse :: (Ord a, Ord b) => Map.Map a b -> Map.Map b (Set.Set a)

def functionsInverse(mm):
    nn = sdict()
    for (x,y) in mm.items():
        if y not in nn:
            nn[y] = sset()
        nn[y] |= sset([x])
    return nn


# data Tree a = Tree (Map.Map a (Tree a))

# emptyTree :: Tree a

def emptyTree():
    return sdict()

# treesRelation :: (Ord a, Ord (Tree a)) => Tree a -> Set.Set (a, Tree a)

def treesRelation(tt):
     return sset(tt.items())

# treesNodes :: (Ord a, Ord (Tree a)) => Tree a -> Set.Set (a, Tree a)

def treesNodes(tt):
    rel = treesRelation
    nodes = treesNodes
    ss = rel(tt)
    for (x,rr) in tt.items():
        ss |= nodes(rr)
    return ss

# treesSteps :: (Ord a, Ord (Tree a)) => Tree a -> Set.Set (a, a)

def treesSteps(tt):
    nodes = treesNodes
    return sset([(x,y) for (x,rr) in nodes(tt) for y in rr])

# treesElements :: (Ord a, Ord (Tree a)) => Tree a -> Set.Set a

def treesElements(tt):
    elements = treesElements
    ss = sset(tt.keys())
    for (x,rr) in tt.items():
        ss |= elements(rr)
    return ss

# treesElements_1 :: (Ord a, Ord (Tree a)) => Tree a -> Set.Set a

def treesElements_1(tt):
    return relationsDomain(treesNodes(tt))

# treesRoots :: (Ord a, Ord (Tree a)) => Tree a -> Set.Set a

def treesRoots(tt):
    return relationsDomain(treesRelation(tt))

# treesLeaves :: (Ord a, Ord (Tree a)) => Tree a -> Set.Set a

def treesLeaves(tt):
    nodes = treesNodes
    return sset([x for (x,rr) in nodes(tt) if len(rr) == 0])

# pairTreesUnion :: (Ord a, Ord (Tree a)) => Tree a -> Tree a -> Tree a

def pairTreesUnion(ss,tt):
    union = pairTreesUnion
    if len(tt) == 0:
        return ss
    if len(ss) == 0:
        return tt
    rr = ss.copy()
    for x in tt:
        if x in ss:
            rr[x] = union(ss[x],tt[x])
        else:
            rr[x] = tt[x]
    return rr

# treesPaths :: (Ord a, Ord (Tree a)) => Tree a -> [[a]]

def treesPaths(tt):
    def listsTreesPaths(ll,xx):
        paths = listsTreesPaths
        rel = treesRelation
        if len(xx) == 0:
            return [ll]
        qq = []
        for (x,rr) in rel(xx):
            qq = qq + paths(ll+[x],rr)
        return qq
    return listsTreesPaths([],tt)

# pathsTree :: (Ord a, Ord (Tree a)) => [[a]] -> Tree a

def pathsTree(qq):
    empty = emptyTree
    tree = pathsTree
    union = pairTreesUnion
    def lltt(jj):
        if len(jj) == 0:
            return empty()
        if len(jj) == 1:
            return sdict([(jj[0],empty())])
        return sdict([(jj[0],tree([jj[1:]]))])
    tt = empty()
    for ll in qq:
        tt = union(tt,lltt(ll))
    return tt

# treeRegular :: Integer -> Integer -> Tree [Integer]

def treeRegular(k,h):
    def reg(k,h,ll):
        if h == 0:
            return emptyTree()
        return sdict([(slist(ll+[i]),reg(k,h-1,ll+[i])) for i in range(1,k+1)])
    if k > 0 and h > 0:
        return reg(k,h,[])
    return emptyTree()   

# funcsTreesMapNode :: (Ord a, Ord (Tree a), Ord b, Ord (Tree b)) => (a -> Tree a -> b) -> Tree a -> Tree b

def funcsTreesMapNode(ff,mm):
    return sdict([(ff(k,xx), funcsTreesMapNode(ff,xx)) for (k,xx) in mm.items()])
    
# funcsTreesMap :: (Ord a, Ord (Tree a), Ord b, Ord (Tree b)) => (a -> b) -> Tree a -> Tree b

def funcsTreesMap(ff,tt):
    gg = lambda x,_: ff(x)
    return funcsTreesMapNode(gg,tt)
  
# funcsTreesMapNodeAccum :: (Ord a, Ord (Tree a), Ord b, Ord (Tree b)) => ([a] -> Tree a -> b) -> Tree a -> Tree b

def funcsTreesMapNodeAccum(ff,tt):
    def accum(ff,ll,zz):
        ss = sdict()
        for (x,rr) in zz.items():
            mm = slist(ll)
            mm.append(x)
            ss[ff(mm,rr)] = accum(ff,mm,rr)
        return ss
    return accum(ff,slist([]),tt)
   
# funcsTreesMapAccum :: (Ord a, Ord (Tree a), Ord b, Ord (Tree b)) => ([a] -> b) -> Tree a -> Tree b

def funcsTreesMapAccum(ff,tt):
    gg = lambda x,_: ff(x)
    return funcsTreesMapNodeAccum(gg,tt)
   
# treesPlaces :: (Ord a, Ord (Tree a)) => Tree a -> Set.Set ([a], Tree a)

def treesPlaces(tt):
    ff = lambda ll,rr: (ll,rr)
    return treesElements(funcsTreesMapNodeAccum(ff,tt))

# treesSubPaths :: (Ord a, Ord (Tree a)) => Tree a -> Set.Set [a]

def treesSubPaths(tt):
    return relationsDomain(treesPlaces(tt))

# funcsListsTreesTraversePreOrder :: (Ord a, Ord (Tree a), Ord c, Ord (Tree c)) => (a -> b -> c) -> [b] -> Tree a -> (Tree c,[b])

def funcsListsTreesTraversePreOrder(ff,ll,mm):
    traverse = funcsListsTreesTraversePreOrder
    def next(nn,jj,kk):
        if len(nn) == 0:
            return (sdict(kk),jj)
        ((a,xx),b) = nn[0]
        (tt,ii) = traverse(ff,jj,xx)
        return next(nn[1:],ii,[(ff(a,b),tt)]+kk)
    return next(list(zip(mm.items(),ll)),ll[len(mm):],[])

# funcsListsTreesTraverseInOrder :: (Ord a, Ord (Tree a), Eq b, Ord c, Ord (Tree c)) => (a -> b -> c) -> [b] -> Tree a -> (Tree c,[b])

def funcsListsTreesTraverseInOrder(ff,ll,mm):
    traverse = funcsListsTreesTraverseInOrder
    def next(nn,jj,kk):
        if len(nn) == 0:
            return (sdict(kk),jj)
        (a,xx) = nn[0]
        (tt,ii) = traverse(ff,jj,xx)
        if len(ii) > 0:
            return next(nn[1:],ii[1:],[(ff(a,ii[0]),tt)]+kk)
        return next(nn[1:],[],kk)
    return next(list(mm.items()),ll,[])
  
# factorial :: Integer -> Integer

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

# factorialFalling :: Integer -> Integer -> Integer

def factorialFalling(n,k):
    if k <= 0:
        return 1
    if k == 1:
        return n
    if k > n:
        return factorialFalling(n,n)
    return n * factorialFalling(n-1,k-1)

# factorialRising :: Integer -> Integer -> Integer

def factorialRising(n,k):
    if k <= 0:
        return 1
    if k == 1:
        return n
    return n * factorialRising(n+1,k-1)

# combination :: Integer -> Integer -> Integer

def combination(n,k):
    if n < 0 or k < 0 or k > n:
        return 0
    return factorialFalling(n,k) // factorial(k)

# combination_1 :: Integer -> Integer -> Integer

def combination_1(n,k):
    fac = factorial
    if n < 0 or k < 0 or k > n:
        return 0
    return fac(n) // fac(k) // fac(n-k)

# combinationMultinomial :: Integer -> [Integer] -> Integer

def combinationMultinomial(n,kk):
    fac = factorial
    def product(list):
        p = 1
        for i in list:
            p *= i
        return p
    return fac(n) // product([fac(k) for k in kk])

# compositionWeak :: Integer -> Integer -> Integer

def compositionWeak(z,v):
    return factorial(z+v-1) // factorial(z) // factorial(v-1)

# compositionStrong :: Integer -> Integer -> Integer

def compositionStrong(z,v):
    return combination(z-1,v-1)

# stirlingSecond :: Integer -> Integer -> Integer

def stirlingSecond(n,k):
    fac = factorial
    if n < 1 or k < 1 or k > n:
        return 0
    if n == 1 or k == 1 or k == n:
        return 1
    return sum([(-1)**(k-j) * j**(n-1) * fac(k) // fac(j-1) // fac(k-j) for j in range(1,k+1)]) // fac(k)

# bell :: Integer -> Integer

def bell(n):
    if n < 1:
        return 0
    if n == 1:
        return 1
    return sum([stirlingSecond(n,k) for k in range(1,n+1)])

# entropy :: [Rational] -> Double

def entropy(ll):
    if len(ll) == 0:
        return 0
    s = sum(ll)
    if s <= 0:
        return 0
    return -sum([r/s * log(r/s) for r in ll])


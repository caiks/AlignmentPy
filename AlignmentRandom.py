from Alignment import *
from random import shuffle, seed

# historiesShuffle :: History -> Int -> Maybe History

def historiesShuffle(hh,s):
    hvars = historiesSetVar
    llhh = listsHistory
    hhll = historiesList
    def hred(hh,v):
        return setVarsHistoriesReduce(sset([v]),hh)
    join = pairHistoriesJoin
    if len(hvars(hh)) == 0:
        return None
    seed(s)
    gg = historyEmpty()
    for v in hvars(hh):
        [ii,qq] = zip(*hhll(hred(hh,v)))
        ii1 = list(ii)
        shuffle(ii1)
        gg1 = llhh(list(zip(ii1,qq)))
        if len(gg) == 0:
            gg = gg1
        else:
            gg = join(gg,gg1)
    return gg

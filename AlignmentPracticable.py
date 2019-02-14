from Alignment import *
from AlignmentRandom import *
from AlignmentSubstrate import *
from AlignmentApprox import *

# rollValuer :: RollValue -> Histogram -> Histogram -> (Histogram, Histogram) 

def rollValuer(rv,aa,bb):
    ssgl = stateSingleton
    ssll = statesList
    llss = listsState
    sunion = pairStatesUnionLeft
    def sminus(ss,rr):
        return llss(list(sset(ssll(ss)) - sset(ssll(rr))))
    llaa = listsHistogram
    aall = histogramsList
    def unit(ss):
        return setStatesHistogramUnit(sset([ss]))
    add = pairHistogramsAdd
    sub = pairHistogramsSubtract
    mul = pairHistogramsMultiply
    rvvvvst = rollValuesSetVariableVariableValueValue
    def rollv(v,s,t,aa):
        xx = mul(aa,unit(ssgl(v,t)))
        xx = add(xx,llaa([(sunion(sminus(ss,ssgl(v,s)),ssgl(v,t)), c) for (ss,c) in aall(mul(aa,unit(ssgl(v,s))))]))
        xx = add(xx,sub(aa,mul(aa,add(unit(ssgl(v,s)),unit(ssgl(v,t))))))
        return xx
    (_,v,s,t) = rvvvvst(rv)
    if s == t:
        return (aa, bb)
    return (rollv(v,s,t,aa), rollv(v,s,t,bb))

# rollValueAlignmenter_u :: RollValue -> Map.Map Variable (Map.Map State Double) -> Double -> Histogram -> Histogram -> (Double, Histogram, Histogram) 

def rollValueAlignmenter_u(rv,yy,a,aa,bb):
    ssgl = stateSingleton
    aall = histogramsList
    mul = pairHistogramsMultiply
    def unit(ss):
        return setStatesHistogramUnit(sset([ss]))
    def facln(x):
        return gammaln(float(x) + 1)
    def sumfacln(aa):
        return sum([facln(c) for (_,c) in aall(aa)])
    rvvvvst = rollValuesSetVariableVariableValueValue
    (_,v,s,t) = rvvvvst(rv)
    if s == t:
        return (a,aa,bb)
    (aa1,bb1) = rollValuer(rv,aa,bb)
    r1 = sumfacln(mul(aa1,unit(ssgl(v,t)))) - sumfacln(mul(bb1,unit(ssgl(v,t))))
    a1 = a - yy[v][ssgl(v,s)] - yy[v][ssgl(v,t)] + r1
    return (a1,aa1,bb1)

# parametersSystemsBuilderTupleNoSumlayerMultiEffective :: 
#   Integer -> Integer -> Integer -> Integer -> System -> Set.Set Variable -> Fud -> Histogram -> Histogram ->  
#   Maybe (Set.Set ((Set.Set Variable, Histogram, Histogram),Double))

def parametersSystemsBuilderTupleNoSumlayerMultiEffective(xmax,omax,bmax,mmax,uu,vv,ff,xx,xxrr):
    def sgl(x):
        return sset([x])
    def top(amax,mm):
        return sdict([(x,y) for (y,x) in list(sset([(b,a) for (a,b) in mm.items()]))[-amax:]])
    def topd(amax,mm):
        return sset([x for (y,x) in list(sset([(b,a) for (a,b) in mm.items()]))[-amax:]])
    uvars = systemsSetVar
    vol = systemsSetVarsVolume
    aall = histogramsList
    def red(aa,vv):
        return setVarsHistogramsReduce(vv,aa)
    eff = histogramsEffective
    acard = histogramsCardinality
    vars = histogramsSetVar
    ind = histogramsIndependent
    def facln(x):
        return gammaln(float(x) + 1)
    def sumfacln(aa):
        return sum([facln(c) for (_,c) in aall(aa)])
    fder = fudsDerived
    fvars = fudsVars
    def init(vv):
        return sdict([(((sgl(w), histogramEmpty(), histogramEmpty()),0),(0,0,0)) for w in vv])
    def final(nn):
        return sdict([(((kk,aa,bb),y),a) for (((kk,aa,bb),y),a) in nn.items() if len(kk) > 1])
    def buildb(ww,qq,nn):
        pp = sset([kk|sgl(w) for (((kk,_,_),_),_) in qq.items() for w in (ww-kk)])
        mm = sdict()
        for jj in pp:
            u = vol(uu,jj)
            if u <= xmax:
                bb = red(xx,jj)
                bbrr = red(xxrr,jj)
                a1 = sumfacln(bb)
                a2 = sumfacln(ind(bb))
                b1 = sumfacln(bbrr)
                b2 = sumfacln(ind(bbrr))
                mm[((jj, bb, bbrr), a1-b1)] = (a1-a2-b1+b2, -b1+b2, -u)
        mm = top(omax,mm)
        if len(mm) > 0:
            rr = nn.copy()
            rr.update(mm)
            return buildb(ww,mm,rr) 
        return final(nn) 
    if xmax < 0 or omax < 0 or mmax < 1 or bmax < mmax:
        return None
    if not (vars(xx).issubset(uvars(uu)) and vars(xx) == vars(xxrr) and vv.issubset(vars(xx))):
        return None
    meff = sset([v for v in vv if acard(eff(red(xx,sgl(v)))) > 1])
    if len(meff) == 0:
        return sset()
    if len(ff) == 0:
        return topd(bmax//mmax,buildb(meff,init(meff),sdict()))
    if fvars(ff).issubset(vars(xx)):
        return topd(bmax//mmax,buildb(fvars(ff)|meff,init(fder(ff)),sdict()))
    return None

# parametersSystemsPartitionerMaxRollByM_1 :: 
#   Integer -> Integer -> Integer -> System -> Set.Set Variable -> Histogram -> Histogram -> Double ->
#   Maybe [([Set.Set (State,Int)], Histogram, Histogram)]

def parametersSystemsPartitionerMaxRollByM_1(mmax,umax,pmax,uu,kk,bb,bbrr,y1):
    def topd(amax,mm):
        return [x for (y,x) in list(sdict([(b,a) for (a,b) in mm]).items())[-amax:]]
    prod = listSetsProduct
    stirsll = setsSetPartitionFixed
    uvars = systemsSetVar
    vol = systemsSetVarsVolume
    cart = systemsSetVarsSetStateCartesian
    sempty = stateEmpty
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    vars = histogramsSetVar
    aall = histogramsList
    llaa = listsHistogram_u
    def unit(qq):
        return llaa([(ss,1) for ss in qq])
    ind = histogramsIndependent
    def facln(x):
        return gammaln(float(x) + 1)
    def sumfacln(aa):
        return sum([facln(c) for (_,c) in aall(aa)])
    trans = histogramsSetVarsTransform_u
    def tmul(aa,tt):
        return transformsHistogramsApply(tt,aa)
    if umax < 0 or mmax < 0 or pmax < 0:
        return None
    if not (vars(bb).issubset(uvars(uu)) and vars(bb) == vars(bbrr) and kk.issubset(vars(bb))):
        return None
    pp = []
    for m in range(2,mmax+1):
        c = vol(uu,kk) ** (1.0/m)
        mm = []
        for yy in stirsll(kk,m):
            if all([vol(uu,jj) <= umax for jj in yy]):
                nn = [sset([(ss,i) for (i,ss) in enumerate(cart(uu,jj))]) for jj in yy]
                qq = []
                for ll in prod(nn):
                    rr = sempty()
                    for (w,(ss,u)) in enumerate(ll):
                        rr = sunion(rr,sunion(ss,ssgl(VarIndex(w),ValIndex(u))))
                    qq.append(rr)
                tt = trans(unit(qq),sset([VarIndex(w) for w in range(0,m)]))
                cc = tmul(bb,tt)
                ccrr = tmul(bbrr,tt)
                a2 = sumfacln(ind(cc))
                b2 = sumfacln(ind(ccrr))
                mm.append(((nn, cc, ccrr), ((y1-a2+b2)/c, b2, -m)))
        pp.extend(topd(pmax,mm))
    return pp

# parametersSystemsPartitionerMaxRollByM :: 
#   Integer -> Integer -> Integer -> System -> Set.Set Variable -> Histogram -> Histogram -> Double ->
#   Maybe (Set.Set (Set.Set (Set.Set (State,Int)), Histogram, Histogram))

def parametersSystemsPartitionerMaxRollByM(mmax,umax,pmax,uu,kk,bb,bbrr,y1):
    def topd(amax,mm):
        return [x for (y,x) in list(sset([(b,a) for (a,b) in mm]))[-amax:]]
    prod = listSetsProduct
    stirsll = setsSetPartitionFixed
    uvars = systemsSetVar
    vol = systemsSetVarsVolume
    cart = systemsSetVarsSetStateCartesian
    sempty = stateEmpty
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    vars = histogramsSetVar
    aall = histogramsList
    llaa = listsHistogram_u
    def unit(qq):
        return llaa([(ss,1) for ss in qq])
    ind = histogramsIndependent
    def facln(x):
        return gammaln(float(x) + 1)
    def sumfacln(aa):
        return sum([facln(c) for (_,c) in aall(aa)])
    trans = histogramsSetVarsTransform_u
    def tmul(aa,tt):
        return transformsHistogramsApply(tt,aa)
    if umax < 0 or mmax < 0 or pmax < 0:
        return None
    if not (vars(bb).issubset(uvars(uu)) and vars(bb) == vars(bbrr) and kk.issubset(vars(bb))):
        return None
    pp = []
    for m in range(2,mmax+1):
        c = vol(uu,kk) ** (1.0/m)
        mm = []
        for yy in stirsll(kk,m):
            if all([vol(uu,jj) <= umax for jj in yy]):
                nn = sset([sset([(ss,i) for (i,ss) in enumerate(cart(uu,jj))]) for jj in yy])
                qq = []
                for ll in prod(nn):
                    rr = sempty()
                    for (w,(ss,u)) in enumerate(ll):
                        rr = sunion(rr,sunion(ss,ssgl(VarIndex(w),ValIndex(u))))
                    qq.append(rr)
                tt = trans(unit(qq),sset([VarIndex(w) for w in range(0,m)]))
                cc = tmul(bb,tt)
                ccrr = tmul(bbrr,tt)
                a2 = sumfacln(ind(cc))
                b2 = sumfacln(ind(ccrr))
                mm.append(((nn, cc, ccrr), ((y1-a2+b2)/c, b2, -m)))
        pp.extend(topd(pmax,mm))
    return sset(pp)

# parametersRoller :: 
#   Integer -> Set.Set (Set.Set (Set.Set (State,Int)), Histogram, Histogram) -> Maybe (Set.Set (Set.Set (Set.Set (State,Int))))

def parametersRoller(pmax,qq):
    def sgl(x):
        return sset([x])
    def dom(qq):
        return sset([a for (a,b) in qq])
    def ran(qq):
        return sset([b for (a,b) in qq])
    def join(xx,yy):
        zz = dom(yy)
        return sset([(s1,t2) for (s1,t1) in xx for (s2,t2) in yy if s2 == t1] + 
                    [(s1,t1) for (s1,t1) in xx if t1 not in zz])
    def top(amax,mm):
        return sdict([(x,y) for (y,x) in list(sset([(b,a) for (a,b) in mm.items()]))[-amax:]])
    def topd(amax,mm):
        return sset([x for (y,x) in list(sset([(b,a) for (a,b) in mm.items()]))[-amax:]])
    ssgl = stateSingleton
    vvvstrv = setVariablesVariablesValuesValuesRollValue
    aall = histogramsList
    mul = pairHistogramsMultiply
    ind = histogramsIndependent
    llaa = listsHistogram_u
    def unit(qq):
        return llaa([(ss,1) for ss in qq])
    algn = histogramsAlignment
    def facln(x):
        return gammaln(float(x) + 1)
    def sumfacln(aa):
        return sum([facln(c) for (_,c) in aall(aa)])
    def algner(rv,yy,x):
        (a,aa,aaxx) = x
        return rollValueAlignmenter_u(rv,yy,a,aa,aaxx)
    def rals(nn,aa,aaxx):
        xx = sdict()
        for (i,ii) in enumerate(nn):
            v = VarIndex(i)
            yy = sdict()
            for (_,j) in ii:
                u = ValIndex(j)
                ss = ssgl(v,u)
                yy[ss] = sumfacln(mul(aa,unit(sgl(ss)))) - sumfacln(mul(aaxx,unit(sgl(ss))))
            xx[v] = yy
        return xx
    def rollb(qq,pp):
        mm = sdict()
        for ((nn,rraa,rrbb),_) in qq.items():
            vv = sset([VarIndex(i) for i in range(0,len(nn))])
            (_,aa,aaxx) = rraa
            (_,bb,bbxx) = rrbb
            yyaa = rals(nn,aa,aaxx)
            yybb = rals(nn,bb,bbxx)
            for (v,ii) in enumerate(nn):
                if len(ran(ii)) > 2:
                    for s in ran(ii):
                        for t in ran(ii): 
                            if s > t:
                                nn1 = sset(list(nn)[:v] + [join(ii,sgl((s,t)))] + list(nn)[v+1:])
                                rv = vvvstrv((vv, VarIndex(v), ValIndex(s), ValIndex(t)))
                                rraa1 = algner(rv,yyaa,rraa)
                                rrbb1 = algner(rv,yybb,rrbb)
                                (a1,_,_) = rraa1
                                (b1,_,_) = rrbb1
                                w = 1
                                for ii1 in nn1:
                                    w = w * len(ran(ii1))
                                m = len(vv)
                                c1 = w ** (1.0/m)
                                mm[(nn1,rraa1,rrbb1)] = (a1-b1)/c1
        mm = top(pmax,mm)
        if len(mm) > 0:
            rr = pp.copy()
            rr.update(mm)
            return rollb(mm,rr)
        return pp
    if pmax < 0:
        return None
    mm = sdict()
    for (nn,aa,bb) in qq:
        a = algn(aa)
        b = algn(bb)
        w = 1
        for ii in nn:
            w = w * len(ran(ii))
        m = len(nn)
        c = w ** (1.0/m)
        rraa = (a, aa, ind(aa))
        rrbb = (b, bb, ind(bb))
        mm[(nn,rraa,rrbb)] = (a-b)/c
    return sset([nn1 for (nn1,_,_) in topd(pmax,rollb(mm,mm))])

# parametersRollerExcludedSelf :: 
#   Integer -> Set.Set (Set.Set (Set.Set (State,Int)), Histogram, Histogram) -> Maybe (Set.Set (Set.Set (Set.Set (State,Int))))

def parametersRollerExcludedSelf(pmax,qq):
    def sgl(x):
        return sset([x])
    def dom(qq):
        return sset([a for (a,b) in qq])
    def ran(qq):
        return sset([b for (a,b) in qq])
    def join(xx,yy):
        zz = dom(yy)
        return sset([(s1,t2) for (s1,t1) in xx for (s2,t2) in yy if s2 == t1] + 
                    [(s1,t1) for (s1,t1) in xx if t1 not in zz])
    def top(amax,mm):
        return sdict([(x,y) for (y,x) in list(sset([(b,a) for (a,b) in mm.items()]))[-amax:]])
    def topd(amax,mm):
        return sset([x for (y,x) in list(sset([(b,a) for (a,b) in mm.items()]))[-amax:]])
    ssgl = stateSingleton
    vvvstrv = setVariablesVariablesValuesValuesRollValue
    aall = histogramsList
    mul = pairHistogramsMultiply
    ind = histogramsIndependent
    llaa = listsHistogram_u
    def unit(qq):
        return llaa([(ss,1) for ss in qq])
    algn = histogramsAlignment
    def facln(x):
        return gammaln(float(x) + 1)
    def sumfacln(aa):
        return sum([facln(c) for (_,c) in aall(aa)])
    def algner(rv,yy,x):
        (a,aa,aaxx) = x
        return rollValueAlignmenter_u(rv,yy,a,aa,aaxx)
    def rals(nn,aa,aaxx):
        xx = sdict()
        for (i,ii) in enumerate(nn):
            v = VarIndex(i)
            yy = sdict()
            for (_,j) in ii:
                u = ValIndex(j)
                ss = ssgl(v,u)
                yy[ss] = sumfacln(mul(aa,unit(sgl(ss)))) - sumfacln(mul(aaxx,unit(sgl(ss))))
            xx[v] = yy
        return xx
    def rollb(qq,pp):
        mm = sdict()
        for ((nn,rraa,rrbb),_) in qq.items():
            vv = sset([VarIndex(i) for i in range(0,len(nn))])
            (_,aa,aaxx) = rraa
            (_,bb,bbxx) = rrbb
            yyaa = rals(nn,aa,aaxx)
            yybb = rals(nn,bb,bbxx)
            for (v,ii) in enumerate(nn):
                if len(ran(ii)) > 2:
                    for s in ran(ii):
                        for t in ran(ii): 
                            if s > t:
                                nn1 = sset(list(nn)[:v] + [join(ii,sgl((s,t)))] + list(nn)[v+1:])
                                rv = vvvstrv((vv, VarIndex(v), ValIndex(s), ValIndex(t)))
                                rraa1 = algner(rv,yyaa,rraa)
                                rrbb1 = algner(rv,yybb,rrbb)
                                (a1,_,_) = rraa1
                                (b1,_,_) = rrbb1
                                w = 1
                                for ii1 in nn1:
                                    w = w * len(ran(ii1))
                                m = len(vv)
                                c1 = w ** (1.0/m)
                                mm[(nn1,rraa1,rrbb1)] = (a1-b1)/c1
        mm = top(pmax,mm)
        if len(mm) > 0:
            rr = pp.copy()
            rr.update(mm)
            return rollb(mm,rr)
        return pp
    if pmax < 0:
        return None
    mm = sdict()
    for (nn,aa,bb) in qq:
        a = algn(aa)
        b = algn(bb)
        w = 1
        for ii in nn:
            w = w * len(ran(ii))
        m = len(nn)
        c = w ** (1.0/m)
        rraa = (a, aa, ind(aa))
        rrbb = (b, bb, ind(bb))
        mm[(nn,rraa,rrbb)] = (a-b)/c
    return sset([nn1 for (nn1,_,_) in topd(pmax,rollb(mm,sdict()))])

# parametersSystemsBuilderDerivedVarsHighestNoSumlayer :: 
#   Integer -> Integer -> System -> Set.Set Variable -> Fud -> Histogram -> Histogram ->  
#   Maybe (Map.Map (Set.Set Variable, Histogram, Histogram) Double)

def parametersSystemsBuilderDerivedVarsHighestNoSumlayer(wmax,omax,uu,vv,ff,xx,xxrr):
    def sgl(x):
        return sset([x])
    def top(amax,mm):
        return sdict([(x,y) for (y,x) in list(sset([(b,a) for (a,b) in mm.items()]))[-amax:]])
    def maxfst(mm):
        return sdict([(x,a) for ((a,_,_),x) in list(sset([(b,a) for (a,b) in mm.items()]))[-1:]])
    uvars = systemsSetVar
    vol = systemsSetVarsVolume
    aall = histogramsList
    def red(aa,vv):
        return setVarsHistogramsReduce(vv,aa)
    eff = histogramsEffective
    acard = histogramsCardinality
    vars = histogramsSetVar
    ind = histogramsIndependent
    algn = histogramsAlignment
    fder = fudsDerived
    fvars = fudsSetVar
    depends = fudsSetVarsDepends
    def init(vv):
        return sdict([((sgl(w), histogramEmpty(), histogramEmpty()),(0,0,0)) for w in vv])
    def final(nn):
        return sdict([((kk,aa,bb),a) for ((kk,aa,bb),a) in nn.items() if len(kk) > 1])
    def buildd(ww,qq,nn):
        pp = sset([kk|sgl(w) for ((kk,_,_),_) in qq.items() for w in (ww-kk)])
        mm = sdict()
        for jj in pp:
            u = vol(uu,jj)
            if u <= wmax and fder(depends(ff,jj)) == jj:
                bb = red(xx,jj)
                bbrr = red(xxrr,jj)
                m = len(jj)
                a = algn(bb)
                b = algn(bbrr)
                c = u ** (1.0/m)
                mm[(jj, bb, bbrr)] = ((a-b)/c,-b/c,-u)
        mm = top(omax,mm)
        if len(mm) > 0:
            rr = nn.copy()
            rr.update(mm)
            return buildd(ww,mm,rr) 
        return final(nn) 
    if wmax < 0 or omax < 0:
        return None
    if not (vars(xx).issubset(uvars(uu)) and vars(xx) == vars(xxrr) and vv.issubset(vars(xx))):
        return None
    if not fvars(ff).issubset(uvars(uu)):
        return None
    return maxfst(buildd(fvars(ff)-vv,init(fder(ff)),sdict()))

# parametersSystemsLayererMaxRollByMExcludedSelfHighest :: 
#   Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
#   System -> Set.Set Variable -> Histogram -> Histogram -> Integer ->
#   Maybe (System, Fud, Map.Map (Set.Set Variable) Double)

def parametersSystemsLayererMaxRollByMExcludedSelfHighest(wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,uu,vv,aa,aarr,f):
    def sgl(x):
        return sset([x])
    def domcd(qq):
        return len(sset([a for (a,b) in qq]))
    def rancd(qq):
        return len(sset([b for (a,b) in qq]))
    def maxr(mm):
        if len(mm) > 0:
            return list(sset([b for (_,b) in mm.items()]))[-1:][0]
        return 0
    uvars = systemsSetVar
    lluu = listsSystem_u
    uunion = pairSystemsUnion
    sunion = pairStatesUnionLeft
    ssgl = stateSingleton
    llaa = listsHistogram_u
    vars = histogramsSetVar
    def unit(qq):
        return llaa([(ss,1) for ss in qq])
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    trans = histogramsSetVarsTransform_u
    ttpp = transformsPartition
    und = transformsUnderlying
    qqff = setTransformsFud_u
    ffqq = fudsSetTransform
    fvars = fudsSetVar
    fhis = fudsSetHistogram
    def funion(ff,gg):
        return qqff(ffqq(ff) | ffqq(gg))
    def buildfftup(uu,vv,ff,xx,xxrr):
        return parametersSystemsBuilderTupleNoSumlayerMultiEffective(xmax,omax,bmax,mmax,uu,vv,ff,xx,xxrr)
    def parter(uu,kk,bb,bbrr,y1):
        return parametersSystemsPartitionerMaxRollByM(mmax,umax,pmax,uu,kk,bb,bbrr,y1)
    def roller(qq):
        return parametersRoller(1,qq)
    def buildffdervar(uu,vv,ff,xx,xxrr):
        return sdict([(kk,a) for ((kk,_,_),a) in parametersSystemsBuilderDerivedVarsHighestNoSumlayer(wmax,omax,uu,vv,ff,xx,xxrr).items()])
    def layer(vv,uu,ff,mm,xx,xxrr,f,l):
        jj = []
        for ((kk,bb,bbrr),y1) in buildfftup(uu,vv,ff,xx,xxrr):
            for pp in parter(uu,kk,bb,bbrr,y1):
                for nn in roller(sgl(pp)):
                    for ii in nn:
                        if rancd(ii) < domcd(ii):
                            jj.append(ii)
        ll = []
        for (b,ii) in enumerate(jj):
            w = VarPair((VarPair((VarInt(f),VarInt(l))),VarInt(b+1)))
            ww = sset([ValInt(u) for (_,u) in ii])
            tt = trans(unit([sunion(ss,ssgl(w,ValInt(u))) for (ss,u) in ii]),sgl(w))
            ll.append((tt,(w,ww)))
        ll1 = []
        for (tt,(w,ww)) in ll:
            if all([len(ww) != len(ww1) or und(tt) != und(tt1) or ttpp(tt) != ttpp(tt1) for (tt1,(w1,ww1)) in ll if w > w1]):
                ll1.append((tt,(w,ww)))
        hh = qqff(sset([tt for (tt,_) in ll1]))
        uu1 = uunion(uu,lluu([(w,ww) for (_,(w,ww)) in ll1]))
        xx1 = apply(vars(xx),vars(xx)|fvars(hh),fhis(hh),xx)
        xxrr1 = apply(vars(xx),vars(xx)|fvars(hh),fhis(hh),xxrr)
        gg = funion(ff,hh)
        mm1 = buildffdervar(uu1,vv,gg,xx1,xxrr1)
        if l <= lmax and len(hh) > 0 and (len(mm) == 0 or maxr(mm1) > maxr(mm)):
            return layer(vv,uu1,gg,mm1,xx1,xxrr1,f,l+1)
        return (uu,ff,mm) 
    if wmax < 0 or lmax < 0 or xmax < 0 or omax < 0 or bmax < 0 or mmax < 1 or umax < 0 or pmax < 0:
        return None
    if not (vars(aa).issubset(uvars(uu)) and vars(aa) == vars(aarr) and vv.issubset(vars(aa))):
        return None
    return layer(vv,uu,fudEmpty(),sdict(),aa,aarr,f,1)

# parametersSystemsDecomperMaxRollByMExcludedSelfHighest :: 
#   Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> 
#   Integer -> Integer ->
#   System -> Set.Set Variable -> Histogram -> 
#   Maybe (System, DecompFud)

def parametersSystemsDecomperMaxRollByMExcludedSelfHighest(wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,mult,seed,uu,vv,aa):
    dom = relationsDomain
    def maxd(mm):
        if len(mm) > 0:
            return list(sset([(b,a) for (a,b) in mm.items()]))[-1]
        return (0,sset())
    def tsgl(r):
        return sdict([(r,sdict())])
    uvars = systemsSetVar
    cart = systemsSetVarsSetStateCartesian_u
    hshuffle = historiesShuffle
    isint = histogramsIsIntegral
    vars = histogramsSetVar
    size = histogramsSize
    resize = histogramsResize
    aadd = pairHistogramsAdd
    def unit(ss):
        return setStatesHistogramUnit(sset([ss]))
    apply = setVarsSetVarsSetHistogramsHistogramsApply
    aahh = histogramsHistory
    hhaa = historiesHistogram
    def ashuffle(aa,seed,mult):
        hh = aahh(aa)
        bb = histogramEmpty()
        for r in range(0,mult):
            bb = aadd(bb,hhaa(hshuffle(hh,seed+r)))
        return resize(size(aa),bb)
    ffqq = fudsSetHistogram
    fder = fudsDerived
    depends = fudsSetVarsDepends
    zzdf = treePairStateFudsDecompFud
    def zztrim(df):
        pp = []
        for ll in treesPaths(df):
            (_,ff) = ll[-1]
            if len(ff) == 0:
                pp.append(ll[:-1])
            else:
                pp.append(ll)
        return pathsTree(pp)
    def layerer(uu,aa,aarr,f):
        return parametersSystemsLayererMaxRollByMExcludedSelfHighest(wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,uu,vv,aa,aarr,f)
    def decomp(uu,zz,f,s):
        if len(zz) == 0:
            aarr = ashuffle(aa,s,mult)
            (uur,ffr,nnr) = layerer(uu,aa,aarr,f)
            (ar,kkr) = maxd(nnr)
            if len(ffr) == 0 or len(nnr) == 0 or ar <= 0:
                return (uu, decompFudEmpty())
            ffr1 = depends(ffr,kkr)
            zzr = tsgl((stateEmpty(),ffr1))
            return decomp(uur,zzr,f+1,s+mult)
        mm = []
        for (nn,yy) in treesPlaces(zz):
            rrc = sset([unit(ss) for (ss,_) in nn])
            hhc = sset([bb for (_,gg) in nn for bb in ffqq(gg)])
            (_,ff) = nn[-1]
            if len(ff) > 0:
                for ss in cart(uu,fder(ff)) - dom(treesRoots(yy)):
                    xx = hhc | rrc | sset([unit(ss)])
                    bb = apply(vv,vv,xx,aa)
                    if size(bb) > 0:
                        mm.append((size(bb),nn,ss,bb))
        if len(mm) == 0:
            return (uu,zzdf(zztrim(zz)))
        mm.sort(key = lambda x: x[0])
        (_,nn,ss,bb) = mm[-1]
        bbrr = ashuffle(bb,s,mult)
        (uuc,ffc,nnc) = layerer(uu,bb,bbrr,f)
        (ac,kkc) = maxd(nnc)
        ffc1 = fudEmpty()
        if ac > 0:
             ffc1 = depends(ffc,kkc)
        zzc = pathsTree(treesPaths(zz) + [nn+[(ss,ffc1)]])
        return decomp(uuc,zzc,f+1,s+mult)
    if wmax < 0 or lmax < 0 or xmax < 0 or omax < 0 or bmax < 0 or mmax < 1 or umax < 0 or pmax < 0:
        return None
    if (not isint(aa)) or mult < 1:
        return None
    if not (vars(aa).issubset(uvars(uu)) and vv.issubset(vars(aa))):
        return None
    return decomp(uu,emptyTree(),1,seed)

# systemsDecompFudsNullablePracticable :: System -> DecompFud -> Integer -> Maybe Fud

def systemsDecompFudsNullablePracticable(uu,df,g):
    def sgl(x):
        return sset([x])
    rel = treesRelation
    llss = listsState
    ssll = statesList
    cart = systemsSetVarsSetStateCartesian_u
    unit = setStatesHistogramUnit
    add = pairHistogramsAdd
    mul = pairHistogramsMultiply
    trans = histogramsSetVarsTransform_u
    def ttff(tt):
        return setTransformsFud(sset([tt]))
    qqff = setTransformsFud_u
    ffqq = fudsSetTransform
    fder = fudsDerived
    def funion(ff,gg):
        return qqff(ffqq(ff) | ffqq(gg))
    dfzz = decompFudsTreePairStateFud
    dfund = decompFudsUnderlying
    def dfvars(df):
        return fudsVars(decompFudsFud(df))
    uo = ValStr("out")
    ui = ValStr("in")
    un = ValStr("null")
    gstr = ""
    if g > 1:
        gstr = str(g)
    gs = VarStr("s" + gstr)
    gc = VarStr("c" + gstr)
    gn = VarStr("n" + gstr)
    def okvar(x):
        if isinstance(x, VarPair):
            (w,_) = x._rep
            if isinstance(w, VarPair):
                (_,v) = w._rep
                if v != gs and v != gc and v != gn:
                    return True
        return False
    def cont(ffp,rrp,ffc,i):
        if len(ffp) == 0:
            return fudEmpty()
        f = ssll(rrp)[0][0]._rep[0]._rep[0]
        ws = VarPair((VarPair((f,gs)),VarInt(i)))
        aaso = unit(sgl(llss([(ws,uo)])))
        aasi = unit(sgl(llss([(ws,ui)])))
        tts = trans(add(mul(unit(cart(uu,fder(ffp))-sgl(rrp)),aaso),mul(unit(sgl(rrp)),aasi)),sgl(ws))
        if len(ffc) == 0:
            return funion(ffp,ttff(tts))
        vc = list(fder(ffc))[0]
        wc = VarPair((VarPair((f,gc)),VarInt(i)))
        aac = unit(sset([llss([(ws,uo),(vc,uo),(wc,uo)]),llss([(ws,uo),(vc,ui),(wc,uo)]),
                         llss([(ws,ui),(vc,uo),(wc,uo)]),llss([(ws,ui),(vc,ui),(wc,ui)])]))
        ttc = trans(aac,sgl(wc))
        return funion(funion(funion(ffc,ffp),ttff(tts)),ttff(ttc))
    def nullable(ff,ggc):
        def wwttr(w,i):
            f = w._rep[0]._rep[0]
            w1 = VarPair((VarPair((f,gn)),VarInt(i)))
            aa = unit(sset([llss([(w,u),(w1,u)]) for ss in cart(uu,sgl(w)) for (_,u) in ssll(ss)]))
            return trans(aa,sgl(w1)) 
        def wwttc(w,i):
            f = w._rep[0]._rep[0]
            w1 = VarPair((VarPair((f,gn)),VarInt(i)))
            vc = list(fder(ggc))[0]
            aao = unit(sset([llss([(vc,uo),(w,u),(w1,un)]) for ss in cart(uu,sgl(w)) for (_,u) in ssll(ss)]))
            aai = unit(sset([llss([(vc,ui),(w,u),(w1,u)]) for ss in cart(uu,sgl(w)) for (_,u) in ssll(ss)]))
            return trans(add(aao,aai),sgl(w1)) 
        if len(ggc) == 0:
            return funion(ff,qqff(sset([wwttr(w,i+1) for (i,w) in enumerate(fder(ff))])))
        return funion(funion(ggc,ff),qqff(sset([wwttc(w,i+1) for (i,w) in enumerate(fder(ff))])))
    def trff(ffp,ffc,zz):
        gg = fudEmpty()
        for (i,((rrp,ff),xx)) in enumerate(rel(zz)):
            ggc = cont(ffp,rrp,ffc,i+1)
            ffn = nullable(ff,ggc)
            gg = funion(gg,funion(funion(ggc,ffn),trff(ff,ggc,xx)))
        return gg
    for v in dfvars(df) - dfund(df):
        if not okvar(v):
            return None
    return trff(fudEmpty(),fudEmpty(),dfzz(df))

# variablesVariableFud :: Variable -> Variable

def variablesVariableFud(x):
    if isinstance(x, VarPair):
        (w,_) = x._rep
        if isinstance(w, VarPair):
            (f,_) = w._rep
            return f
    return VarInt(0)

# parametersBuilderConditionalVars :: 
#   Integer -> Integer -> Integer -> Set.Set Variable -> Histogram -> 
#   Maybe (Map.Map (Set.Set Variable) Double)

def parametersBuilderConditionalVars(kmax,omax,qmax,ll,aa):
    def sgl(x):
        return sset([x])
    def bot(amax,mm):
        return sdict([(x,y) for (y,x) in list(sset([(b,a) for (a,b) in mm.items()]))[:amax]])
    vars = histogramsSetVar
    def red(aa,vv):
        return setVarsHistogramsReduce(vv,aa)
    ent = histogramsEntropy 
    def buildc(qq,nn):
        pp = sset([kk|sgl(w) for (kk,e) in qq.items() if e > 0 for w in vvk-kk])
        mm = bot(omax,sdict([(jj,ent(red(aa,ll|jj))-ent(red(aa,jj))) for jj in pp if len(jj) <= kmax]))
        if len(mm) > 0:
            nn1 = nn.copy()
            nn1.update(mm)
            return buildc(mm,nn1)
        return nn
    if kmax < 0 or omax < 0 or qmax < 0:
        return None
    vvk = vars(aa) - ll
    rr = bot(omax,sdict([(sgl(w),ent(red(aa,ll|sgl(w)))-ent(red(aa,sgl(w)))) for w in vvk]))
    return bot(qmax,buildc(rr,rr))


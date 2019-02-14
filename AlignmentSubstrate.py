from Alignment import *

# systemsSetVarsSetRollValueSubstrate :: System -> Set.Set Variable -> Maybe (Set.Set RollValue)

def systemsSetVarsSetRollValueSubstrate(uu,vv):
    vvvstvr = setVariablesVariablesValuesValuesRollValue
    uvalsll = systemsVarsSetValue
    uvars = systemsVars
    if vv.issubset(uvars(uu)):
        return sset([vvvstvr((vv,v,s,t)) for v in vv for s in uvalsll(uu,v) for t in uvalsll(uu,v)])
    return None

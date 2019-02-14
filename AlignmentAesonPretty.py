from AlignmentAeson import *
import json
import itertools

# historyPersistentsEncode :: HistoryPersistent -> ByteString

def historyPersistentsEncode(hh):
    return ("{\n\t\"hsystem\":[\n" + 
        ",\n".join(["\t\t" + json.dumps(vv) for vv in hh["hsystem"]]) + 
        "\n\t],\n\t\"hstates\":[\n" + 
        ",\n".join(["\t\t" + json.dumps(ss) for ss in hh["hstates"]]) +
        "\n\t]\n}")

# historyPersistentsEncodePrefixed :: Int -> HistoryPersistent -> ByteString

def historyPersistentsEncodePrefixed(i,hh):
    p = ''.join(itertools.repeat('\t',i))
    return ("{\n" + p + "\t\"hsystem\":[\n" + 
        ",\n".join([p + "\t\t" + json.dumps(vv) for vv in hh["hsystem"]]) + 
        "\n" + p + "\t],\n" + p + "\t\"hstates\":[\n" + 
        ",\n".join([p + "\t\t" + json.dumps(ss) for ss in hh["hstates"]]) +
        "\n" + p + "\t]\n" + p + "}")

# transformPersistentsEncodePrefixed :: Int -> TransformPersistent -> ByteString

def transformPersistentsEncodePrefixed(i,tt):
    p = ''.join(itertools.repeat('\t',i))
    return ("{\n" + p + "\t\"derived\":[" + 
        ",".join([json.dumps(ww) for ww in tt["derived"]]) + 
        "],\n" + p + "\t\"history\":" + 
        historyPersistentsEncodePrefixed(i+1,tt["history"]) +
        "\n" + p + "}")

# fudPersistentsEncodePrefixed :: Int -> FudPersistent -> ByteString

def fudPersistentsEncodePrefixed(i,ff):
    p = ''.join(itertools.repeat('\t',i))
    return ("[\n" + 
        ",\n".join([p + "\t" + transformPersistentsEncodePrefixed(i+1,tt) for tt in ff]) + 
        "\n" + p + "]")

# fudPersistentsEncode :: FudPersistent -> ByteString

def fudPersistentsEncode(ff):
    return fudPersistentsEncodePrefixed(0,ff)

# decompFudsPersistentsEncode :: DecompFudPersistent -> ByteString

def decompFudsPersistentsEncode(df):
     return ("{\n\t\"paths\":[\n" + 
        ",\n".join(["\t\t" + json.dumps(ss) for ss in df["paths"]]) +
        "\n\t],\n\t\"nodes\":[\n" +
        ",\n".join(["\t\t[\n\t\t\t" + historyPersistentsEncodePrefixed(3,hh) + ",\n\t\t\t" + fudPersistentsEncodePrefixed(3,ff) + "\n\t\t]" for (hh,ff) in df["nodes"]]) + 
        "\n\t]\n}")

import json
import itertools 
def ndlqr_ReadLQRProblemJSONFile(filename):
    f = open(filename)
    data = json.load(f)
    nhorizon = data['nhorizon']
    x0 = data['x0']
    lqrdata = data['lqrdata']
    nstates = lqrdata[0]["nstates"]
    ninputs = lqrdata[0]["ninputs"]
    f.close()
    lqrprob = LQRProblem(nstates, ninputs, nhorizon)
    lqrprob.ndlqr_InitializeLQRProblem(x0, lqrdata)
    return lqrprob

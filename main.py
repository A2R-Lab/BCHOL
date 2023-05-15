#Read LQRProblem from json file
filename = 'lqr_prob.json' #LQRPROBFILE
lqrprob = ndlqr_ReadLQRProblemJSONFile(filename)
nstates = lqrprob.lqrdata[0].nstates
ninputs = lqrprob.lqrdata[0].ninputs
nhorizon = lqrprob.nhorizon

#Initialize solver and solve
solver = NdLqrSolver(nstates, ninputs, nhorizon)
if solver.ndlqr_InitializeWithLQRProblem(lqrprob) == 0:
    ndlqr_Solve(solver)
    print(ndlqr_GetSolution(solver))

#Print solve summary
solver.ndlqr_PrintSolveSummary()

from pymoo.core.problem import ElementwiseProblem
import numpy as np
import matplotlib.pyplot as plt

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([1,2.7815]),
                         xu=np.array([4,3.2315]))

    def _evaluate(self, x, out, *args, **kwargs):
        from INDTIMEwrtSS_SurrogateModel import Ia, Ib
        from GRWRATEwrtSS_SurrogateModel import Ga, Gb
        from GRWRATEwrtT_SurrogateModel import Gc, Gd
        from NUCRATEwrtSS_SurrogateModel import Na, Nb
        from NUCRATEwrtT_SurrogateModel import Nc, Nd

        #x[0] is SS and x[1] is Temperature
        from INDTIMEwrtSS_SurrogateModel import IndTime_Target_s
        y_SIM_IndwrtSS = Ia * np.exp(-Ib * x[0])
        y_SIM_Diff_IndwrtSS = abs(np.log(IndTime_Target_s) - np.log(y_SIM_IndwrtSS))

        from GRWRATEwrtSS_SurrogateModel import GrwRate_Target_ums
        y_SIM_GrwRatewrtSS = Ga * x[0] + Gb
        y_SIM_Diff_GrwRatewrtSS = abs(GrwRate_Target_ums - y_SIM_GrwRatewrtSS)

        from GRWRATEwrtT_SurrogateModel import GrwRate_Target_ums
        y_SIM_GrwRatewrtT = Gc * x[1] + Gd
        y_SIM_Diff_GrwRatewrtT = abs(GrwRate_Target_ums - y_SIM_GrwRatewrtT)

        from NUCRATEwrtSS_SurrogateModel import NucRate_Target_cps
        y_SIM_NucRatewrtSS = Na * np.exp(Nb * x[0])
        y_SIM_Diff_NucRatewrtSS = abs(np.log(NucRate_Target_cps) - np.log(y_SIM_NucRatewrtSS))

        from NUCRATEwrtT_SurrogateModel import NucRate_Target_cps
        y_SIM_NucRatewrtT = Nc * x[1] + Nd
        y_SIM_Diff_NucRatewrtT = abs(NucRate_Target_cps - y_SIM_NucRatewrtT)

        '''IndTime = Ia * x[0] ** 2 + Ib * x[0] + Ic
        GrwRate = Ga * x[0] ** 2 + Gb * x[0] + Gc + Gd * x[1] ** 2 + Ge * x[1] + Gf
        NucRate = Na * x[0] ** 2 + Nb * x[0] + Nc + Nd * x[1] ** 2 + Ne * x[1] + Nf'''

        IndTime = y_SIM_Diff_IndwrtSS
        GrwRate = y_SIM_Diff_GrwRatewrtSS + y_SIM_Diff_GrwRatewrtT
        NucRate = y_SIM_Diff_NucRatewrtSS + y_SIM_Diff_NucRatewrtT
        f = IndTime + 10*GrwRate + NucRate #Growth rate has 10x factor due to being very small numbers/ magnitude out of normalisation compared to the others
        out["F"] = [f]

problem = MyProblem()

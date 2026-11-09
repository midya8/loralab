import copy
import math
import random
import unittest
from decimal import Decimal, localcontext

from loralab.coordinates import audit, inverse, matmul, matrix, maxdiff, objective, validate_problem
from loralab.eval import build_fixture
from loralab.lora import LoRAClassifier


def problem(seed=4, m=3, d=5, r=2):
    rng = random.Random(seed)
    values = lambda n, k: [[rng.uniform(-.7,.7) for _ in range(k)] for _ in range(n)]
    return {"W": values(m,d), "bias": values(1,m)[0], "A": values(r,d), "B": values(m,r),
            "examples": [{"x": values(1,d)[0], "y": i%m} for i in range(7)]}


def independent_loss(p):
    losses = []
    for e in p["examples"]:
        x,y = e["x"],e["y"]
        h = [sum(a*t for a,t in zip(row,x)) for row in p["A"]]
        z = [sum(w*t for w,t in zip(row,x)) + bias + sum(v*t for v,t in zip(b,h))
             for row,bias,b in zip(p["W"],p["bias"],p["B"])]
        top = max(z)
        losses.append(top-z[y]+math.log(sum(math.exp(v-top) for v in z)))
    return sum(losses)/len(losses)


class TestCoordinates(unittest.TestCase):
    def test_independent_factor_finite_differences(self):
        for seed in range(5):
            p=problem(seed)
            report=audit(p)
            for key,gradient in [("A",report["original"]["gradient_A"]),("B",report["original"]["gradient_B"])]:
                for i,row in enumerate(p[key]):
                    for j,v in enumerate(row):
                        q=copy.deepcopy(p);q[key][i][j]=v+1e-5
                        t=copy.deepcopy(p);t[key][i][j]=v-1e-5
                        derivative=(independent_loss(q)-independent_loss(t))/2e-5
                        self.assertAlmostEqual(gradient[i][j],derivative,delta=3e-10)

    def test_random_gauges_finite_step_and_compensation(self):
        for seed in range(40):
            p=problem(seed)
            for c in [.125,.5,1,4,-2]:
                r=audit(p,scalar_gauge=c,learning_rate=.2)
                self.assertLess(r["same_model_delta_max_abs_residual"],1e-14)
                self.assertLess(r["same_model_logits_max_abs_residual"],1e-14)
                self.assertLess(r["compensated_next_delta_max_abs_residual"],1e-14)
                for section in ["original","transformed","scalar_compensated"]:
                    self.assertLess(r[section]["update_identity_max_abs_residual"],1e-14)
                if c not in [-1,1]:
                    self.assertGreater(r["next_delta_max_abs_difference"],1e-5)

    def test_shear_preserves_model_but_not_ordinary_step(self):
        r=audit(problem(),gauge=[[1,2],[0,1]])
        self.assertLess(r["same_model_logits_max_abs_residual"],1e-14)
        self.assertGreater(r["next_delta_max_abs_difference"],.01)
        self.assertLess(r["transformed"]["update_identity_max_abs_residual"],1e-14)

    def test_orthogonal_rotations_preserve_plain_sgd(self):
        theta=.73
        q=[[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]]
        for c in [q,[[0,1],[1,0]],[[-1,0],[0,1]]]:
            r=audit(problem(),gauge=c)
            self.assertLess(r["same_model_delta_max_abs_residual"],1e-14)
            self.assertLess(r["next_delta_max_abs_difference"],1e-14)

    def test_zero_B_and_zero_learning_rate(self):
        p=problem();p["B"]=[[0.,0.] for _ in range(3)]
        r=audit(p,scalar_gauge=4)
        self.assertEqual(r["original"]["gradient_A_norm"],0)
        self.assertEqual(r["original"]["second_order_norm"],0)
        self.assertGreater(r["next_delta_max_abs_difference"],0)
        r=audit(p,scalar_gauge=4,learning_rate=0)
        self.assertEqual(r["next_delta_max_abs_difference"],0)

    def test_decimal_finite_step_identity(self):
        with localcontext() as ctx:
            ctx.prec=80
            for seed in range(20):
                p=problem(seed)
                out=audit(p,learning_rate=.25)["original"]
                D=lambda x: Decimal.from_float(float(x))
                A=[[D(x) for x in row] for row in p["A"]]
                B=[[D(x) for x in row] for row in p["B"]]
                G=[[D(x) for x in row] for row in out["before"]["gradient_delta"]]
                ga=[[sum(B[j][i]*G[j][k] for j in range(3)) for k in range(5)] for i in range(2)]
                gb=[[sum(G[j][k]*A[i][k] for k in range(5)) for i in range(2)] for j in range(3)]
                for j in range(3):
                    for k in range(5):
                        delta=sum(B[j][i]*A[i][k] for i in range(2))
                        nextd=sum((B[j][i]-D(.25)*gb[j][i])*(A[i][k]-D(.25)*ga[i][k]) for i in range(2))
                        first=-D(.25)*sum(B[j][i]*ga[i][k]+gb[j][i]*A[i][k] for i in range(2))
                        second=D(.25)**2*sum(gb[j][i]*ga[i][k] for i in range(2))
                        self.assertLess(abs(nextd-delta-first-second),Decimal('1e-77'))
                        self.assertAlmostEqual(out["actual_delta_change"][j][k],float(nextd-delta),delta=5e-16)

    def test_shape_parameter_bounds_not_measured_rank(self):
        r=audit(problem(r=8))
        self.assertEqual(r["factor_width"],8)
        self.assertEqual(r["matrix_rank_upper_bound"],3)
        self.assertEqual(r["factor_parameters"],64)
        self.assertEqual(r["full_weight_parameters"],15)
        self.assertEqual(r["frozen_bias_parameters"],3)

    def test_rejects_invalid_problem(self):
        for key,value in [("W",[]),("A",[[1]]),("B",[[1,2]]),("bias",[0]),("examples",[])]:
            p=problem();p[key]=value
            with self.assertRaises(ValueError):audit(p)
        for value in [float('nan'),float('inf'),True,1e7,'1']:
            p=problem();p['A'][0][0]=value
            with self.assertRaises(ValueError):audit(p)
        for y in [True,-1,3,1.5]:
            p=problem();p['examples'][0]['y']=y
            with self.assertRaises(ValueError):audit(p)
        for gauge in [[[1,0],[0,0]],[[1,0],[0,1e-8]],[[1,0,0]],[[1e7,0],[0,1]]]:
            with self.assertRaises(ValueError):audit(problem(),gauge=gauge)
        for lr in [-1,3,float('nan'),True]:
            with self.assertRaises(ValueError):audit(problem(),learning_rate=lr)
        for c in [0,1e-4,33,True,float('inf')]:
            with self.assertRaises(ValueError):audit(problem(),scalar_gauge=c)

    def test_original_one_example_step_exact_coordinates(self):
        fx=build_fixture()
        adapter=LoRAClassifier(fx['base'],2,random.Random(0))
        for _ in range(2):adapter.train_epoch(fx['trainB'],fx['lr'])
        x,y=fx['trainB'][0]
        p={"W":copy.deepcopy(adapter.base.W),"bias":adapter.base.b[:],
           "A":copy.deepcopy(adapter.A),"B":copy.deepcopy(adapter.B),"examples":[{"x":x,"y":y}]}
        r=audit(p,scalar_gauge=4,learning_rate=fx['lr'])
        adapter.train_epoch([(x,y)],fx['lr'])
        self.assertLess(maxdiff(adapter.A,r['original']['A_after']),1e-14)
        self.assertLess(maxdiff(adapter.B,r['original']['B_after']),1e-14)
        self.assertEqual(p['W'],adapter.base.W)
        self.assertEqual(p['bias'],adapter.base.b)


if __name__=='__main__':unittest.main()

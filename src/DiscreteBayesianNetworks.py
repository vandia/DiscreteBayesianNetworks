from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.inference.base import Inference
from pgmpy.factors import factor_product
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

import itertools


class SimpleInference(Inference):

    def query(self, var, evidence):
        # self.factors is a dict of the form of {node: [factors_involving_node]}
        factors_list = set(itertools.chain(*self.factors.values()))
        product = factor_product(*factors_list)
        reduced_prod = product.reduce(evidence, inplace=False)
        #reduced_prod.normalize()
        var_to_marg = set(self.model.nodes()) - set(var) - set([state[0] for state in evidence])
        marg_prod = reduced_prod.marginalize(var_to_marg, inplace=False)
        return marg_prod

def bayesian_net ():
    musicianship_model = BayesianModel([('Difficulty', 'Rating'),
                                  ('Musicianship', 'Rating'),
                                  ('Musicianship', 'Exam'),
                                  ('Rating', 'Letter')])
    cpd_diff = TabularCPD(variable='Difficulty', variable_card=2,
                          values=[[0.6], [0.4]]) #0->Low, 1->High
    cpd_music = TabularCPD(variable='Musicianship', variable_card=2,
                           values=[[0.7], [0.3]]) #0->Weak 1->Strong
    cpd_rating = TabularCPD(variable='Rating', variable_card=3,
                            values=[[0.3, 0.05, 0.9, 0.5],
                                    [0.4, 0.25, 0.08, 0.3],
                                    [0.3, 0.7, 0.02, 0.2]],
                            evidence=['Difficulty', 'Musicianship'],
                            evidence_card=[2, 2]) #0->* 1->** 2-->***
    cpd_exam = TabularCPD(variable='Exam', variable_card=2,
                          values=[[0.95, 0.2], [0.05, 0.8]],
                          evidence=['Musicianship'], evidence_card=[2]) #0-->Low 1-->High

    cpd_letter = TabularCPD(variable='Letter', variable_card=2,
                          values=[[0.1, 0.4, 0.99], [0.9, 0.6, 0.01]],
                            evidence=['Rating'], evidence_card=[3]) #0-->Weak 1-->Strong

    musicianship_model.add_cpds(cpd_diff,cpd_music,cpd_rating,cpd_exam,cpd_letter)
    musicianship_model.check_model()

    infer = SimpleInference(musicianship_model) # query without normalization

    print('------------------------')
    print(' EXACT INFERENCE')
    print('------------------------')
    print('--------------------')
    print(' QUERY Letter with evidence Difficulty: 0, Musicianship: 1, Rating: 1, Exam:1  NOT NORMALIZED')
    print('--------------------')
    print(infer.query(['Letter'],evidence={('Difficulty', 0), ('Musicianship', 1), ('Rating', 1), ('Exam',1)}))
    print('--------------------')
    print(' QUERY Letter with evidence Difficulty: 0, Musicianship: 1, Rating: 1, Exam:1  NORMALIZED')
    print('--------------------')
    infer = VariableElimination(musicianship_model) # query normalized
    print(infer.query(['Letter'], evidence={'Difficulty': 0, 'Musicianship': 1, 'Rating': 1, 'Exam': 1})['Letter'])

    print('--------------------')
    print(' QUERY Letter with no evidence')
    print('--------------------')
    print(infer.query(['Letter'])['Letter'])
    print('--------------------')
    print(' QUERY Letter with evidence Musicianship: 0  NORMALIZED')
    print('--------------------')
    print(infer.query(['Letter'],evidence={'Musicianship': 0})['Letter'])

    sampling = BayesianModelSampling(musicianship_model)
    data = sampling.likelihood_weighted_sample(evidence={}, size=2000, return_type='dataframe')

    musicianship_model_bis=BayesianModel([('Difficulty', 'Rating'),
                                  ('Musicianship', 'Rating'),
                                  ('Rating', 'Letter'),
                                  ('Musicianship', 'Exam')])
    musicianship_model_bis.fit(data, estimator=BayesianEstimator)
    musicianship_model_bis.check_model()
    infer = VariableElimination(musicianship_model_bis) # query normalized
    for cpd in musicianship_model_bis.get_cpds():
        print("CPD of {variable}:".format(variable=cpd.variable))
        print(cpd)


    print('------------------------')
    print(' APPROXIMATE INFERENCE')
    print('------------------------')

    print('--------------------')
    print(' QUERY Letter with evidence Difficulty: 0, Musicianship: 1, Rating: 1, Exam:1  NORMALIZED')
    print('--------------------')

    print(infer.query(['Letter'], evidence={'Difficulty': 0, 'Musicianship': 1, 'Rating': 1, 'Exam': 1})['Letter'])

    print('--------------------')
    print(' QUERY Letter with no evidence')
    print('--------------------')
    print(infer.query(['Letter'])['Letter'])
    print('--------------------')
    print(' QUERY Letter with evidence Musicianship: 0  NORMALIZED')
    print('--------------------')
    print(infer.query(['Letter'],evidence={'Musicianship': 0})['Letter'])

if __name__ == '__main__':
    bayesian_net()
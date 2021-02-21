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
        reduced_prod = product.reduce([(evidence_var, evidence[evidence_var]) for evidence_var in evidence], inplace=False)
        #reduced_prod.normalize()
        var_to_marg = set(self.model.nodes()) - set(var) - set([state for state in evidence])

        query_var_factor = {}
        for query_var in var:
            query_var_factor[query_var] = reduced_prod.marginalize(var_to_marg, inplace=False)
        return query_var_factor



def bayesian_net():
    alarm_model = BayesianModel([('Burglary', 'Alarm'),  # Alarm has two parents, thus, it is twice as son.
                                 ('Earthquake', 'Alarm'),
                                 ('Alarm', 'JohnCalls'),
                                 ('Alarm', 'MaryCalls')])
    for i in alarm_model.get_parents('Alarm'):
        print(i)

    # variable_card indicates the number of posible values this variable can take.

    cpd_burglary = TabularCPD(variable='Burglary', variable_card=2, # 0->True, 1->False
                              values=[[0.001],  # true probabilities of the table
                                      [0.999]])  # false probabilities of the table
    cpd_earthquake = TabularCPD(variable='Earthquake', variable_card=2,# 0->True 1->False
                                values=[[0.002],  # true probabilities of the table
                                        [0.998]])  # false probabilities of the table

    # evidence_card indicates the number of possible values the parents of the variable can take

    cpd_alarm = TabularCPD(variable='Alarm', variable_card=2, # 0->True 1->False
                           values=[[0.95, 0.94, 0.29, 0.001],  # true probabilities of the table
                                   [0.05, 0.06, 0.71, 0.999]], # false probabilities of the table
                           evidence=['Burglary', 'Earthquake'],
                           evidence_card=[2, 2])
    cpd_john_calls = TabularCPD(variable='JohnCalls', variable_card=2, # 0->True 1->False
                                values=[[0.95, 0.05],
                                        [0.05, 0.95]],
                                evidence=['Alarm'], evidence_card=[2])

    cpd_mary_calls = TabularCPD(variable='MaryCalls', variable_card=2, # 0->True 1->False
                            values=[[0.7, 0.1],  # true probabilities of the table
                                    [0.3, 0.9]], # false probabilities of the table
                            evidence=['Alarm'], evidence_card=[2])
    for i in [cpd_burglary,cpd_earthquake,cpd_alarm,cpd_john_calls,cpd_mary_calls]:
        print(i)

    alarm_model.add_cpds(cpd_burglary,cpd_earthquake,cpd_alarm,cpd_john_calls,cpd_mary_calls)
    alarm_model.check_model()

    infer = VariableElimination(alarm_model)

    # Uncomment to obtain the result before normalization
    # infer = SimpleInference(alarm_model)


    print(infer.query(['JohnCalls'],evidence={'Burglary': 1, 'Earthquake': 1, 'Alarm': 0, 'MaryCalls': 0},)['JohnCalls'])

    print(infer.query(['Burglary'],evidence={'JohnCalls': 0, 'MaryCalls': 0})['Burglary'])

    # Variable order can be specified if necessary

    print(infer.query(['Burglary'], evidence={'JohnCalls': 0, 'MaryCalls': 0}, elimination_order=['Alarm','Earthquake'])['Burglary'])


    sampling = BayesianModelSampling(alarm_model)
    data = sampling.rejection_sample(evidence={}, size=20, return_type="dataframe")
    print(data)



    data = sampling.rejection_sample(evidence=[('JohnCalls',0), ('MaryCalls',0)], size=20,
                                               return_type = 'dataframe')
    print (data)

    sampling = BayesianModelSampling(alarm_model)
    data = sampling.rejection_sample(evidence=None, size=5000, return_type="dataframe")
    approx_alarm_model=BayesianModel([('Burglary', 'Alarm'),
                                 ('Earthquake', 'Alarm'),
                                 ('Alarm', 'JohnCalls'),
                                 ('Alarm', 'MaryCalls')])
    approx_alarm_model.fit(data, estimator=BayesianEstimator)
    approx_alarm_model.check_model()

    for cpd in approx_alarm_model.get_cpds():
        print("CPD of {variable}:".format(variable=cpd.variable))
        print(cpd)


    infer = VariableElimination(approx_alarm_model)

    print(infer.query(['JohnCalls'],evidence={'Burglary': 1, 'Earthquake': 1, 'Alarm': 0, 'MaryCalls': 0},)['JohnCalls'])

    print(infer.query(['Burglary'],evidence={'JohnCalls': 0, 'MaryCalls': 0})['Burglary'])

    print(alarm_model.predict_probability(data[['Burglary', 'Earthquake', 'Alarm', 'JohnCalls']]))


if __name__ == '__main__':
    bayesian_net()

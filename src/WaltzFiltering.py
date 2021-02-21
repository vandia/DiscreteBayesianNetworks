from constraint import *

A = 0
R = 1
M = 2
P = 3


def l_constraint(x, y):
    return (x == R and y == P) or (x == R and y == R) or (x == P and y == R) or (x == A and y == M) or \
           (x == A and y == A) or (x == M and y == A)


def arrow_constraint(x, y, z):
    return (x == A and y == P and z == A) or (x == M and y == P and z == M) or (x == P and y == M and z == P)


def fork_constraint(x, y, z):
    return (x == A and y == A and z == M) or (x == M and y == A and z == A) or (x == A and y == M and z == A) or (
            x == P and y == P and z == P) or (x == M and y == M and z == M)


def waltz_filtering():
    problem = Problem()
    variables = ['E' + str(i) for i in range(1, 16)]
    problem.addVariables(variables, range(0, 4))
    problem.addConstraint(FunctionConstraint(l_constraint), ['E1', 'E2'])
    problem.addConstraint(FunctionConstraint(l_constraint), ['E5', 'E6'])
    problem.addConstraint(FunctionConstraint(l_constraint), ['E7', 'E8'])

    problem.addConstraint(FunctionConstraint(fork_constraint), ['E4', 'E3', 'E14'])
    problem.addConstraint(FunctionConstraint(fork_constraint), ['E9', 'E11', 'E10'])
    problem.addConstraint(FunctionConstraint(fork_constraint), ['E12', 'E13', 'E15'])

    problem.addConstraint(FunctionConstraint(arrow_constraint), ['E8', 'E9', 'E1'])
    problem.addConstraint(FunctionConstraint(arrow_constraint), ['E2', 'E10', 'E3'])
    problem.addConstraint(FunctionConstraint(arrow_constraint), ['E4', 'E15', 'E5'])
    problem.addConstraint(FunctionConstraint(arrow_constraint), ['E6', 'E13', 'E7'])
    problem.addConstraint(FunctionConstraint(arrow_constraint), ['E12', 'E14', 'E11'])

    for sol in problem.getSolutions():
        print([sol[j] for j in variables])

        print("\n")


if __name__ == '__main__':
    waltz_filtering()

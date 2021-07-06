"""@package scipy_dumper
Dumper for the scipy optimization methods.
"""
import numpy as np


class ScipyBasicDumper:
    def __init__(self, x_file, fun_file, emod=None):
        """ Constructor.

        :param str x_file: Path to the file to write the primal variable history.
        :param str fun_file: Path the the file to write the objective function history.
        :param float emod: Elastic modulus, optional, used when elastic modulus is specified directly.

        Notes:
            - The constructor clears the files.
        """
        self.dump_file = x_file
        self.function_file = fun_file
        self.emod = emod

        with open(self.dump_file, 'w') as f:
            # Clear the file
            pass

        with open(self.function_file, 'w') as f:
            f.write('iteration, function, norm_grad_Lagr\n')

    def dump(self, x, state):
        if self.emod is not None:
            x2 = np.array([self.emod] + list(x))
        else:
            x2 = x
        it_num = state.niter
        f_val = state.fun
        norm_grad_lag = state.optimality
        with open(self.dump_file, 'a') as fi:
            np.savetxt(fi, x2.reshape((1, len(x2))), fmt='%7.6e')
        with open(self.function_file, 'a') as fi:
            fi.write('{0}, {1:5.4e}, {2:5.4e}\n'.format(it_num, f_val, norm_grad_lag))

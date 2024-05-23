import numpy as np
import scipy.sparse.csgraph as spgr
from openpnm.topotools import is_fully_connected
from openpnm.algorithms import Algorithm
from openpnm import solvers
from ._solution import SteadyStateSolution, SolutionContainer


__all__ = ['Transport']


class Transport(Algorithm):
    """
    This class implements steady-state linear transport calculations.

    Parameters
    ----------
    %(Algorithm.parameters)s

    """

    def __init__(self, phase, name='trans_?', **kwargs):
        super().__init__(name=name, **kwargs)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if key == 'pore.source':
                return {}
            else:
                raise KeyError(key)

    def build_A(self, gvals):
        am = create_adjacency_matrix(weights=gvals, fmt='coo')
        A = spgr.laplacian(am).astype(float)
        return A

    def build_b(self):
        """Initializes the RHS vector, b, with zeros."""
        b = np.zeros(self.Np, dtype=float)
        return b

    def apply_BCs(self, A, b, locs, values=None, rates=None):
        mask = np.zeros_like(b, dtype=False)
        mask[locs] = True
        if rates is not None:
            self.b[locs] = rates
        if values is not None:
            f = A.diagonal().mean()
            # Update b (impose bc values)
            b[locs] = values * f
            # Update b (subtract quantities from b to keep A symmetric)
            x_BC = np.zeros_like(b)
            x_BC[locs] = b
            b[~mask] -= (A * x_BC)[~mask]
            # Update A
            temp = np.isin(A.row, locs) | np.isin(A.col, locs)
            # Remove entries from A for all BC rows/cols
            A.data[temp] = 0
            # Add diagonal entries back into A
            datadiag = A.diagonal()
            datadiag[mask] = np.ones_like(mask, dtype=float) * f
            A.setdiag(datadiag)
            A.eliminate_zeros()
        return A, b

    def run(self, solver=None, x0=None, verbose=False):
        """
        Builds the A and b matrices, and calls the solver specified in the
        ``settings`` attribute.

        This method stores the solution in the algorithm's ``soln``
        attribute as a ``SolutionContainer`` object. The solution itself
        is stored in the ``x`` attribute of the algorithm as a NumPy array.

        Parameters
        ----------
        x0 : ndarray
            Initial guess of unknown variable

        Returns
        -------
        None

        """
        if solver is None:
            solver = getattr(solvers, ws.settings.default_solver)()
        # Perform pre-solve validations
        self._validate_settings()
        self._validate_topology_health()
        self._validate_linear_system()
        # Write x0 to algorithm (needed by _update_iterative_props)
        self.x = x0 = np.zeros_like(self.b) if x0 is None else x0.copy()
        self["pore.initial_guess"] = x0
        self._validate_x0()
        # Initialize the solution object
        self.soln = SolutionContainer()
        self.soln[self.settings['quantity']] = SteadyStateSolution(x0)
        self.soln.is_converged = False
        # Build A and b, then solve the system of equations
        self._update_A_and_b()
        self._run_special(solver=solver, x0=x0, verbose=verbose)

    def _run_special(self, solver, x0, w=1.0, verbose=None):
        # Make sure A and b are 'still' well-defined
        self._validate_linear_system()
        # Solve and apply under-relaxation
        x_new, exit_code = solver.solve(A=self.A, b=self.b, x0=x0)
        self.x = w * x_new + (1 - w) * self.x
        # Update A and b using the recent solution otherwise, for iterative
        # algorithms, residual will be incorrectly calculated ~0, since A & b
        # are outdated
        self._update_A_and_b()
        # Update SteadyStateSolution object on algorithm
        self.soln[self.settings['quantity']][:] = self.x
        self.soln.is_converged = not bool(exit_code)

    def _update_A_and_b(self):
        """Builds A and b, and applies specified boundary conditions."""
        self._build_A()
        self._build_b()
        self._apply_BCs()

    def _validate_x0(self):
        """Ensures x0 doesn't contain any nans/infs."""
        x0 = self["pore.initial_guess"]
        if not np.isfinite(x0).all():
            raise Exception("x0 contains inf/nan values")

    def _validate_settings(self):
        if self.settings['quantity'] is None:
            raise Exception("'quantity' hasn't been defined on this algorithm")
        if self.settings['conductance'] is None:
            raise Exception("'conductance' hasn't been defined on this algorithm")
        if self.settings['phase'] is None:
            raise Exception("'phase' hasn't been defined on this algorithm")

    def _validate_topology_health(self):
        """
        Ensures the network is not clustered, and if it is, they're at
        least connected to a boundary condition pore.
        """
        Ps = ~np.isnan(self['pore.bc.rate']) + ~np.isnan(self['pore.bc.value'])
        if not is_fully_connected(network=self.network, pores_BC=Ps):
            msg = ("Your network is clustered, making Ax = b ill-conditioned")
            raise Exception(msg)

    def _validate_linear_system(self):
        """Ensures the linear system Ax = b doesn't contain any nans/infs."""
        if np.isfinite(self.A.data).all() and np.isfinite(self.b).all():
            return
        raise Exception("A or b contains inf/nan values")

    def rate(self, pores=[], throats=[], mode='group'):
        """
        Calculates the net rate of material moving into a given set of
        pores or throats

        Parameters
        ----------
        pores : array_like
            The pores for which the rate should be calculated
        throats : array_like
            The throats through which the rate should be calculated
        mode : str, optional
            Controls how to return the rate. The default value is 'group'.
            Options are:

            ===========  =====================================================
            mode         meaning
            ===========  =====================================================
            'group'      Returns the cumulative rate of material
            'single'     Calculates the rate for each pore individually
            ===========  =====================================================

        Returns
        -------
        If ``pores`` are specified, then the returned values indicate the
        net rate of material exiting the pore or pores.  Thus a positive
        rate indicates material is leaving the pores, and negative values
        mean material is entering.

        If ``throats`` are specified the rate is calculated in the
        direction of the gradient, thus is always positive.

        If ``mode`` is 'single' then the cumulative rate through the given
        pores (or throats) are returned as a vector, if ``mode`` is
        'group' then the individual rates are summed and returned as a
        scalar.

        """
        pores = self._parse_indices(pores)
        throats = self._parse_indices(throats)

        if throats.size > 0 and pores.size > 0:
            raise Exception('Must specify either pores or throats, not both')
        if (throats.size == 0) and (pores.size == 0):
            raise Exception('Must specify either pores or throats')

        network = self.project.network
        phase = self.project[self.settings['phase']]
        g = phase[self.settings['conductance']]

        P12 = network['throat.conns']
        X12 = self.x[P12]
        if g.size == self.Nt:
            g = np.tile(g, (2, 1)).T    # Make conductance an Nt by 2 matrix
        # The next line is critical for rates to be correct
        # We could also do "g.T.flatten()" or "g.flatten('F')"
        g = np.flip(g, axis=1)
        Qt = np.diff(g*X12, axis=1).ravel()

        if throats.size:
            R = np.absolute(Qt[throats])
            if mode == 'group':
                R = np.sum(R)
        elif pores.size:
            Qp = np.zeros((self.Np, ))
            np.add.at(Qp, P12[:, 0], -Qt)
            np.add.at(Qp, P12[:, 1], Qt)
            R = Qp[pores]
            if mode == 'group':
                R = np.sum(R)

        return np.array(R, ndmin=1)

    def clear_value_BCs(self):
        self.set_BC(pores=None, bctype='value', mode='remove')

    def clear_rate_BCs(self):
        self.set_BC(pores=None, bctype='rate', mode='remove')

    def set_value_BC(self, pores=None, values=[], mode='add'):
        self.set_BC(pores=pores, bctype='value', bcvalues=values, mode=mode)

    def set_rate_BC(self, pores=None, rates=[], mode='add'):
        self.set_BC(pores=pores, bctype='rate', bcvalues=rates, mode=mode)

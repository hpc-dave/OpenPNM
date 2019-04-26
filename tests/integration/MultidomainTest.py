import openpnm as op
import scipy as sp


class MultipleDomainTest():

    def setup_class(self):
        pass

    def test_multiple_domains_all_pores_in_one_domain(self):
        pn = op.network.Cubic(shape=[3, 3, 3])
        Ts = pn.find_neighbor_throats(pores=9)
        geo1 = op.geometry.StickAndBall(network=pn, pores=pn.Ps, throats=Ts)
        Ts = ~pn['throat.geo_01']
        geo2 = op.geometry.StickAndBall(network=pn, pores=[], throats=Ts)

    def test_multiple_domains_all_pores_in_other_domain(self):
        pn = op.network.Cubic(shape=[3, 3, 3])
        Ts = pn.find_neighbor_throats(pores=9)
        geo1 = op.geometry.StickAndBall(network=pn, pores=[], throats=Ts)
        Ts = ~pn['throat.geo_01']
        geo2 = op.geometry.StickAndBall(network=pn, pores=pn.Ps, throats=Ts)

    def test_multiple_domains_single_pore_in_one_domain(self):
        pn = op.network.Cubic(shape=[3, 3, 3])
        Ts = pn.find_neighbor_throats(pores=9)
        geo1 = op.geometry.StickAndBall(network=pn, pores=[0], throats=Ts)
        Ts = ~pn['throat.geo_01']
        geo2 = op.geometry.StickAndBall(network=pn, pores=range(1, 27), throats=Ts)


if __name__ == '__main__':

    t = MultipleDomainTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()

import numpy as np
from uuid import uuid4
from pnmlib.core import get_data, set_data, count


__all__ = [
    'add_network',
    'create_phase',
    'get_group',
]


def get_group(project, name):
    if name in ['/', '', '*']:
        return project
    temp = get_data(project, name+"/*")
    group = {}
    for k, v in temp.items():
        if not hasattr(v, 'keys'):
            group[k.rsplit('/', 1)[1]] = v
    return group


def add_network(project, network):
    coords = network.pop('pore.coords')
    set_data(project, 'pore.x', coords[:, 0])
    set_data(project, 'pore.y', coords[:, 1])
    set_data(project, 'pore.z', coords[:, 2])
    conns = network.pop('throat.conns')
    set_data(project, 'throat.t', conns[:, 0])
    set_data(project, 'throat.h', conns[:, 1])
    for k, v in network.items():
        if v.dtype == bool:
            set_data(project, k, v)
        else:
            set_data(project, 'network' + '/' + k, v)


def create_phase(target, name=None):
    if name is None:
        name = generate_name(target, prefix='phase')
    create_group(target=target, name=name)


def generate_name(target, prefix):
    i = 1
    while True:
        if prefix + '_' + str(i).zfill(2) not in target.keys():
            name = prefix + '_' + str(i).zfill(2)
            break
        i += 1
    return name


def create_group(target, name):
    for element in ['pore', 'throat']:
        try:
            n = count(target, element)
        except Exception:
            n = count(target['network'], element)
        set_data(target, name + '/' + element + '.all', np.ones((n, ), dtype=bool))
    return target


# %% Below here is rubbish

def validate_group(group):
    # Will scan given group to make sure all rules are met. This will be called
    # before writing a nested dict, like "pn['subgroup'] = Group()"
    # Note that this could also be called after using "update" method to batch
    # write an existing dictionary like network.
    raise NotImplementedError()


def update_parents(project):
    # Will scan entire project and make sure each child is properly associated
    # with it's parent by doing an OpenPNM style top-down scan
    raise NotImplementedError()


def update_project(project):
    # Will can a project and ensure that all objects/gropus have the project's ID
    # in their attr['project'].
    raise NotImplementedError()


def find_group_name(group):
    parent = find_parent(group)
    for k, v in parent.items():
        if hasattr(v, 'keys') and 'attrs' in v.keys():
            if v['attrs']['ID'] == group['attrs']['ID']:
                return k


def find_parent(group):
    # This might be a bit too fancy, and could actually be done using the
    # openpnm way to doing a top down scan until finding itself
    def dfs(p, group):
        for g in p.values():
            if hasattr(g, 'keys') and 'attrs' in g.keys():
                if g['attrs']['ID'] == group['attrs']['parent']:
                    return g
                else:
                    dfs(g, group)

    ws = Workspace()
    for p in ws.projects:
        return dfs(p, group)


def find_project(group):
    ws = Workspace()
    for p in ws.projects:
        if 'project' in group['attrs'].keys():
            if p['attrs']['ID'] == group['attrs']['project']:
                return p
        else:
            parent = find_parent(group)
            return find_project(parent)


def find_network(item):
    project = find_project(item)
    network = project['network']
    return network


def create_project(backend='dict'):
    if backend == 'zarr':
        import zarr
        project = zarr.group()
        project['attrs']['ID'] = str(uuid4())
    elif backend == 'dict':
        project = Project()
    ws = Workspace()
    ws.projects.append(project)
    return project


def add_group(target, name, group=None, data=True):
    if group is None:
        group = Group()
    target[name] = group
    # target[name]['attrs']['parent'] = get_data(target, 'attrs/ID')


class Group(dict):
    r"""
    This class can grow a lot by adding convenience wrappers for all the various
    functions above. Not sure if this is a good idea or not...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This next line requires "SubDicts" which is a problem. MAYBE, these
        # attrs should be stored in the dictionary itself, perhaps along with
        # all the params?  Settings could go in there too.  We would then pull
        # them out and attach them as attrs when saving to zarr?
        self['attrs'] = {}
        # Every Group has its own ID, but crucially they all track the ID of their
        # 'parent' and their 'project', to facilitate very easy and robust look-ups.
        self['attrs']['ID'] = str(uuid4())
        self['attrs']['parent'] = ''

    def __getitem__(self, key):
        return get_data(self, key)

    def __setitem__(self, key, value):
        set_data(self, key, value)


class Project(Group):
    r"""
    This subclass of Group does one thing...it stores its own ID as its project
    so that lookups don't break.  It also gives it a recognizable name.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.attrs['ID'] is created by parent class
        self['attrs']['project'] = self['attrs']['ID']


class Workspace:
    r"""
    Unlike the Project class, which can be a UserDict if we wish, this *needs* to
    be a subclass.  The key insight of this class is that class attributes act like
    global variable, so we can store the projects in this list attribute then
    access this from anywhere.
    """

    projects = []  # This class attribute acts like a global variable

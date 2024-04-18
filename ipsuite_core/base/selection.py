"""Base Node for ConfigurationSelection."""

import dataclasses
import logging
import typing

import ase
import matplotlib.pyplot as plt
import numpy as np
import znflow
import zntrack

from ipsuite_core import base

log = logging.getLogger(__name__)


def get_flat_data_from_dict(data: dict, silent_ignore: bool = False) -> list:
    """Flatten a dictionary of lists into a single list.

    Parameters
    ----------
    data : dict
        Dictionary of lists.
    silent_ignore : bool, optional
        If True, the function will return the input if it is not a
        dictionary. If False, it will raise a TypeError.

    Example
    -------
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> get_flat_data_from_dict(data)
        [1, 2, 3, 4, 5, 6]

    """
    if not isinstance(data, dict):
        if silent_ignore:
            return data
        else:
            raise TypeError(f"data must be a dictionary and not {type(data)}")

    flat_data = []
    for x in data.values():
        flat_data.extend(x)
    return flat_data


def get_ids_per_key(
    data: dict, ids: list, silent_ignore: bool = False
) -> typing.Dict[str, list]:
    """Get the ids per key from a dictionary of lists.

    Parameters
    ----------
    data : dict
        Dictionary of lists.
    ids : list
        List of ids. The ids are assumed to be taken from the flattened
        'get_flat_data_from_dict(data)' data. If the ids aren't sorted,
        they will be sorted.
    silent_ignore : bool, optional
        If True, the function will return the input if it is not a
        dictionary. If False, it will raise a TypeError.

    Example
    -------
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> get_ids_per_key(data, [0, 1, 3, 5])
        {'a': [0, 1], 'b': [0, 2]}

    """
    if not isinstance(data, dict):
        if silent_ignore:
            return np.array(ids).tolist()
        else:
            raise TypeError(f"data must be a dictionary and not {type(data)}")

    ids_per_key = {}
    ids = np.array(ids).astype(int)
    ids = np.sort(ids)
    start = 0

    for key, val in data.items():
        condition = ids - start
        condition = np.logical_and(condition < len(val), condition >= 0)

        ids_per_key[key] = np.array(ids[condition] - start).tolist()
        start += len(val)

    return ids_per_key


@dataclasses.dataclass
class ExcludeIds:
    """Remove entries from a dataset."""

    data: typing.Union[list, dict]
    ids: typing.Union[list, dict]

    def __post_init__(self):
        """Sort the ids and check if they are valid."""
        if self.ids is None:
            return
        if isinstance(self.ids, list):
            log.debug("ids is list")
            if isinstance(self.ids[0], dict):
                log.debug("ids is list of dicts")
                # we assume list[dict]. IF mixed it will raise some error
                ids = {}
                for data in self.ids:
                    for key, value in data.items():
                        if key in ids:
                            ids[key].extend(value)
                        else:
                            if not isinstance(value, list):
                                raise ValueError(
                                    f"Ids can not be {type(value)} but must be "
                                    f"int Found {value} instead."
                                )
                            ids[key] = value
                self.ids = {}
                for key, val in ids.items():
                    self.ids[key] = np.sort(val).astype(int).tolist()
            else:
                log.debug("ids is list of ints")
                self.ids = np.sort(self.ids).astype(int).tolist()
        else:
            log.debug("ids is dict")
            for key, ids in self.ids.items():
                self.ids[key] = np.sort(ids).astype(int).tolist()

    def get_clean_data(self, flatten: bool = False) -> list:
        """Remove the 'ids' from the 'data'."""
        # TODO do we need a dict return here or could we just return a flat list?
        if self.ids is None:
            if isinstance(self.data, list):
                return self.data
            elif isinstance(self.data, dict):
                if flatten:
                    return get_flat_data_from_dict(self.data)
                return self.data
        if isinstance(self.data, list) and isinstance(self.ids, list):
            return [x for i, x in enumerate(self.data) if i not in self.ids]
        elif isinstance(self.data, dict) and isinstance(self.ids, dict):
            clean_data = {}
            for key, data in self.data.items():
                if key in self.ids:
                    clean_data[key] = [
                        x for i, x in enumerate(data) if i not in self.ids[key]
                    ]
                else:
                    clean_data[key] = data
            if flatten:
                return get_flat_data_from_dict(clean_data)
            return clean_data
        else:
            raise TypeError(
                "ids and data must be of the same type. "
                f"ids is {type(self.ids)} and data is {type(self.data)}"
            )

    def get_original_ids(self, ids: list, per_key: bool = False) -> list:
        """Shift the 'ids' such that they are valid for the initial data."""
        ids = np.array(ids).astype(int)
        ids = np.sort(ids)

        if isinstance(self.ids, list):
            for removed_id in self.ids:
                ids[ids >= removed_id] += 1
        elif isinstance(self.ids, dict):
            for removed_id in self.ids_as_list:
                ids[ids >= removed_id] += 1
        if per_key:
            return get_ids_per_key(self.data, ids, silent_ignore=True)
        return ids.tolist()

    @property
    def ids_as_list(self) -> list:
        """Return the ids as a list."""
        # {a: [1, 2], b: [1, 3]}
        # {a: list(10), b:list(10)}
        # [1, 2, 1+10, 3+10]
        ids = []
        size = 0
        for key in self.data:
            # we iterate through data, not ids, because ids must not contain all keys
            if key in self.ids:
                ids.append(np.array(self.ids[key]) + size)
            size += len(self.data[key])
        if len(ids):
            ids = np.concatenate(ids)
            ids = np.sort(ids)
            return ids.astype(int).tolist()
        return []


class ConfigurationSelection(base.ProcessAtoms):
    """Base Node for ConfigurationSelection.

    Attributes
    ----------
    data: list[Atoms]|list[list[Atoms]]|utils.types.SupportsAtoms
        the data to select from
    exclude_configurations: dict[str, list]|utils.types.SupportsSelectedConfigurations
        Atoms to exclude from the
    exclude: list[zntrack.Node]|zntrack.Node|None
        Exclude the selected configurations from these nodes.

    """

    exclude_configurations: typing.Union[
        typing.Dict[str, typing.List[int]], base.protocol.HasSelectedConfigurations
    ] = zntrack.deps(None)
    exclude: typing.Union[zntrack.Node, typing.List[zntrack.Node]] = zntrack.deps(None)
    selected_configurations: typing.Dict[str, typing.List[int]] = zntrack.outs()

    img_selection = zntrack.outs_path(zntrack.nwd / "selection.png")

    _name_ = "ConfigurationSelection"

    def _post_init_(self):
        if self.data is not None and not isinstance(self.data, dict):
            try:
                self.data = znflow.combine(
                    self.data, attribute="atoms", return_dict_attr="name"
                )
            except TypeError:
                self.data = znflow.combine(self.data, attribute="atoms")

    def run(self):
        """ZnTrack Node Run method."""
        if self.exclude is not None:
            if self.exclude_configurations is None:
                self.exclude_configurations = {}
            if not isinstance(self.exclude, list):
                self.exclude = [self.exclude]
            for exclude in self.exclude:
                for key in exclude.selected_configurations:
                    if key in self.exclude_configurations:
                        self.exclude_configurations[key].extend(
                            exclude.selected_configurations[key]
                        )
                    else:
                        self.exclude_configurations[key] = (
                            exclude.selected_configurations[key]
                        )

        exclude = ExcludeIds(self.get_data(), self.exclude_configurations)
        data = exclude.get_clean_data(flatten=True)

        log.debug(f"Selecting from {len(data)} configurations.")

        selected_configurations = self.select_atoms(data)

        self.selected_configurations = exclude.get_original_ids(
            selected_configurations, per_key=True
        )

        self._get_plot(data, selected_configurations)

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Run the selection method.

        Attributes
        ----------
        atoms_lst: List[ase.Atoms]
            List of ase Atoms objects to select configurations from.

        Returns
        -------
        List[int]:
            A list of the selected ids from 0 .. len(atoms_lst)

        """
        raise NotImplementedError

    @property
    def atoms(self) -> typing.Sequence[ase.Atoms]:
        """Get a list of the selected atoms objects."""
        with znflow.disable_graph():
            results = []
            data = self.get_data()
            if isinstance(data, list):
                for idx, atoms in enumerate(self.get_data()):
                    if idx in self.selected_configurations:
                        results.append(atoms)
            elif isinstance(data, dict):
                # This only triggers, if the file was changed manually.
                if data.keys() != self.selected_configurations.keys():
                    raise ValueError(
                        f"Data keys {data.keys()} must match"
                        f" selected keys {self.selected_configurations.keys()}"
                    )
                for key, atoms_lst in data.items():
                    if key in self.selected_configurations:
                        for idx, atoms in enumerate(atoms_lst):
                            if idx in self.selected_configurations[key]:
                                results.append(atoms)
            else:
                raise ValueError(f"Data must be a list or dict, not {type(data)}")
            return results

    @property
    def excluded_atoms(self) -> typing.Sequence[ase.Atoms]:
        """Get a list of the atoms objects that were not selected."""
        with znflow.disable_graph():
            results = []
            data = self.get_data()
            if isinstance(data, list) and isinstance(
                self.selected_configurations, list
            ):
                for idx, atoms in enumerate(data):
                    if idx not in self.selected_configurations:
                        results.append(atoms)
            elif isinstance(data, dict) and isinstance(
                self.selected_configurations, dict
            ):
                # This only triggers, if the file was changed manually.
                if data.keys() != self.selected_configurations.keys():
                    raise ValueError(
                        f"Data keys {data.keys()} must match"
                        f" selected keys {self.selected_configurations.keys()}"
                    )
                for key, atoms_lst in data.items():
                    if key not in self.selected_configurations:
                        results.extend(atoms_lst)
                    else:
                        for idx, atoms in enumerate(atoms_lst):
                            if idx not in self.selected_configurations[key]:
                                results.append(atoms)
            else:
                raise ValueError(f"Data must be a list or dict, not {type(data)}")
            return results

    def _get_plot(self, atoms_lst: typing.List[ase.Atoms], indices: typing.List[int]):
        """Plot the selected configurations."""
        # if energies are available, plot them, otherwise just plot indices over time
        fig, ax = plt.subplots()

        try:
            line_data = np.array([atoms.get_potential_energy() for atoms in atoms_lst])
            ax.set_ylabel("Energy")
        except Exception:
            line_data = np.arange(len(atoms_lst))
            ax.set_ylabel("Configuration")

        ax.plot(line_data)
        ax.scatter(indices, line_data[indices], c="r")
        ax.set_xlabel("Configuration")
        fig.savefig(self.img_selection, bbox_inches="tight")


class BatchConfigurationSelection(ConfigurationSelection):
    """Base node for BatchConfigurationSelection.

    Attributes
    ----------
    data: list[ase.Atoms]
        The atoms data to process. This must be an input to the Node
    atoms: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
        It does not have to be 'field.Atoms' but can also be e.g. a 'property'.

    """

    train_data: list[ase.Atoms] = zntrack.deps()

    def _post_init_(self):
        if self.train_data is not None and not isinstance(self.train_data, dict):
            try:
                self.train_data = znflow.combine(
                    self.train_data, attribute="atoms", return_dict_attr="name"
                )
            except TypeError:
                self.train_data = znflow.combine(self.train_data, attribute="atoms")

        if self.data is not None and not isinstance(self.data, dict):
            try:
                self.data = znflow.combine(
                    self.data, attribute="atoms", return_dict_attr="name"
                )
            except TypeError:
                self.data = znflow.combine(self.data, attribute="atoms")

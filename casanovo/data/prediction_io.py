from typing import Any, List, Tuple, Type, Callable


class PredictionWriter:
    def log_prediction(
        self,
        next_prediction: Tuple[
            str, Tuple[str, str], float, float, float, float, str
        ],
    ) -> None:
        """
        Add new prediction to writer context

        Parameters
        ----------
        next_prediction : Tuple[str, Tuple[str, str], float, float, float, float, str]
            Tuple containing next prediction data. The tuple should contain the following:
                - str: next peptide prediction
                - Tuple[str, str]: sample origin file path, origin file index number ("index={i}")
                - float: peptide prediction score (search engine score)
                - float: charge
                - float: precursor m/z
                - float: peptide mass
                - str: aa scores for each peptide in sequence, comma separated
        """
        pass

    def log_skipped_spectra(self, num_skipped: int) -> None:
        """
        Append some number of skipped spectra to the writer context

        Parameters
        ----------
        num_skipped : str
            number of skipped spectra
        """
        pass

    def save(self) -> None:
        """
        Save predictions
        """
        pass


def get_writer_methods() -> List[str]:
    """
    Get list defining the PredictionWriter interface methods

    Returns
    -------
        List[str]:
            list containing the names of all of the methods defined in the
            PredictionWriter interface
    """
    return [
        attr
        for attr in dir(PredictionWriter)
        if callable(getattr(PredictionWriter, attr))
        and not attr.startswith("__")
    ]


class PredictionMultiWriter(PredictionWriter):
    """
    Write predictions to multiple prediction writers

    Parameters
    ----------
        writers : List[Type[PredictionWriter]]
            prediction writers to write to
    """

    def __init__(self, writers: List[Type[PredictionWriter]]) -> None:
        self.writers = writers

        # Dynamically create member functions from PredictionWriter methods
        self.__setstate__(dict())

    def __multi_writer_method_factory(self, method_name: str) -> Callable:
        """
        Create a function that calls method_name on all writers in the
        writer list

        Parameters
        ----------
            method_name : str
                name of writer member

        Returns
        -------
            Callable: function that calls method_name on all writers in the writer list
        """

        def __writer_passthrough_fun(*args, **kwargs):
            for writer in self.writers:
                getattr(writer, method_name)(*args, **kwargs)

        return __writer_passthrough_fun

    def __reduce__(
        self,
    ) -> Tuple[Callable, Tuple[List[PredictionWriter]], dict]:
        """
        Reduce method for object serialization.

        Returns
        -------
            Tuple[Callable, Tuple[List[PredictionWriter]], dict]:
                A tuple containing a callable object that can create a new PredictionMultiWriter,
                tuple of arguments to pass into said callable object, and a dictionary
                representation of the current object state
        """
        return (self.__class__, (self.writers,), self.__getstate__())

    def __getstate__(self) -> dict:
        """
        Get object state sans dynamically generated member functions, used for
        pickeling

        Returns
        -------
            state : dict
                current object state dict not including dynamically generated member functions
        """
        writer_methods = get_writer_methods()

        return {
            attr: attr_state
            for attr, attr_state in self.__dict__.items()
            if attr not in writer_methods
        }

    def __setstate__(self, state: dict) -> None:
        """
        Regenerate dynamic member functions after resetting state

        Parameters
        ----------
            state : dict
                state dictionary used to reset multiwriter state
        """
        self.__dict__.update(state)

        # Restore dynamically generated methods
        for writer_method in get_writer_methods():
            self.__dict__[writer_method] = self.__multi_writer_method_factory(
                writer_method
            )

    def add_writer(self, writer: Type[PredictionWriter]) -> None:
        """
        Add writer to prediction multi-writer

        Parameters
        ----------
            writer : Type[PredictionWriter]
                writer to add to prediction multi-writer
        """
        self.writers.append(writer)

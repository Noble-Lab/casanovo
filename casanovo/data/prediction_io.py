from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Type

class PredictionWriter(metaclass=ABCMeta):
    @abstractmethod
    def append_prediction(
        self,
        next_prediction: Tuple[
            str,
            Tuple[str, str],
            float,
            float,
            float,
            float,
            str
        ]
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

    @abstractmethod
    def save(self) -> None:
        """
        Save predictions
        """
        pass

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

    def add_writer(self, writer: Type[PredictionWriter]) -> None:
        """
        Add writer to prediction multi-writer

        Parameters
        ----------
            writer : Type[PredictionWriter]
                writer to add to prediction multi-writer
        """
        self.writers.append(writer)

    def append_prediction(
        self,
        next_prediction: Tuple[
            str,
            Tuple[str, str],
            float,
            float,
            float,
            float,
            str
        ]
    ) -> None:
        """
        Write prediction to all prediction writers in multi writer

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
        for writer in self.writers:
            writer.append_prediction(next_prediction)

    def save(self) -> None:
        """
        Save predictions to all writers in multi writer
        """
        for writer in self.writers:
            writer.save()
        
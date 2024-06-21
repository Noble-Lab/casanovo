from abc import ABCMeta, abstractmethod
from typing import Tuple

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
    
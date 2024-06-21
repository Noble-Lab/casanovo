from logging import Logger
from typing import Tuple

from .prediction_io import PredictionWriter

class LogPredictionWriter(PredictionWriter):
    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        self.predictions = {
            ""
        }

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
        Add new prediction to log writer context

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
        predicted_sequence = next_prediction[0]
        prediction_score = next_prediction[2]
    
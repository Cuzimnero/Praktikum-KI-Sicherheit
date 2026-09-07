import numpy as np
'Bewertungsmetriken'
def mean_absolute_error(y_true, y_predicted):
    """
        Berechnet den Mean Absolute Error (MAE) zwischen echten und vorhergesagten Werten

        Parameters
        ----------
        y_true : np.ndarray or list
            Die tatsächlichen Zielwerte (Ground Truth)
        y_predicted : np.ndarray or list
            Die vom Modell vorhergesagten Werte

        Returns
        -------
        float
            Der durchschnittliche absolute Fehler
        """
    return np.mean(np.abs(y_true - y_predicted))

def cumulative_score(y_true, y_predicted, tolerance = 1):
    """
        Berechnet den Cumulative Score (CS) innerhalb einer vorgegebenen Toleranzgrenze

        Gibt den relativen Anteil der Vorhersagen zurück, deren absoluter Fehler
        kleiner oder gleich der angegebenen Toleranz ist

        Parameters
        ----------
        y_true : np.ndarray or list
            Die tatsächlichen Zielwerte (Ground Truth)
        y_predicted : np.ndarray or list
            Die vom Modell vorhergesagten Werte
        tolerance : float, optional
            Die zulässige Abweichung, standard ist 1.0 (z. B. CS1)

        Returns
        -------
        float
            Der Anteil der Vorhersagen innerhalb der Toleranz
        """
    return np.mean(np.abs(y_true - y_predicted) <= tolerance)
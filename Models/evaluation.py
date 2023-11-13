from typing import List


def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """

    t_p = sum(e and a for e, a in zip(expected_results, actual_results))
    f_p = sum(e and not a for e, a in zip(expected_results, actual_results))
    f_n = sum(not e and a for e, a in zip(expected_results, actual_results))
    if (t_p == f_p == 0):
        raise Exception("TP and FP cannot be 0")
    precision = t_p/(t_p+f_p)
    if (t_p == f_n == 0):
        raise Exception("TP and FN cannot be 0")
    recall = t_p/(t_p+f_n)
    return precision, recall


def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    precision, recall = precision_recall(expected_results, actual_results)
    return (2*precision*recall/(precision+recall))

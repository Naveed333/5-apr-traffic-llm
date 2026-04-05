# prompt_builder.py — build all prompt components for TrafficLLM

from config import GRID_CELL_ID, LOCATION, L


def format_input(x, timestep):
    """
    Build p_input: natural-language description of the 24h input window.
    """
    hours = '  '.join(
        f'H{i+1:02d}: {v:.4f} MB' for i, v in enumerate(x)
    )
    return (
        f'At timestep {timestep}, Milan grid cell {GRID_CELL_ID}, '
        f'location {LOCATION},\n'
        f'past 24h traffic was:\n{hours}'
    )


def format_output(y):
    """
    Format a 24-value prediction/ground-truth as H01..H24 string.
    """
    return '  '.join(f'H{i+1:02d}: {v:.4f} MB' for i, v in enumerate(y))


def build_p_ques():
    """Build the prediction question prompt."""
    return (
        f'What is the predicted internet traffic for the next {L} hours?\n'
        f'Provide one value per hour from H01 to H{L:02d} in MB.\n'
        f'Format: H01: X MB  H02: X MB ... H{L:02d}: X MB'
    )


def build_p_feed_prompt(y_hat, y_true, mae, iteration):
    """
    Build the 4-question feedback prompt (verbatim from paper Section II-C).
    MAE is precomputed in Python and stated directly — not asked from the LLM.
    """
    gt_str   = format_output(y_true)
    pred_str = format_output(y_hat)

    q1 = (
        f'Q1: The Mean Absolute Error of the predictions is {mae:.4f} MB.\n'
        f'    Ground truth: {gt_str}\n'
        f'    Predictions:  {pred_str}'
    )
    q2 = (
        'Q2: For ground truths and predictions, what are their projected\n'
        '    functions derived from the combination of sine and cosine functions?'
    )
    q3 = (
        'Q3: Do the predictions align with the format of the ground truths\n'
        '    and provide a complete prediction for each timestamp?'
    )
    q4 = (
        'Q4: What is the prediction method applied in the current iteration?'
    )

    return (
        f'--- Feedback Questions (iteration {iteration}) ---\n'
        f'{q1}\n\n{q2}\n\n{q3}\n\n{q4}\n'
        f'Please answer all four questions thoroughly.'
    )


def build_p_refine(feedback_text, target_time=None):
    """
    Build the refinement instruction (paper Section II-D).
    """
    if target_time is None:
        target_time = 'the target time window'
    return (
        f'Please refine predictions on {target_time} based on the previous '
        f'thorough feedback. To enhance performance, the overall prediction '
        f'error should be decreased. The prediction should match the function '
        f'of the real traffic. The prediction should be complete for each '
        f'timestamp and match the format. More accurate numerical time series '
        f'prediction methods should be considered, including numerical methods, '
        f'hybrid methods such as Seasonal ARIMA and LSTM+ARIMA combinations.\n'
        f'Provide ONLY the refined predictions in the format: '
        f'H01: X MB  H02: X MB ... H{L:02d}: X MB'
    )


def build_validation_prompt():
    """Build the self-validation prompt."""
    return 'Please review the previous answers and find potential mistakes.'


def build_critique_prompt():
    """Build the self-correction prompt."""
    return 'Please correct the answers based on the identified mistakes.'

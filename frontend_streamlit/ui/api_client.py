from ui.errors import ApiError
from ui.groq_explainer import explain_market_analysis
from ui.local_inference import get_metadata, predict_locally
from ui.supabase_rest import get_history, insert_prediction


def predict(payload):
    prediction = predict_locally(payload)
    created_at = insert_prediction(prediction)
    return {**prediction, "created_at": created_at}

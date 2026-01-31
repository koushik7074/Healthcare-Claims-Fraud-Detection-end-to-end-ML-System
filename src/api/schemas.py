from pydantic import BaseModel

class ProviderFeatures(BaseModel):
    total_claims: float
    total_reimbursed: float
    avg_reimbursed: float
    avg_duration_gap: float
    pct_claimed_gt_admitted: float
    avg_cost_per_day: float
    age_avg: float
    pct_chronic: float
    has_inpatient: int

from sqlalchemy.orm import Session
from sqlalchemy import select
from models.clustering_run import ClusteringRun

def get_latest_cluster_run(session: Session, partner: str) -> ClusteringRun:
    """
    Fetch the latest clustering run for the given partner.
    """
    try:
        latest_run = session.execute(
            select(ClusteringRun)
            .filter(ClusteringRun.partner == partner)
            .order_by(ClusteringRun.created_at.desc())  # Order by created_at to get the latest
        ).scalars().first()

        if not latest_run:
            raise ValueError(f"No clustering run found for partner: {partner}")

        return latest_run
    except Exception as e:
        raise ValueError(f"Error fetching latest run for partner {partner}: {str(e)}")

"""Hegre face-dataset creation and management tool.

Commands:
    discover       Scan ground truth for identities and plan extraction.
    extract-faces  Run MTCNN face detection and produce face crops.
    review         Start the Flask review UI.
    enrich         Run stratum-hq enrichment on approved face crops.
    export         Export a gate-ready dataset to a target directory.
    catalog        List and manage dataset versions.
"""
__version__ = "0.1.0"

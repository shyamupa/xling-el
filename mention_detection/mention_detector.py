class MentionDetector:
    """
    Abstract class. All mention detectors should take text and return mentions from it.
    """
    def __init__(self):
        pass

    def get_provided_view(self) -> str:
        raise NotImplementedError

    def get_mentions_from_text(self, text):
        raise NotImplementedError

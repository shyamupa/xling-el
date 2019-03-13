from mention_detection.mention_detector import MentionDetector


class AllennlpNerAnnotator(MentionDetector):
    def get_provided_view(self) -> str:
        pass

    def __init__(self):
        super().__init__()

    def get_mentions_from_text(self, text):
        raise NotImplementedError

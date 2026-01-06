class ExperimentResult:
    def __init__(
        self,
        experiment_name: str,
        model_type: str,
        dataset_name: str,
        config: dict
    ):
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.config = config

        self.metrics = {}
        self.history = {}
        self.artifacts = {}

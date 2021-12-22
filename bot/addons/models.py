import typing
from typing import Any, Dict, List, Optional, Text, Type

import requests
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.constants import INTENT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from requests import exceptions

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class MultiLingualBertIntentClassifier(Component):
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return []

    defaults = {"url": "http://localhost:8000/classify"}

    supported_language_list = None

    not_supported_language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        pass

    def process(self, message: Message, **kwargs: Any) -> None:

        text = message.data.get("text")

        if not text:
            return

        url = self.component_config["url"]

        payload = {"text": text}

        try:
            r = requests.post(url, json=payload)
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        results = r.json()

        label = results["label"]
        label_ranking = results["label_ranking"]

        message.set(INTENT, label, add_to_output=True)
        message.set("intent_ranking", label_ranking, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""

        pass

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":

        if cached_component:
            return cached_component
        else:
            return cls(meta)

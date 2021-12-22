from typing import Any, Dict, List, Text

from iso639 import languages
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from textblob import TextBlob


class ActionDetectLanguage(Action):
    def name(self) -> Text:
        return "action_detect_language"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        text = tracker.latest_message.get("text")

        langcode = TextBlob(text).detect_language()

        langname = languages.get(alpha2=langcode).name
        langname = langname if "(" not in langname else langname.split(" ")[0]

        return [SlotSet("langcode", langcode), SlotSet("langname", langname)]

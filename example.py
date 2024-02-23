from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

key = "ee802142b6874f0da9596513b8a754e4"
endpoint = "https://donetconf.cognitiveservices.azure.com/"


def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint, credential=ta_credential
    )
    return text_analytics_client


def sentiment_analysis_example(client):
    documents = ["Hoy es un día horrible. Odio programar en python."]
    response = client.analyze_sentiment(documents=documents)[0]
    print("Sentimiento del documento: {}".format(response.sentiment))
    print(
        "Puntuaciones generales: positivo={0:.2f}; neutral={1:.2f}; negativo={2:.2f} \n".format(
            response.confidence_scores.positive,
            response.confidence_scores.neutral,
            response.confidence_scores.negative,
        )
    )


def language_detection_example(client):
    try:
        documents = ["Ce document est rédigé en Français."]
        response = client.detect_language(documents=documents, country_hint="us")[0]
        print("Idioma: ", response.primary_language.name)

    except Exception as err:
        print("Encountered exception. {}".format(err))


def key_phrase_extraction_example(client):
    try:
        documents = ["Mi perro es adorable. Le encanta jugar en el parque."]
        response = client.extract_key_phrases(documents=documents)[0]

        if not response.is_error:
            print("\tFrases clave:")
            for phrase in response.key_phrases:
                print("\t\t", phrase)
        else:
            print(response.id, response.error)

    except Exception as err:
        print("Encountered exception. {}".format(err))


def entity_recognition_example(client):
    try:
        documents = ["I had a wonderful trip to Seattle last week."]
        result = client.recognize_entities(documents=documents)[0]

        print("Named Entities:\n")
        for entity in result.entities:
            print(
                "\tText: \t",
                entity.text,
                "\tCategory: \t",
                entity.category,
                "\tSubCategory: \t",
                entity.subcategory,
                "\n\tConfidence Score: \t",
                round(entity.confidence_score, 2),
                "\n",
            )

    except Exception as err:
        print("Encountered exception. {}".format(err))


client = authenticate_client()
sentiment_analysis_example(client)

language_detection_example(client)

key_phrase_extraction_example(client)

entity_recognition_example(client)

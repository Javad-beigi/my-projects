from transformers import pipeline


class LanguageAPI:
    """
    A class that translate language and
    detect idiom.
    Attributes:
    en_to_es (transformers.pipeline): translate english to spanish.
    en_to_zh (transformers.pipeline): translate english to chinese.
    idiom_detector (transformers.pipeline): detect idioms in english text.

    """
    def __init__(self):
        """
        initialize the language API class by loading
        language models.

        """
        self.en_to_es = pipeline('translation' , model='Helsinki-NLP/opus-mt-en-es' )
        self.en_to_zh = pipeline('translation' , model='Helsinki-NLP/opus-mt-en-zh')
        self.idiom_detector = pipeline('text-classification', model='nlptown/idiom-detector')

    def translate(self, text, target_language):
        """
        translate input text to specified target language


        Args:
            text (str): 
            target_language (str):

        Returns:
            str: translated text or a message only spanish and chinese translations are available.

        """
        if target_language == 'es':
            return self.en_to_es(text)[0]['translation_text']
        elif target_language == 'zh':
            return self.en_to_zh(text)[0]['translation_text']
        else:
            return 'Sorry i can only translate to spanish or chinese at the moment.'
    
    def detect_idioms(self, text):
        """
        detect idioms in input text.

        Args:
            text (str):

        Returns:
            list: a list of detected idioms.

        """
        results = self.idiom_detector(text)
        idioms = [result['label'] for result in results if result['label'] != 'not_idiom']
        return idioms
    
        
if __name__ == '__main__':
        
        """
        Example usage of languageAPI class.
        """
        api = LanguageAPI()

        input_text = "It's raining cats and dogs outside!"
        translated_text = api.translate(input_text, 'es')
        detected_idioms = api.detect_idioms(input_text)


        print(f"original text : {input_text}")
        print(f"translated to spanish: {translated_text}")
        print(f"idioms detected: {', '.join(detected_idioms) }")
        





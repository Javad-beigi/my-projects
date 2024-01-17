from wordcloud import WordCloud


class WordCloudGenerator:
    def __init__(self, file_path: str):
        with open(file_path) as f:
            self.text = f.read()

    def run(self, output_path: str, **kwargs) -> None:
        """
        Generate a word cloud image.

        :param output_path: The path to the output file.
        """
        WordCloud(**kwargs).generate(self.text).to_file(output_path)


if __name__ == '__main__':
    import sys
    #input_file = sys.argv[1]
    #output_file = sys.argv[2]

    wc_gen = WordCloudGenerator('data/movies.txt')
    wc_gen.run(
        'output.png',
        width=600, height=400,
        background_color='white',
    )

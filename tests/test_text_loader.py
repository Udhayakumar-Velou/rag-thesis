from loaders.text_loader import TextLoader

loader = TextLoader()

text = loader.load("documents/sample.txt")

print(text)
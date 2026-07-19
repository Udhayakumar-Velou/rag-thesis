from llm.gemini import GeminiLLM


def test_gemini():
    llm = GeminiLLM()

    response = llm.generate(
        "Explain Retrieval-Augmented Generation in two sentences."
    )

    print("\nGemini Response:\n")
    print(response)


if __name__ == "__main__":
    test_gemini()
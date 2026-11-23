
from generation.llm_client import OllamaLLM


def main():

    llm = OllamaLLM()

    prompt = """
    Explain Docker in simple terms.
    Give your answer in 3 bullet points.
    """

    answer = llm.generate(prompt)

    print("\nGenerated Answer:")
    print(answer)


if __name__ == "__main__":
    main()

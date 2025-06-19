from abc import ABC, abstractmethod

# Abstract base class for all LLMs
class LLM(ABC):
    @abstractmethod
    def complete_sentence(self, prompt):
        pass

# Concrete implementations of LLM
class OpenAI(LLM):
    def complete_sentence(self, prompt):
        return prompt + " ... OpenAI end of sentence."

class Anthropic(LLM):
    def complete_sentence(self, prompt):
        return prompt + " ... Anthropic end of sentence."

class GooglePaLM(LLM):
    def complete_sentence(self, prompt):
        return prompt + " ... Google PaLM end of sentence."

class Cohere(LLM):
    def complete_sentence(self, prompt):
        return prompt + " ... Cohere end of sentence."

class Mistral(LLM):
    def complete_sentence(self, prompt):
        return prompt + " ... Mistral end of sentence."

class CustomLLM(LLM):
    def complete_sentence(self, prompt):
        suffix = " ... CustomLLM generated response."
        return prompt + suffix

# Test function to run all LLMs
def test_llms():
    prompt = "The future of AI is"
    llms = [OpenAI(), Anthropic(), GooglePaLM(), Cohere(), Mistral(), CustomLLM()]
    for llm in llms:
        print(f"{llm.__class__.__name__}: {llm.complete_sentence(prompt)}")

# Only run if script is executed directly
if __name__ == "__main__":
    test_llms()
